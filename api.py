import os
import cv2
import os
import uuid
import json
import easyocr
import chromadb
from tqdm import tqdm
from typing import Optional
import config as cfg
from threading import Thread
from collections import deque
from pydantic import BaseModel
from pymongo import MongoClient
from lingua import Language, LanguageDetectorBuilder
from agents import graph_agent, conversational_agent, document_summary_agent
from utils import get_embeds, extract_text_from_file, ollama_llm, extract_json, remove_chinese_characters, extract_thought, ocr

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form

print("""
 /$$$$$$$                       /$$$$$$  /$$                   /$$    
| $$__  $$                     /$$__  $$| $$                  | $$    
| $$  \ $$  /$$$$$$   /$$$$$$$| $$  \__/| $$$$$$$   /$$$$$$  /$$$$$$  
| $$  | $$ /$$__  $$ /$$_____/| $$      | $$__  $$ |____  $$|_  $$_/  
| $$  | $$| $$  \ $$| $$      | $$      | $$  \ $$  /$$$$$$$  | $$    
| $$  | $$| $$  | $$| $$      | $$    $$| $$  | $$ /$$__  $$  | $$ /$$
| $$$$$$$/|  $$$$$$/|  $$$$$$$|  $$$$$$/| $$  | $$|  $$$$$$$  |  $$$$/
|_______/  \______/  \_______/ \______/ |__/  |__/ \_______/   \___/  
""")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

linuga2lang = {Language.ENGLISH: "English", Language.FRENCH: "French", Language.ARABIC: "Arab", Language.SPANISH: "Spanish", Language.PORTUGUESE: "Portuguese", Language.HINDI: "Hindi", Language.PUNJABI: "Punjabi", Language.CHINESE: "Mandarin Chinese", Language.URDU: "Urdu"}

detector = LanguageDetectorBuilder.from_languages(*linuga2lang.keys()).build()

@app.get("/")
async def read_index():
    """
    This endpoint serves your main index.html file.
    It's the entry point for users visiting your IP address.
    """
    return FileResponse('frontend/index.html')

batch_size = 12
client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = client.get_or_create_collection(name="documents")
mongo_client = MongoClient(cfg.MONGO_URI)
db = mongo_client["agentic"]
chat_collection = db["chats"]
os.makedirs('documents', exist_ok=True)

def text2lang(text):
    if "hello" in text:
        return "English"
    lang = linuga2lang[detector.detect_language_of(text)]
    if lang == None:
        return "English"
    return linuga2lang[detector.detect_language_of(text)]

class DocumentStatus(BaseModel):
    document_id: str
    status: str  # 'processing', 'completed', 'failed'
    progress: float = 0.0

class ChatQuery(BaseModel):
    user_id: str
    document_id: Optional[str] = None
    query: str
    
document_status = {}

def update_document_status(user_id, document_id, status, progress):
    if user_id not in document_status:
        status = DocumentStatus(
            document_id=document_id,
            status=status,
            progress = progress
        ).__dict__
        document_status[user_id] = {document_id: status}
    else:
        document_status[user_id][document_id] = status

for user_id in os.listdir('documents'):
    for file in os.listdir(os.path.join('documents', user_id)):
        document_id = file.split('_')[0]
        file_name = file.split('_')[1]
        update_document_status(user_id, document_id, 'completed', 1.0)

def generate_document_summary(pages, filename):
    prompt = document_summary_agent('qwen', filename, pages)

    json_payload = {
        'model': 'qwen3:1.7b',
        'prompt': prompt
    }

    llm_response=''
    for chunk in ollama_llm(f"{cfg.OLLAMA_HOST}/api/generate", json_payload, None, None, True):
        data = json.loads(chunk)['response']
        llm_response += data
        print(data, end='', flush=True)
    return llm_response

def generate_graph(page, filename, summary, attempts=10):
    success=False
    output = []
    thought = ''
    for _ in range(attempts):

        if success == True:
            break
        try:
            prompt = graph_agent('qwen', filename, page, summary)
            json_payload = {
                "model":  "qwen3:1.7b",
                "prompt": prompt
            }

            output = ''
            for chunk in ollama_llm(f"{cfg.OLLAMA_HOST}/api/generate", json_payload, None, None, True):
                data = json.loads(chunk)['response']
                print(data, end='', flush=True)
                output += data
            graph = extract_json(output)
            thought = extract_thought(output)
            if isinstance(graph, dict):
                print('\n\n>>> Graph:', graph)
                success = True
            else:
                print('\n\n>>> Error occurred, retrying')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('\33[31m')
            print(e)
            print('\33[0m')
            pass
    if isinstance(graph, dict):
        output = [json.dumps({k:{"value": v, "thought": thought}}, ensure_ascii=False) for k,v in graph.items()]
    else:
        output = [json.dumps({"graph": output}, ensure_ascii=False)]
    return output

def generate_graph_subprocess(user_id:str, document_id:str, filepath:str):
    texts = extract_text_from_file(filepath)
    if len(texts) > 0:
        summary = generate_document_summary(texts, filepath)
        for n, page in enumerate(tqdm(texts, desc='Page')):
            print('>>> Page:', page)
            graph = generate_graph(page, filepath, summary)
            try:
                if graph != [] and graph is not None:
                    ingest_graph(graph, document_id, filepath)
                update_document_status(user_id, document_id, 'processing', float(n/len(texts)))
            except Exception as e:
                print(e)
        update_document_status(user_id, document_id, 'completed', 1.0)

def ingest_graph(graph, document_id, filename):
    for i in tqdm(range(0, len(graph), batch_size), desc='Ingesting graph'):
        batch = graph[i: i+batch_size]
        embeddings = get_embeds(batch)
        collection.add(
                ids=[f'{document_id}_{str(uuid.uuid4())}' for _ in range(len(batch))],
                documents=batch,
                embeddings=embeddings,
                metadatas=[{"document_id": document_id, "document_name": filename} for _ in range(len(batch))]
            )

def query_graph(query, document_id, topk=5, similarity_threshold=0.3):    
    result = collection.query(
        query_embeddings=get_embeds(query),
        n_results=topk,
        where={"document_id": document_id} if document_id else None
          
    )
    distances = result['distances'][0]  # cosine distances
    similarities = [1 - d for d in distances]
    print("\nSimilarities:\n", similarities)
    print(result['documents'][0])
    return result['documents'][0]


def query_graph_branched(start_query, document_id, topk=5, max_levels=2):
    visited = set()
    result_nodes = []
    
    queue = deque([(start_query, 0)])

    while queue:
        query, level = queue.popleft()

        if level > max_levels:
            break

        results = query_graph(query, document_id, topk=topk)

        for res_str in results:
            if res_str in visited:
                continue
            visited.add(res_str)
            result_nodes.append(res_str)

            try:
                res_obj = json.loads(res_str)
            except json.JSONDecodeError:
                continue  # skip non-JSON results

            if isinstance(res_obj, dict):
                for val in res_obj.values():
                    if isinstance(val, str):
                        queue.append((val, level + 1))
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                queue.append((item, level + 1))
    result_nodes = list(set(result_nodes))
    result_nodes = [json.loads(r) for r in result_nodes]
    return result_nodes

@app.post("/ocr/")
async def ocr(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    filename = file.filename.replace(' ', '_')
        
    os.makedirs(f'documents/{user_id}', exist_ok=True)
    filepath = f"documents/{user_id}/{document_id}_{filename}"

    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    data = ocr(filepath)
    return data

@app.post("/upload/")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        document_id = str(uuid.uuid4())
        filename = file.filename.replace(' ', '_')
        
        os.makedirs(f'documents/{user_id}', exist_ok=True)
        filepath = f"documents/{user_id}/{document_id}_{filename}"

        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        
        Thread(target=generate_graph_subprocess, args=(user_id, document_id, filepath)).start()
        update_document_status(user_id, document_id, 'processing', 0.0)

        print(document_status)
        return {"document_id": document_id, "filename": filename}
    except Exception as e:
        print(e)
        return {"error": e}
    
@app.get("/documents/{user_id}")
async def list_documents(user_id: str):
    try:
        id2filename = {path.split('_')[0]: "_".join(path.split('_')[1:]) for path in os.listdir(f'documents/{user_id}')}
        return id2filename
    except:
        return {}
    
@app.post("/embed/")
async def embed(request: dict):
    sentence = request['sentence']
    try:
        embeddings = get_embeds(sentence)
        print(type(embeddings))
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def remove_key(obj, key):
    if isinstance(obj, dict):
        keys_to_delete = [key for key in obj if key == key]
        for key in keys_to_delete:
            del obj[key]
        for key, value in obj.items():
            remove_key(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_key(item)
    return obj

@app.post("/chat/stream/")
def chat_stream(request: ChatQuery):
    user_id = request.user_id
    document_id = request.document_id
    query = request.query
    print(query)

    language = text2lang(query)

    if user_id not in document_status:
        raise Exception(f"Invalid user id '{user_id}'")
    
    if document_id != None and document_id not in document_status[user_id]:
        raise Exception(f"Invalid document id '{document_id}'")

    def response_generator(user_id):
        if query == 'reset':
            chat_collection.delete_one({"user_id": user_id})
            yield 'history reset\n'
            return
        try:
            history = chat_collection.find_one({"user_id": user_id})["history"]
        except Exception as e:
            print('Error:', e)
            history = []
        history.append({"role": "user", "content": query})
        chat_collection.update_one(
            {"user_id": user_id},
            {"$push": {"history": {"role": "user", "content": query}}},
            upsert=True
        )
        knowledge = query_graph_branched(query, document_id)
        if not isinstance(knowledge, list):
            knowledge = [{k:item[k]['value'] for k in item} for item in knowledge]
        prompt = conversational_agent('qwen', knowledge, history[-10:], language)
        json_payload = {"prompt": prompt, "model": 'qwen3:1.7b'}
        llm_response = ''
        thought = True
        for chunk in ollama_llm(f"{cfg.OLLAMA_HOST}/api/generate", json_payload, None, None, True):
            if not thought:
                data = json.loads(chunk.replace('</think>', ''))['response']
                if not language == 'Mandarin Chinese':
                    data = remove_chinese_characters(data)
                llm_response += data
                yield data
            if '</think>' in chunk:
                thought = False
        history.append({"role": "assistant", "content": llm_response})
        chat_collection.update_one(
            {"user_id": user_id},
            {"$push": {"history": {"role": "assistant", "content": llm_response}}},
            upsert=True
        )
    return StreamingResponse(response_generator(user_id), media_type="text/plain") 

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9082)
