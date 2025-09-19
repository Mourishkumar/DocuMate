import requests
import time
import os

BASE_URL = "http://localhost:9082"
document_id = False
user_id = '4'
if __name__ == "__main__":
    while True:
        query = input('>>> ')    
        chat_payload = {
            "user_id": user_id,
            # "document_id": document_id,
            "query": query,
            "language": 'en'
        }

        with requests.post(f"{BASE_URL}/chat/stream/", json=chat_payload, stream=True) as response:
            if response.status_code == 200:
                print("--- ", end="", flush=True)
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        print(chunk.decode('utf-8'), end='', flush=True)
                print()  # Newline after stream ends
            else:
                print("Error:", response.status_code, response.text)