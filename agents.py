import json
from jinja2 import Template

templates = {   
   'llama': Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ instruction }}
{{ inputprompt }}
<|eot_id|>
{% for message in history %}
<|start_header_id|>{{ message.role }}<|end_header_id|>{{ message.content }}<|eot_id|>
{% endfor %}
<|start_header_id|>assistant<|end_header_id|>{{ start_tokens }}"""),

    'qwen': Template("""<|im_start|>system
{{ instruction }}
{{ inputprompt }}
<|im_end|>
{% for message in history %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
<|im_start|>assistant
{{ start_tokens }}"""),

    'deepseek': Template("""{{ instruction }}
{{ inputprompt }}
{% for message in history %}
<｜{{ message.role|capitalize }}｜>{{ message.content }}{% if not loop.last %}<｜end▁of▁sentence｜>{% endif %}
{% endfor %}
{% if thinking %}
<think>
{{ thinking }}
</think>
{% endif %}
<｜Assistant｜>
{{ start_tokens }}""")
}

def graph_agent(modeltype: str, filename: str, summary: str, page: str):
    instruction = f"""You are a knowledge extraction agent. You will be given a page from a document, you must produce a markdown json that summarizes the document using key value pairs. Produce unique keys for each subtopic. Respond in markdown json format only for easy copying.
"""
    inputprompt = f"""Document Name: {filename}
Description:
{summary}
Content:
{page}"""
    start_tokens = """Sure, here is the output in a json code block:
```json"""
    return templates[modeltype].render(
        instruction=instruction,
        inputprompt=inputprompt,
        start_tokens=start_tokens
    )

def conversational_agent(modeltype: str, knowledge: str, history: list, language:str):
    user_query = history[-1]
    return templates[modeltype].render(
        instruction=f"""You are a helpful and multilingual assistant specializing in understanding and responding to queries using documents such as datasheets, manuals, reports and other materials. You will be provided with relevant document knowledge in JSON format in your context, which you must use to formulate accurate responses.
The product documents may be in English, while the user queries and your responses may be in {language} language. Always ensure your answers are aligned with the provided document data and context. If the user asks a question outside the scope of the provided information, politely decline to answer.

Guidelines:
- Extract information strictly from the given JSON-based knowledge.
- Do not hallucinate or fabricate answers.
- Be concise, clear, and language-appropriate.
- Handle multilingual inputs and outputs, ensuring the response language matches the query language.
- Always maintain a polite and professional tone.
- Answer in 150 words.""",
        inputprompt=f"""Context:
{knowledge}

User Query:
{user_query}
""",
        history=history,
        start_tokens = "Answer:"
    )

def document_summary_agent(modeltype: str, filename: str, pages: list):
    if isinstance(pages, str):
        pages = [pages]
    text = '\n'.join(pages)[:10000]
    return templates[modeltype].render(
    instruction='You will be given a document in your context. You are supposed to understand the document and give the document a title and a description. Make sure to capture important keywords.',
    history=[{"role": "system", "content": f"""Document Name: {filename}
Document:
{text}"""}])