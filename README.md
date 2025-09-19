# 📄 DocuMate AI – Multilingual Document Chatbot 🤖🌍

DocuMate AI is an **AI-powered document chatbot** that allows users to upload PDF files and interact with them conversationally.  
It supports **multilingual documents**, enabling seamless question answering across different languages.

---

## 🚀 Features
- 📂 Upload and process PDF documents  
- 💬 Ask natural language questions and get precise answers  
- 🌍 **Multilingual support** – works with:  
  - English  
  - French  
  - Arabic  
  - Spanish  
  - Portuguese  
  - Hindi  
  - Punjabi  
  - Mandarin Chinese  
  - Urdu  
- 🧠 Powered by **embeddings + LLMs** for semantic understanding  
- ⚡ Backend with FastAPI and frontend with HTML/JS  
- 🗄️ MongoDB integration for document storage  

---

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)  
- **Frontend:** HTML, CSS, JavaScript  
- **Database:** MongoDB  
- **AI Models:** Sentence Transformers + Ollama LLM  

---

## ⚡ Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Backend
```bash
uvicorn api:app --reload --port 9082
```

### 5. Open the Frontend
Open `frontend/index.html` in your browser.

---

## 📌 Roadmap
- [ ] Add Docker support for easier deployment  
- [ ] Deploy backend on Render / Railway  
- [ ] Deploy frontend on Netlify / Vercel  
- [ ] Expand multilingual support with additional languages  

---

## 👨‍💻 Author
Developed by **Mourish Kumar** ✨  
If you like this project, don’t forget to ⭐ star the repo!  
