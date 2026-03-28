# 📄 AI Research Assistant for Scientific Papers
![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![API](https://img.shields.io/badge/API-Google%20Gemini-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-8A2BE2)


A Retrieval-Augmented Generation (RAG) system designed to analyze and compare complex scientific literature at scale.

Unlike standard RAG pipelines that suffer from context fragmentation, this system implements a hybrid retrieval-routing architecture that dynamically switches between vector search and full-context reasoning—enabling accurate cross-document synthesis across large PDFs.

Built to explore how retrieval strategies affect reasoning quality in long-context LLM applications.

---

## ⚠️ Vector Search & Context Handling
When asking an AI to compare two massive PDFs using standard RAG architectures, it often suffers from "semantic drowning"—grabbing tiny, disconnected chunks of text and missing the overarching themes.

To solve this, the application implements a Hybrid Routing Architecture. For specific questions, the app uses standard FAISS similarity search to save tokens and processing time. However, for full document comparisons, it dynamically bypasses the vector database and feeds the entire text directly into Gemini's massive 1-million token context window, resulting in a much more accurate and unbiased comparison.


---

## 🎥 Demo


**🌍 [View Live AI Research Assistant Here](https://ai-flashcard-app-bpxmr9lmafudmkm2r7yuao.streamlit.app/)**

![Demo GIF](/screenshots/demo.gif) 

⚡ Best experienced with 2–3 research papers for comparison mode.

---


## 🚀 Features

- **Multi-Document Upload:** Users can drag and drop several unstructured PDFs at once, which are automatically chunked and loaded into a single vector space.
- **Smart Citations:** A custom function bakes the original file names and page numbers into the hidden metadata of every chunk. When the AI generates an answer, it provides exact citations showing where the information was retrieved.
- **Hybrid RAG Router:** The application intelligently switches between Vector Search (for targeted Q&A) and Direct Context Injection (for deep multi-paper comparisons).
- **Efficient Caching:** By utilizing Streamlit's `@st.cache_resource`, the FAISS database is stored in memory. This prevents the app from re-embedding documents on every query, drastically speeding up response times and preventing API rate-limit errors (429s).
- **Error Handling:** Built-in session state checks gracefully warn the user to upload a document before attempting to query the AI.

---
## 🌍 Why This Matters

Standard RAG systems degrade in quality when applied to long, multi-document reasoning tasks. Chunk-based retrieval often fragments context, leading to incomplete or biased outputs during cross-document analysis.

This system addresses that limitation through a hybrid routing strategy that selectively bypasses retrieval when global context is required.

As a result, it:
- Preserves document-level coherence during full-text analysis  
- Reduces retrieval-induced bias from partial context exposure  
- Improves reliability in comparative reasoning across multiple sources  

Working on this made it clear to me that model capability alone isn’t enough—how context is constructed, routed, and presented plays a huge role in the final output.

---
## 💬 Example Queries

Users can ask questions such as:

• "Compare the methodologies used in the uploaded papers."  
• "What dataset was used in this study?"  
• "Summarize the results section of the second paper."  
• "What are the key limitations mentioned in the research?"

---

## 🎯 Skills Demonstrated

This project served as a practical application of several core software and AI concepts:

* Building a RAG (Retrieval-Augmented Generation) pipeline from scratch
* Integrating and querying Vector Databases (FAISS)
* Prompt Engineering and managing LLM context windows
* Handling API rate limits and optimizing memory with caching
* Modular system design (separating the frontend UI from the backend AI logic)
* Interactive Frontend Development in Python using Streamlit

---

## 🛠 Tech Stack

**Frontend**
- Streamlit (Reactive web UI and session state management)

**Backend**
- Python 3.14
- LangChain (Data pipelining, chunking, and retrieval logic)
- FAISS (Local dense vector indexing)
- PyPDF (Unstructured document parsing)

**APIs / Models**
- Google Gemini 2.5 Flash (Generative LLM)
- Google Gemini Embedding 2 Preview (Text-to-vector embedding)
- `python-dotenv` (Environment variable management)

---

## 📸 Screenshots

### 1. Multi-Document Ingestion
*(Drag-and-drop interface for processing complex PDFs into the FAISS index)*

![Multi-Document Ingestion Screenshot](/screenshots/screenshot1.gif)

### 2. Citation-Backed Q&A
*(Targeted answers featuring dynamically extracted page numbers and file names)*

![Citation-Backed Q&A Screenshot](/screenshots/screenshot2.gif)

### 3. Hybrid Document Comparison
*(Using the bypass router to synthesize methodologies across multiple papers)*

![Hybrid Document Comparison Screenshot](/screenshots/screenshot3.gif)

---

## ⚙️ How It Works

1. **Ingestion:** User uploads multiple PDFs. The `PyPDFLoader` reads the text, and `RecursiveCharacterTextSplitter` chops it into overlapping 1000-character chunks.
2. **Metadata Tagging:** The Python logic silently bakes the source filename into both the hidden metadata dictionary and the raw text of every chunk.
3. **Embedding:** The chunks are sent to the Gemini Embedding API, converted into high-dimensional vectors, and stored in a local FAISS database.
4. **Routing:** 
   * If the user types a question, it queries FAISS for the top 12 most relevant chunks.
   * If the user clicks "Compare Papers," the router bypasses FAISS, dumps all raw text from memory, and sends it directly to the LLM.
5. **Generation & Citation:** The LangChain pipeline passes the context to Gemini 2.5 Flash, formats the output, extracts the metadata to build a "Sources" list, and renders it in the Streamlit UI.

---

## ⚠️ Limitations

- Full-context mode is memory-intensive and may not scale for extremely large document sets
- Performance depends on the quality of PDF parsing (complex layouts may degrade results)
- No persistent storage (FAISS index resets on app restart)

Future iterations will address these through distributed vector storage and improved document parsing.

---

## 🧠 System Architecture

This application adheres to a strict modular design, separating the backend AI engine from the frontend presentation:
```text
 User Input 
     │
     ▼
[ Streamlit UI ] ──(Query or Action)──► [ Hybrid Router ]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
             (Specific Q&A)                                    (Compare Papers)
                    ▼                                                   ▼
         [ FAISS Vector Search ]                             [ Full Context Extraction ]
         Retrieves Top-K Chunks                              Bypasses DB, Loads All Text
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                   [ Gemini 2.5 Flash LLM ]
                                              │
                                              ▼
                                [ Answer + Precise Citations ]
```
* **Ingestion Layer (`ingestion.py`):** Handles PDF parsing, chunking, and filename injection.
* **Vector Store (`vector_store.py`):** Manages the FAISS database and embedding model initialization.
* **Retrieval & Prompts (`retrieval.py` & `prompts.py`):** Defines the LangChain retrieval pipeline, system instructions, and LLM instantiation.
* **Frontend UI (`streamlit_app.py`):** A Streamlit application handling routing, state caching, and chat message rendering.

---

## 💻 Local Installation
1. Clone the repository and create a virtual environment:
    ```bash
    git clone https://github.com/AayushWaney/AI-Research-Assistant.git
    cd AI-Research-Assistant
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` includes: `streamlit langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv`)*
3. Set up your Environment Variables:
Create a `.env` file in the root directory and add your Google API key:
    ```bash
    GOOGLE_API_KEY="your_api_key_here"
    ```
4. Boot the Streamlit Server:
    ```bash
    streamlit run ui/streamlit_app.py
    ```
5. Access the dashboard at:
    ```bash
    http://localhost:8501
    ```

---

## 🔮 Future Improvements
* Integrate cloud vector databases (like Pinecone or Qdrant) for persistent storage across sessions.
* Add support for parsing `.docx`, `.txt`, and direct URL webpage scraping.
* Implement conversational memory so the AI remembers previous chat messages during a session.

---

## 📁 Project Structure
```text
ai-research-assistant/
│
├── requirements.txt            # Python dependencies
├── .env                        # Local environment variables (Git ignored)
│
├── rag_engine/                 # Backend AI Logic
│   ├── __init__.py
│   ├── ingestion.py            # Document loading and chunking
│   ├── prompts.py              # System instructions and templates
│   ├── retrieval.py            # LangChain Q&A pipelines
│   └── vector_store.py         # FAISS database initialization
│
└── ui/                
    └── streamlit_app.py        # Streamlit frontend and Hybrid Router
```
---
## 👨‍💻 Author
Aayush Waney  
B.Tech – Metallurgical Engineering  
VNIT Nagpur

GitHub: https://github.com/AayushWaney

---

 ## 📄 License
This project is released for educational and portfolio purposes.

---