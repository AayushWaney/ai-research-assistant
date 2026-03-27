import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Let Python know where to find our rag_engine folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.ingestion import load_and_chunk_files
from rag_engine.vector_store import get_vector_store
from rag_engine.retrieval import get_rag_chain

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")
st.title("📄 AI Research Assistant for Scientific Papers")
st.caption("Powered by Hybrid RAG (FAISS + Full Context Routing)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# Cached Database Creation
@st.cache_resource(show_spinner="Building Vector Database...")
def create_cached_db(_chunks, file_names):
    return get_vector_store(_chunks)


preset_prompt = None

# Sidebar
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload scientific papers (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process Papers"):
        with st.spinner("Reading, chunking, and embedding all documents..."):
            all_chunks = load_and_chunk_files(uploaded_files)

            # Extract the names of the currently uploaded files
            current_files = ", ".join([f.name for f in uploaded_files])

            # Store raw docs safely in session state to avoid hacking FAISS later
            st.session_state.all_chunks = all_chunks

            # Pass the names into the cache.
            st.session_state.vector_db = create_cached_db(all_chunks, current_files)

            st.success(f"Successfully processed {len(uploaded_files)} papers! You can now compare them.")

    if st.session_state.vector_db:
        st.divider()
        st.header("2. Quick Actions")

        if st.button("📝 Summarize Paper(s)"):
            preset_prompt = "Provide a comprehensive summary of the main objectives, findings, and conclusions of the provided text."

        if st.button("🔬 Extract Methodology"):
            preset_prompt = "Explain the methodology, experimental setup, and techniques used. Detail how the study was conducted."

        if st.button("📊 Extract Results"):
            preset_prompt = "What are the key quantitative and qualitative results, data points, and final outcomes presented?"

        # THE COMPARE BUTTON
        if st.button("⚖️ Compare Papers"):
            preset_prompt = "COMPARE_ALL_DOCUMENTS"


# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the papers...")
prompt = preset_prompt or user_input

if prompt:
    if not st.session_state.vector_db:
        st.warning("Please upload and process a PDF in the sidebar first.")
    else:

        display_prompt = "Compare all the provided papers." if prompt == "COMPARE_ALL_DOCUMENTS" else prompt

        st.session_state.messages.append({"role": "user", "content": display_prompt})
        with st.chat_message("user"):
            st.markdown(display_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):

                # THE HYBRID ROUTER
                if prompt == "COMPARE_ALL_DOCUMENTS":
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

                    # Use the safely stored chunks instead of accessing private _dict
                    full_text = "\n\n".join([doc.page_content for doc in st.session_state.all_chunks])
                    comparison_prompt = f"You are an AI Research Assistant. Read the following text extracted from multiple uploaded documents. Write a comprehensive comparison of their approaches, methodologies, and findings:\n\n{full_text}"

                    # Implement try/except block for API stability
                    try:
                        response = llm.invoke(comparison_prompt)
                        answer = response.content
                    except Exception as e:
                        answer = f"⚠️ API Error occurred: {str(e)}\n\nPlease wait 60 seconds and try again."

                    st.markdown(answer)

                else:
                    # NORMAL RAG: Use the vector database for specific Q&A
                    rag_chain = get_rag_chain(st.session_state.vector_db)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]

                    # Extract Citations
                    citations = []
                    for doc in response["context"]:
                        raw_source = doc.metadata.get("source", "Unknown Document")
                        filename = os.path.basename(raw_source)
                        page_num = doc.metadata.get("page", 0) + 1
                        citations.append(f"{filename} - Page {page_num}")

                    unique_citations = list(set(citations))
                    if unique_citations:
                        citation_text = "\n\n**Sources:**\n"
                        for citation in unique_citations:
                            citation_text += f"* {citation}\n"
                        answer += citation_text

                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})