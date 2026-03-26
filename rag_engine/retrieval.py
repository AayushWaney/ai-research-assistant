from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from .prompts import get_rag_prompt


def get_rag_chain(vector_db):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt_template = get_rag_prompt()

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

    # FIX: Use MMR to ensure diverse context and avoid token exhaustion
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50}
    )

    return create_retrieval_chain(retriever, question_answer_chain)