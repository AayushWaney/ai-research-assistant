import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_files(uploaded_files):
    all_chunks = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)

        # Fix metadata and bake filenames into the text
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
            chunk.page_content = f"[Source Document: {uploaded_file.name}]\n" + chunk.page_content

        all_chunks.extend(chunks)

    return all_chunks