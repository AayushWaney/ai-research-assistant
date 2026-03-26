from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt():
    system_prompt = (
        "You are an expert scientific research analyst. "
        "Your task is to synthesize information across multiple academic papers. "
        "Prioritize methodological differences, datasets, and experimental results. "
        "Do not generalize—be precise and comparative.\n\n"
        "If the answer is not in the context, say 'I cannot find the answer in the document.' "
        "Do not make up information.\n\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])