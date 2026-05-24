from langchain_ollama import OllamaEmbeddings, OllamaLLM                                                                            
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate                                                                                   
from langchain_core.output_parsers import StrOutputParser 
from src.config import CHROMA_DIR, EMBED_MODEL, LLM_MODEL                                                                                                   

PROMPT_TEMPLATE = """You are a precise assistant that answers questions using only the reference passages below.

Reference passages:
{context}

Rules:
1. Answer using ONLY information found in the reference passages above.
2. If the passages contain partial information, use what is there and note what is missing.
3. Quote or closely paraphrase the relevant passage when possible — this keeps your answer grounded.
4. Say "The reference documents do not contain information about this topic" ONLY if there is truly zero relevant content.
5. Do not add facts, opinions, or context from outside the passages.
6. Keep your answer concise and factual.

Question: {question}

Answer:"""

def ask(question: str) -> dict:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL) #no writing in DB, just build retriever 和 chain
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)                                                        
    retriever = db.as_retriever(search_kwargs={"k": 5}) #return top-5 relevant file，fill in the below {context}. Improving steps.
    llm = OllamaLLM(model=LLM_MODEL)                                                                                                
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
                                                                                                                                    
    # Retrieve once; reuse docs for both context injection and source attribution
    source_docs = retriever.invoke(question) #pure vector similarity search. Takes the question, converts it to an embedding, finds the 3 closest chunks in Chroma, returns them as a list of Document, including page_content 和 metadata息
    context = "\n\n".join(doc.page_content for doc in source_docs)

    chain = (
        prompt  # 接收 {"context": ..., "question": ...} 
        | llm
        | StrOutputParser() 

    answer = chain.invoke({"context": context, "question": question}) #
    sources = list({doc.metadata.get("source", "未知") for doc in source_docs})                                                     
    return {"answer": answer, "sources": sources}  
