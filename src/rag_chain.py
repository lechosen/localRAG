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
    embeddings = OllamaEmbeddings(model=EMBED_MODEL) #不作写入的动作，直接用来构建 retriever 和 chain
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)                                                        
    retriever = db.as_retriever(search_kwargs={"k": 5}) #这里设置每次检索返回最相关的5个文档块，填入下面的{context}。后续可以改成更复杂的检索策略，比如先用一个轻量级的模型来筛选出相关文档，再用更强大的模型来生成答案；或者根据问题的类型来动态调整 k 的值，比如如果是需要具体事实的提问就增加 k 的值，如果是需要总结归纳的提问就减少 k 的值
    llm = OllamaLLM(model=LLM_MODEL)                                                                                                
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
                                                                                                                                    
    # Retrieve once; reuse docs for both context injection and source attribution
    source_docs = retriever.invoke(question) #pure vector similarity search. Takes the question, converts it to an embedding, finds the 3 closest chunks in Chroma, returns them as a list of Document 对象，包含 page_content 和 metadata（比如来源文档的 id 或者 url）等信息
    context = "\n\n".join(doc.page_content for doc in source_docs)

    chain = (
        prompt  # 接收 {"context": ..., "question": ...} 字典，填充模板
        | llm
        | StrOutputParser() #这里直接把 llm 的输出当成最终答案，不做额外的处理,可能包含其他信息，比如来源文档的 id，这时候就需要一个更复杂的 output parser 来把答案和来源分开
    ) #后续可以改成更复杂的 chain 来处理检索结果，比如先用一个 LLM 来筛选出最相关的文档，再用另一个 LLM 来生成最终答案，这里为了简化直接把 retriever 的结果放到 prompt 里了；或者用多轮对话的方式来让模型逐步筛选和整合信息

    answer = chain.invoke({"context": context, "question": question}) #这里是链路的开始，用户输入问题后，先经过 retriever 找到相关文档块，再把这些文档块和问题一起放到 Prompt_template,生成最终的问题，发给LLM， 最终生成答案
    sources = list({doc.metadata.get("source", "未知") for doc in source_docs})                                                     
    return {"answer": answer, "sources": sources}  