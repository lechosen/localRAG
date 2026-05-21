# RAG 17 Optimization Strategies — Code Reference

## Strategy Summary Table

| # | Strategy | Category | Key Idea |
|---|----------|----------|----------|
| 1 | Fixed Chunking | Chunking | Baseline; split by size with overlap |
| 2 | Semantic Chunking | Chunking | Split at semantic boundaries |
| 3 | Small-to-Big | Chunking | Retrieve small, return large |
| 4 | Context Enrichment | Chunking | Return neighboring chunks too |
| 5 | Chunk Headers | Chunking | LLM-generated title improves embedding quality |
| 6 | Doc Augmentation | Query | Pre-generate questions at index time |
| 7 | Query Transform | Query | Rewrite / step-back / decompose |
| 8 | Re-ranking | Retrieval | Cross-encoder precision after coarse retrieval |
| 9 | RSE | Retrieval | Sliding window for highest-scoring contiguous segment |
| 10 | Context Compression | Retrieval | LLM strips irrelevant content from chunks |
| 11 | Feedback Loop | Advanced | Weight documents by user feedback |
| 12 | Self-RAG | Advanced | Model decides if retrieval is needed |
| 13 | Knowledge Graph RAG | Advanced | Entity-relationship graph instead of flat vectors |
| 14 | Hierarchical Indexing | Advanced | Summary layer → detail layer two-stage search |
| 15 | HyDE | Query | Hypothetical answer → search vector |
| 16 | Fusion RAG | Retrieval | Vector + BM25 hybrid |
| 17 | CRAG | Advanced | Corrective retrieval with web search fallback |

---

## Standard LCEL RAG Chain (baseline for all strategies)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the context:\n{context}\n\nQuestion: {question}"
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = rag_chain.invoke("your query")
```

---

## Part 1: Chunking Strategies

### Strategy 1 — Fixed-Length Chunking (Simple RAG)
Split text into fixed-size overlapping windows.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
docs = splitter.split_documents(documents)
```

### Strategy 2 — Semantic Chunking
Split at semantic boundaries (low cosine similarity between adjacent sentences).

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
docs = chunker.create_documents([text])
```

### Strategy 3 — Small-to-Big Retrieval
Index small child chunks; return the larger parent chunk on retrieval hit.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=150),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=600),
)
retriever.add_documents(documents)
results = retriever.invoke("query")  # hits child chunk → returns parent
```

### Strategy 4 — Context-Enriched Retrieval
On a retrieval hit, also return the adjacent neighboring chunks.

```python
def retrieve_with_context(query, all_chunks, vector_db, window=1, top_k=3):
    results = vector_db.similarity_search(query, k=top_k)
    enriched = []
    for r in results:
        idx = all_chunks.index(r.page_content)
        start, end = max(0, idx - window), min(len(all_chunks), idx + window + 1)
        enriched.append(" ".join(all_chunks[start:end]))
    return enriched
```

### Strategy 5 — Contextual Chunk Headers
Prepend an LLM-generated title to each chunk before embedding.

```python
HEADER_PROMPT = "Generate a short descriptive title (<10 words) for this text:\n{chunk}\nTitle only:"

for doc in docs:
    header = llm.invoke(HEADER_PROMPT.format(chunk=doc.page_content)).content.strip()
    doc.page_content = f"[{header}]\n{doc.page_content}"
```

---

## Part 2: Query-Side Strategies

### Strategy 6 — Document Augmentation
At index time, generate hypothetical questions per chunk and embed them alongside the chunk.

```python
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import uuid

QUESTION_GEN_PROMPT = 'Generate 3 questions a user might ask about this text. JSON list only:\n"{chunk}"'

store = InMemoryByteStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key="doc_id")
doc_ids = [str(uuid.uuid4()) for _ in docs]

question_docs = []
for i, doc in enumerate(docs):
    questions = json.loads(llm.invoke(QUESTION_GEN_PROMPT.format(chunk=doc.page_content)).content)
    question_docs.extend([
        Document(page_content=q, metadata={"doc_id": doc_ids[i]}) for q in questions
    ])

retriever.vectorstore.add_documents(question_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
# retriever.invoke("query") → hits question vector → returns original chunk
```

### Strategy 7 — Query Transformation
Three sub-techniques: rewrite (clarify), step-back (broaden), decompose (split complex queries).

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Automatically generates multiple rewritten queries and merges results
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)
results = retriever.invoke("complex query")
```

### Strategy 15 — HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, then use it as the search vector.

```python
HYDE_PROMPT = "Write a 2-3 sentence hypothetical answer to this question:\n{query}\nAnswer:"

def retrieve_with_hyde(query, vector_db, llm, top_k=3):
    hypothetical_doc = llm.invoke(HYDE_PROMPT.format(query=query)).content
    return vector_db.similarity_search(hypothetical_doc, k=top_k)
```

---

## Part 3: Retrieval Patterns

### Strategy 8 — Re-ranking (Cross-Encoder)
Retrieve a large candidate set (top-50), then re-rank with a cross-encoder for precision.

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
)
results = retriever.invoke("query")
```

### Strategy 10 — Context Compression
Use an LLM to strip irrelevant content from retrieved chunks before passing to the generator.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(),
)
compressed_docs = retriever.invoke("query")
```

### Strategy 16 — Fusion RAG (Hybrid Retrieval)
Combine vector search (semantic) with BM25 (keyword) for better coverage.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)
results = ensemble.invoke("query")
```

---

## Part 4: Advanced Architectures

### Strategy 12 — Self-RAG (LangGraph)
Model decides whether retrieval is needed, then filters retrieved chunks for relevance.

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List

class SelfRAGState(TypedDict):
    query: str
    need_retrieval: str
    documents: List[str]
    answer: str

NEED_RETRIEVAL_PROMPT = 'Does this question require external knowledge? Reply "yes" or "no".\nQuestion: {query}'
RELEVANCE_PROMPT = 'Is this chunk relevant to the question? Reply "relevant" or "irrelevant".\nQ: {query}\nChunk: {chunk}'

def check_node(state):
    need = llm.invoke(NEED_RETRIEVAL_PROMPT.format(query=state["query"])).content.strip()
    return {**state, "need_retrieval": need}

def retrieve_node(state):
    candidates = vector_db.similarity_search(state["query"], k=5)
    relevant = [r.page_content for r in candidates
                if llm.invoke(RELEVANCE_PROMPT.format(query=state["query"], chunk=r.page_content))
                          .content.strip() == "relevant"]
    return {**state, "documents": relevant}

def generate_node(state):
    context = "\n\n".join(state["documents"]) or "(no relevant documents)"
    answer = llm.invoke(f"Answer based on context:\n{context}\n\nQuestion: {state['query']}").content
    return {**state, "answer": answer}

graph = StateGraph(SelfRAGState)
graph.add_node("check", check_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "check")
graph.add_conditional_edges("check", lambda s: s["need_retrieval"], {"yes": "retrieve", "no": "generate"})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()
result = app.invoke({"query": "what is RAG?", "need_retrieval": "", "documents": [], "answer": ""})
```

### Strategy 17 — CRAG (Corrective RAG, LangGraph)
Grade retrieved docs (high/medium/low relevance); fall back to web search when relevance is low.

```python
from langgraph.graph import StateGraph, END, START
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, List

class CRAGState(TypedDict):
    query: str
    documents: List[str]
    relevance: str
    answer: str

GRADE_PROMPT = 'Rate this chunk vs the question: "high", "medium", or "low".\nQ: {query}\nChunk: {chunk}'
web_search = TavilySearchResults(max_results=3)

def retrieve_node(state):
    docs = [r.page_content for r in vector_db.similarity_search(state["query"], k=3)]
    return {**state, "documents": docs}

def grade_node(state):
    scores = [llm.invoke(GRADE_PROMPT.format(query=state["query"], chunk=c)).content.strip()
              for c in state["documents"]]
    relevance = "high" if "high" in scores else "medium" if "medium" in scores else "low"
    return {**state, "relevance": relevance}

def web_search_node(state):
    web_results = [r["content"] for r in web_search.invoke(state["query"])]
    return {**state, "documents": state["documents"] + web_results}

def generate_node(state):
    context = "\n\n".join(state["documents"])
    answer = llm.invoke(f"Answer based on context:\n{context}\n\nQuestion: {state['query']}").content
    return {**state, "answer": answer}

graph = StateGraph(CRAGState)
for name, fn in [("retrieve", retrieve_node), ("grade", grade_node),
                 ("web_search", web_search_node), ("generate", generate_node)]:
    graph.add_node(name, fn)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", lambda s: s["relevance"],
                            {"high": "generate", "medium": "web_search", "low": "web_search"})
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

app = graph.compile()
```

---

## Recommended Production Combinations

```python
# Combination 1: Hybrid retrieval + re-ranking (high precision)
ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
final_retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=HuggingFaceCrossEncoder(...), top_n=5),
    base_retriever=ensemble,
)

# Combination 2: Self-RAG + CRAG state machine (high reliability)
# Self-RAG decides whether to retrieve → CRAG grades relevance → web search fallback
```
