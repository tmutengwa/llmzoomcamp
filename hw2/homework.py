import numpy as np
import minsearch
from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents

def run_homework():
    print("=================== LLM Zoomcamp: Vector Search Homework ===================")
    
    # Initialize ONNX MiniLM embedder
    embedder = Embedder()

    # --- Q1: Embedding a query ---
    query_q1 = "How does approximate nearest neighbor search work?"
    v = embedder.encode(query_q1)
    print(f"Q1: First value of query vector (v[0]): {v[0]:.4f} (Expected: -0.02)")

    # --- Loading the data ---
    print("\nLoading documents from GitHub repository...")
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    print(f"Loaded {len(documents)} document pages.")

    # --- Q2: Cosine Similarity ---
    target_filename = "02-vector-search/lessons/07-sqlitesearch-vector.md"
    target_doc = next(doc for doc in documents if doc["filename"] == target_filename)
    v_doc = embedder.encode(target_doc["content"])
    
    # Since vectors are normalized, dot product equals cosine similarity
    similarity = v.dot(v_doc)
    print(f"Q2: Cosine similarity with target page: {similarity:.4f} (Expected: 0.37)")

    # --- Q3: Chunking and Search by Hand ---
    print("\nChunking documents...")
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Generated {len(chunks)} chunks.")

    print("Embedding chunks batch...")
    chunk_texts = [c["content"] for c in chunks]
    X = embedder.encode_batch(chunk_texts)

    # Score against the Q1 query vector
    scores = X.dot(v)
    best_idx = np.argmax(scores)
    best_chunk = chunks[best_idx]
    print(f"Q3: Highest-scoring chunk belongs to: {best_chunk['filename']} (Expected: 02-vector-search/lessons/07-sqlitesearch-vector.md)")

    # --- Q4: Vector search with minsearch ---
    print("\nIndexing with minsearch VectorSearch...")
    vector_search = minsearch.VectorSearch(keyword_fields=["filename"])
    vector_search.fit(X, chunks)

    query_q4 = "What metric do we use to evaluate a search engine?"
    v_q4 = embedder.encode(query_q4)
    results_q4 = vector_search.search(v_q4, num_results=5)
    print(f"Q4: First result file of vector search: {results_q4[0]['filename']} (Expected: 04-evaluation/lessons/05-search-metrics.md)")

    # --- Q5: Text search vs vector search ---
    print("\nComparing text search vs vector search...")
    text_index = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)

    query_q5 = "How do I store vectors in PostgreSQL?"
    v_q5 = embedder.encode(query_q5)
    vector_results_q5 = vector_search.search(v_q5, num_results=5)
    text_results_q5 = text_index.search(query=query_q5, num_results=5)

    vector_files_q5 = {doc["filename"] for doc in vector_results_q5}
    text_files_q5 = {doc["filename"] for doc in text_results_q5}
    diff_files = vector_files_q5 - text_files_q5
    print(f"Q5: Files in vector top-5 but not in text top-5: {diff_files} (Expected: {{'02-vector-search/lessons/08-pgvector.md'}})")

    # --- Q6: Hybrid search with RRF ---
    def rrf(result_lists, k=60, num_results=5):
        scores = {}
        docs = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["filename"], doc["start"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]

    query_q6 = "How do I give the model access to tools?"
    v_q6 = embedder.encode(query_q6)
    vector_results_q6 = vector_search.search(v_q6, num_results=10)
    text_results_q6 = text_index.search(query=query_q6, num_results=10)

    fused_results = rrf([vector_results_q6, text_results_q6], num_results=5)
    print(f"Q6: Top file after hybrid RRF: {fused_results[0]['filename']} (Expected: 01-agentic-rag/lessons/13-function-calling.md)")

if __name__ == "__main__":
    run_homework()
