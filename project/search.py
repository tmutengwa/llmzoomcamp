import os
import json
import sqlite3
import numpy as np
import duckdb
import minsearch
from embedder import Embedder
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_PATH = "fastapi_docs_ingestion.duckdb"

class FastAPIQAAssistant:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.embedder = Embedder()
        self.load_data()
        self.init_indexes()
        
        # Initialize OpenAI client if key is present
        self.openai_client = None
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def load_data(self):
        """Loads chunks and their vector embeddings from the DuckDB database."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}. Please run ingest.py first.")
            
        conn = duckdb.connect(self.db_path)
        # Retrieve payload columns from the schema created by dlt
        res = conn.execute("SELECT chunk_id, filename, start_line, content, embedding FROM fastapi.docs_chunks").fetchall()
        
        self.chunks = []
        vectors = []
        for row in res:
            chunk = {
                "chunk_id": int(row[0]),
                "filename": row[1],
                "start_line": int(row[2]),
                "content": row[3],
            }
            self.chunks.append(chunk)
            vectors.append(json.loads(row[4]))
            
        self.X = np.array(vectors)
        conn.close()

    def init_indexes(self):
        """Initializes the keyword search index with minsearch."""
        self.text_index = minsearch.Index(
            text_fields=["content"],
            keyword_fields=["filename"]
        )
        self.text_index.fit(self.chunks)

    def keyword_search(self, query, num_results=5):
        """Lexical search using TF-IDF."""
        return self.text_index.search(query=query, num_results=num_results)

    def vector_search(self, query, num_results=5):
        """Semantic search using Cosine Similarity of ONNX embeddings."""
        v = self.embedder.encode(query)
        # Cosine similarity is dot product since vectors are normalized
        scores = self.X.dot(v)
        top_indices = np.argsort(scores)[::-1][:num_results]
        
        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def hybrid_search(self, query, k=60, num_results=5):
        """Hybrid search combining keyword and vector search with Reciprocal Rank Fusion (RRF)."""
        # Retrieve candidates
        v_results = self.vector_search(query, num_results=num_results * 3)
        t_results = self.keyword_search(query, num_results=num_results * 3)
        
        scores = {}
        docs = {}
        
        # Apply RRF formula
        for results in [v_results, t_results]:
            for rank, doc in enumerate(results):
                key = doc["chunk_id"]
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc
                
        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked_keys[:num_results]]

    def rewrite_query(self, query):
        """Uses LLM to expand the user query into 3 distinct search queries."""
        if not self.openai_client:
            return [query]
            
        prompt = f"""
You are an expert search engine assistant. Rewrite the user's question about FastAPI into 3 distinct, high-quality search queries targeting technical documentation.
Provide only a JSON object containing a 'queries' list of 3 strings. Do not include markdown formatting or conversational text.

User Question: {query}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("queries", [query])
        except Exception as e:
            print(f"Error during query rewriting: {e}")
            return [query]

    def search_with_rewriting(self, query, search_type="hybrid", num_results=5):
        """Expands user query and aggregates unique document hits from all queries."""
        expanded_queries = self.rewrite_query(query)
        all_results = {}
        
        for q in expanded_queries:
            if search_type == "vector":
                res = self.vector_search(q, num_results=num_results * 2)
            elif search_type == "text":
                res = self.keyword_search(q, num_results=num_results * 2)
            else:
                res = self.hybrid_search(q, num_results=num_results * 2)
                
            for doc in res:
                all_results[doc["chunk_id"]] = doc
                
        return list(all_results.values())

    def rerank_documents(self, query, documents, num_results=5):
        """Re-ranks candidate documents using the LLM as a cross-encoder judge."""
        if not self.openai_client or len(documents) == 0:
            return documents[:num_results]
            
        candidates = []
        for i, doc in enumerate(documents):
            candidates.append(f"Doc ID: {i}\nPath: {doc['filename']}\nContent: {doc['content']}\n")
        candidates_str = "\n---\n".join(candidates)
        
        prompt = f"""
You are an expert technical documentation evaluator. Rate the relevance of each candidate document to the search query on a scale of 1-5 (5 being highly relevant, 1 being completely irrelevant).
Return a JSON object where keys are the Document IDs (strings) and values are the relevance scores (integers).
Do not include any conversational explanation or markdown formatting.

Query: {query}

Candidates:
{candidates_str}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            scores = json.loads(response.choices[0].message.content)
            
            for i, doc in enumerate(documents):
                doc_id_str = str(i)
                doc["rerank_score"] = int(scores.get(doc_id_str, 1))
                
            sorted_docs = sorted(documents, key=lambda x: x.get("rerank_score", 1), reverse=True)
            return sorted_docs[:num_results]
        except Exception as e:
            print(f"Error during re-ranking: {e}")
            return documents[:num_results]

    def retrieve(self, query, search_type="hybrid", enable_rewriting=False, enable_reranking=False, num_results=5):
        """Unified retrieval flow incorporating candidate search, query expansion, and re-ranking."""
        if enable_rewriting:
            candidates = self.search_with_rewriting(query, search_type=search_type, num_results=num_results)
        else:
            if search_type == "vector":
                candidates = self.vector_search(query, num_results=num_results * 2)
            elif search_type == "text":
                candidates = self.keyword_search(query, num_results=num_results * 2)
            else:
                candidates = self.hybrid_search(query, num_results=num_results * 2)
                
        if enable_reranking:
            results = self.rerank_documents(query, candidates, num_results=num_results)
        else:
            results = candidates[:num_results]
            
        return results

    def generate_answer(self, query, context_docs):
        """Uses LLM to synthesize the final answer from retrieved documentation context."""
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"File: {doc['filename']}\nContent: {doc['content']}\n")
        context_str = "\n".join(context_parts)
        
        prompt = f"""
You are an expert FastAPI developer assistant. Answer the user's question using the provided context from the official FastAPI documentation.
Provide accurate, clear explanations and copyable code examples where applicable. If the answer cannot be found in the context, say "I don't know based on the provided documentation."

QUESTION: {query}

CONTEXT:
{context_str}
"""
        if not self.openai_client:
            offline_ans = f"Offline Mode: Successfully retrieved context from {len(context_docs)} docs.\n\nFirst snippet:\n{context_docs[0]['content'][:600]}..."
            return offline_ans, {"input_tokens": 0, "output_tokens": 0}
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            return answer, usage
        except Exception as e:
            err_msg = f"Error calling OpenAI API: {e}"
            print(err_msg)
            return err_msg, {"input_tokens": 0, "output_tokens": 0}

if __name__ == "__main__":
    # Self-test
    assistant = FastAPIQAAssistant()
    print("Testing Vector Search:")
    res = assistant.vector_search("How do I do dependency injection?")
    print(f"Top result: {res[0]['filename']} (Score: {res[0]['score']:.4f})")
    
    print("\nTesting Hybrid Search:")
    res_hybrid = assistant.hybrid_search("How do I do dependency injection?")
    print(f"Top result: {res_hybrid[0]['filename']}")
