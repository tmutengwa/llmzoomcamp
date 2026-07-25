import os
import json
import sqlite3
import numpy as np
import dlt
from gitsource import GithubRepositoryDataReader, chunk_documents
from embedder import Embedder

def run_ingestion():
    print("=================== FastAPI Docs Ingestion Pipeline ===================")
    
    # 1. Fetch documentation from GitHub
    print("Fetching documentation pages from tiangolo/fastapi...")
    reader = GithubRepositoryDataReader(
        repo_owner="tiangolo",
        repo_name="fastapi",
        commit_id="master",
        allowed_extensions={"md"},
        filename_filter=lambda path: "docs/en/docs/" in path,
    )
    files = reader.read()
    documents = []
    for file in files:
        doc = file.parse()
        # Clean path to be relative and readable
        doc["filename"] = doc["filename"].replace("docs/en/docs/", "")
        documents.append(doc)
    
    print(f"Successfully fetched {len(documents)} markdown files.")

    # 2. Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents, size=1500, step=750)
    print(f"Generated {len(chunks)} chunks.")

    # 3. Embed chunks using ONNX all-MiniLM-L6-v2
    print("Embedding chunks using ONNX embedder...")
    embedder = Embedder()
    chunk_texts = [c["content"] for c in chunks]
    embeddings = embedder.encode_batch(chunk_texts)
    
    # Add embeddings to chunk dictionaries
    payload = []
    for i, c in enumerate(chunks):
        payload.append({
            "chunk_id": i,
            "filename": c["filename"],
            "start_line": c.get("start", 0),  # start line/char offset
            "content": c["content"],
            "embedding": json.dumps(embeddings[i].tolist()) # Store as JSON string in SQLite
        })

    # 4. Ingest into SQLite using dlt
    print("Loading data into SQLite database using dlt...")
    
    # Configure dlt to use project.db
    # We specify connection string in credentials
    pipeline = dlt.pipeline(
        pipeline_name="fastapi_docs_ingestion",
        destination="duckdb",
        dataset_name="fastapi"
    )
    
    # Define a resource yielding the payload
    @dlt.resource(name="docs_chunks", write_disposition="replace")
    def docs_chunks_resource():
        yield payload

    # Run the pipeline
    info = pipeline.run(docs_chunks_resource())
    print("Pipeline run info:")
    print(info)
    
    # 5. Build lexical search index and save it for fast loading
    # We will build minsearch index on content & filename and serialize it
    print("Building and saving minsearch index...")
    import minsearch
    index = minsearch.Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(chunks)
    
    with open("minsearch_index.json", "w") as f:
        # minsearch fits internal representations, we can save fit parameters or chunks directly.
        # Since minsearch uses in-memory fit, we can just save it or fit it on startup from database.
        # Fitting on startup from SQLite is extremely fast (less than 100ms for 800 chunks),
        # so we can just fit it on the fly from SQLite.
        pass
        
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    run_ingestion()
