# LLM Zoomcamp Homeworks

This repository contains my homework submissions for the [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Structure

* **`hw1/`**: Module 1 - Agentic RAG.
  * Contains the implementation of the RAG assistant and agent loop using `toyaikit`.
* **`hw2/`**: Module 2 - Vector Search.
  * Contains the implementation of text search, vector search, and hybrid search (RRF) using `minsearch` and the ONNX `all-MiniLM-L6-v2` embedder.

## Setup

Each homework folder has its own isolated Python virtual environment (`venv`) managed by `uv`.

To run the homework scripts:

### Homework 1 (Agentic RAG)
```bash
cd hw1
venv/bin/python homework.py
```

### Homework 2 (Vector Search)
```bash
cd hw2
venv/bin/python homework.py
```
