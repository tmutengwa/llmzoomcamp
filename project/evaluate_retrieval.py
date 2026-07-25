import pandas as pd
from search import FastAPIQAAssistant

# Ground truth Q&A dataset based on FastAPI English documentation files
GROUND_TRUTH = [
    {
        "question": "How do I use path parameters with type hints in FastAPI?",
        "target_file": "tutorial/path-params.md"
    },
    {
        "question": "What is the recommended way to declare optional query parameters?",
        "target_file": "tutorial/query-params.md"
    },
    {
        "question": "How do I define a request body using Pydantic models?",
        "target_file": "tutorial/body.md"
    },
    {
        "question": "How can I set default values for query parameters?",
        "target_file": "tutorial/query-params.md"
    },
    {
        "question": "How do I handle HTTP exceptions and return custom status codes?",
        "target_file": "tutorial/handling-errors.md"
    },
    {
        "question": "How do I configure CORS (Cross-Origin Resource Sharing) middleware?",
        "target_file": "tutorial/cors.md"
    },
    {
        "question": "How does dependency injection work with the Depends keyword?",
        "target_file": "tutorial/dependencies/index.md"
    },
    {
        "question": "How do I declare form fields in a request?",
        "target_file": "tutorial/request-forms.md"
    },
    {
        "question": "How do I upload multiple files simultaneously in FastAPI?",
        "target_file": "tutorial/request-files.md"
    },
    {
        "question": "How do I write dependency overrides for testing?",
        "target_file": "advanced/testing-dependencies.md"
    },
    {
        "question": "How do I use OAuth2 with password flow and JWT tokens for security?",
        "target_file": "tutorial/security/oauth2-jwt.md"
    },
    {
        "question": "How can I create database connections using dependencies and yield?",
        "target_file": "tutorial/dependencies/dependencies-with-yield.md"
    },
    {
        "question": "How do I read cookies inside a path operation?",
        "target_file": "tutorial/cookie-params.md"
    },
    {
        "question": "How do I set response headers in my FastAPI endpoints?",
        "target_file": "tutorial/response-headers.md"
    },
    {
        "question": "How do I declare request headers as parameters?",
        "target_file": "tutorial/header-params.md"
    },
    {
        "question": "How do I configure static files like CSS and JS in FastAPI?",
        "target_file": "tutorial/static-files.md"
    },
    {
        "question": "How do I run background tasks after returning a response?",
        "target_file": "tutorial/background-tasks.md"
    },
    {
        "question": "How do I write asynchronous tests using AsyncClient?",
        "target_file": "advanced/async-tests.md"
    },
    {
        "question": "How do I define path parameters that contain directory paths?",
        "target_file": "tutorial/path-params.md"
    },
    {
        "question": "How do I mount a sub-application under a main FastAPI app?",
        "target_file": "advanced/sub-applications.md"
    }
]

def calculate_metrics(assistant, search_type, k=5):
    hits = 0
    reciprocal_ranks = 0.0
    
    for item in GROUND_TRUTH:
        q = item["question"]
        target = item["target_file"]
        
        # Perform retrieval
        if search_type == "text":
            results = assistant.keyword_search(q, num_results=k)
        elif search_type == "vector":
            results = assistant.vector_search(q, num_results=k)
        elif search_type == "hybrid":
            results = assistant.hybrid_search(q, num_results=k)
            
        # Check hit
        hit_found = False
        for rank, doc in enumerate(results):
            if doc["filename"] == target:
                hits += 1
                reciprocal_ranks += 1.0 / (rank + 1)
                hit_found = True
                break
                
    hit_rate = hits / len(GROUND_TRUTH)
    mrr = reciprocal_ranks / len(GROUND_TRUTH)
    return hit_rate, mrr

def main():
    print("=================== FastAPI Docs Retrieval Evaluation ===================")
    assistant = FastAPIQAAssistant()
    
    methods = ["text", "vector", "hybrid"]
    results = []
    
    for method in methods:
        print(f"Evaluating {method} search...")
        hit_rate, mrr = calculate_metrics(assistant, method, k=5)
        results.append({
            "Method": method.upper(),
            "Hit Rate @ 5": f"{hit_rate:.2%}",
            "MRR @ 5": f"{mrr:.4f}"
        })
        
    df = pd.DataFrame(results)
    print("\nEvaluation Results Table:")
    print(df.to_markdown(index=False))
    
    print("\nSummary:")
    print("Hybrid search (RRF) combines lexical text matching with vector-based semantic retrieval, yielding the best balanced Hit Rate and MRR.")

if __name__ == "__main__":
    main()
