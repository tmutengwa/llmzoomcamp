import os
import json
import time
from search import FastAPIQAAssistant

# Warm up the assistant on container initialization (so first request is fast)
assistant = FastAPIQAAssistant()

def handler(event, context):
    try:
        # Check if parameters are passed directly in event (SDK direct invocation)
        if "query" in event:
            params = event
        else:
            # Fallback to API Gateway / Function URL format (wrapped in body)
            body_str = event.get("body", "{}")
            if isinstance(body_str, str):
                try:
                    params = json.loads(body_str)
                except json.JSONDecodeError:
                    params = {}
            else:
                params = body_str
            
        # Fallback to query string parameters if body parsing yielded nothing
        if not params:
            params = event.get("queryStringParameters", {}) or {}
            
        query = params.get("query")
        if not query:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"error": "Missing 'query' parameter."})
            }
            
        search_type = params.get("search_type", "hybrid")
        enable_rewriting = str(params.get("enable_rewriting", "false")).lower() in ("true", "1")
        enable_reranking = str(params.get("enable_reranking", "false")).lower() in ("true", "1")
        num_results = int(params.get("num_results", 5))
        
        start_time = time.time()
        
        # 2. Retrieve documentation context
        context_docs = assistant.retrieve(
            query=query,
            search_type=search_type,
            enable_rewriting=enable_rewriting,
            enable_reranking=enable_reranking,
            num_results=num_results
        )
        
        # 3. Generate final answer
        answer, usage = assistant.generate_answer(query, context_docs)
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # 4. Format response payload
        response = {
            "answer": answer,
            "usage": usage,
            "sources": context_docs,
            "response_time_ms": response_time_ms
        }
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*" # CORS enabled
            },
            "body": json.dumps(response)
        }
    except Exception as e:
        print(f"Error handling Lambda request: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }
