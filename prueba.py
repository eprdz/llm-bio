import ollama
import json
import time
from llm import GenomicVectorMockAgent

# This will trigger the database load and native embedding model setup
agent = GenomicVectorMockAgent()

print("\n" + "="*60)
print("Sequencing AI (LLM Routing Agent - CPU Optimized)")
print("="*60)
print("Notice: Running on CPU inference. Expect a few seconds of delay.")
print("="*60)

def classify_intent_with_llm(user_input: str) -> dict:
    """Uses an LLM as a 'Judge/Router' to classify the user's intent."""
    
    system_prompt = """
    You are an intelligent router for a clinical database. Your ONLY goal is to read the user's input and return a strict JSON.
    Do not write additional text or explanations, ONLY return the JSON.
    
    Classification rules:
    1. If the user asks to list, show, or asks for all variants of a specific gene, disease, or pathogenicity level, the intent is "exact_filter".
    2. If the user asks an open-ended question, an explanation, or about the meaning of something (e.g., "What is rs123?", "What impact does...?"), the intent is "rag_question".
    
    Allowed fields for exact_filter: "gene", "condition", "significance".
    
    Example 1: "Give me all variants of the TP53 gene" -> {"intent": "exact_filter", "field": "gene", "value": "TP53"}
    Example 2: "Show variants of breast cancer" -> {"intent": "exact_filter", "field": "condition", "value": "breast cancer"}
    Example 3: "What does rs1042522 mean" -> {"intent": "rag_question", "field": null, "value": null}
    Example 4: "Do you have pathogenic variants?" -> {"intent": "exact_filter", "field": "significance", "value": "pathogenic"}
    """
    
    try:
        response = ollama.chat(
            model='phi3',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input}
            ],
            format='json', 
            options={'temperature': 0.0} 
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"intent": "rag_question"}

while True:
    user_input = input("\nYou (or type 'exit'): ")
    
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue
        
    print("\n[Router] The AI is analyzing your hidden intent. Please wait...")
    
    # Measure Routing Time
    start_time = time.time()
    decision = classify_intent_with_llm(user_input)
    routing_time = time.time() - start_time
    
    print(f"[Router] Decision made: {decision} (Took {routing_time:.2f} seconds)")
    
    if decision.get("intent") == "exact_filter":
        field = decision.get("field")
        value = decision.get("value")
        
        if field and value:
            if field in ["condition", "significance"]:
                filters = {field: {"$contains": value.lower()}}
            else:
                filters = {field: value.upper()}
                
            print("[Agent] Accessing Database via Exact Filter. Retrieving data...")
            # Database calls are instant, no need to time them
            report = agent.exact_metadata_search(filters)
        else:
            print("[Agent] Router provided incomplete data. Falling back to RAG flow. Generating answer...")
            start_rag = time.time()
            report = agent.chat(user_input)
            print(f"[Agent] Generation complete (Took {time.time() - start_rag:.2f} seconds)")
            
    else:
        print("[Agent] Deductive question detected. Reading documents and generating answer...")
        start_rag = time.time()
        report = agent.chat(user_input)
        print(f"[Agent] Generation complete (Took {time.time() - start_rag:.2f} seconds)")
        
    print(f"\nReport:\n{report}")