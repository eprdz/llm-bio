from llm import GenomicVectorMockAgent
agent = GenomicVectorMockAgent()

print("\n" + "="*50)
print("Sequencing AI (Vector Search with Mock DB)")
print("="*50)

while True:
    # You no longer need to ask for the RSID specifically!
    user_input = input("\nYou (or type 'exit'): ")
    
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue
        
    report = agent.chat(user_input)
    print(f"\nRAG Report:\n{report}")
