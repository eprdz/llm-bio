from llm import GenomicClinicalAgentLocal
agent = GenomicClinicalAgentLocal()

while True:
    print("\n" + "-"*50)
    variant = input("Introduce RSID (ej. rs1042522) or 'exit': ")
    if variant.lower() == 'exit':
        break

    quest = input("¿What dp ypu want to know?: ")

    report = agent.generate_report(quest, variant)
    print(f"\nCLINICAL REPORT:\n{report}")
