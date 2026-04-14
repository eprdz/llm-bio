import ollama

class GenomicClinicalAgentLocal:
    def __init__(self):
        self.local_model = "phi3"
        print(f"Using Ollama locally with model {self.local_model}")

        # 1. KNOWLEDGE BASE ## TODO: Create a method to read everything from a vcf and save it to a file
        self.clinical_database = {
            "rs1042522": {
                "gene": "TP53",
                "hgvs": "NM_000546.5:c.215C>G",
                "significance": "Pathogenic",
                "condition": "Li-Fraumeni syndrome",
                "details": "Hereditary predisposition to multiple types of cancer. Requires frequent monitoring."
            },
            "rs1801133": {
                "gene": "MTHFR",
                "hgvs": "NM_000237.2:c.677C>T",
                "significance": "Benign",
                "condition": "Folate metabolism variant",
                "details": "Common in the population. Not pathogenic on its own."
            }
        }

    def _retrieve_context(self, variant_query: str) -> str:
        """Searches our local database (The RAG Retrieval)."""
        variant_query = variant_query.lower().strip()
        if variant_query in self.clinical_database:
            data = self.clinical_database[variant_query]
            return f"Gene: {data['gene']} | Variant: {data['hgvs']} | Classification: {data['significance']} | Condition: {data['condition']} | Details: {data['details']}"
        return None

    def generate_report(self, user_query: str, variant_id: str) -> str:
        """Orchestrates the RAG using the local LLM."""
        context = self._retrieve_context(variant_id)

        # System Prompt (The rules of the game)
        system_prompt = """
        You are a Virtual Genetic Counselor. Answer ONLY based on the 'Retrieved Information'.
        RULES:
        1. If the Retrieved Information says "[NO DATA]", you must answer: "I do not have clinical data on that variant".
        2. Do not invent genetic associations.
        3. Maintain a medical, empathetic, and professional tone.
        """

        if context:
            user_prompt = f"Retrieved Information:\n{context}\n\nQuestion:\n{user_query}"
        else:
            user_prompt = f"Retrieved Information:\n[NO DATA FOUND FOR {variant_id}]\n\nQuestion:\n{user_query}"

        print(f"\nProcessing the report on CPU/GPU...")

        try:
            # Call to the local Ollama server
            response = ollama.chat(
                model=self.local_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    "temperature": 0.1 # Low temperature so it is deterministic
                }
            )
            return response['message']['content']

        except Exception as e:
            return f"Connection error: {e}"
