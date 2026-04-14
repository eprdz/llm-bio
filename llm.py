import ollama
import chromadb

class GenomicVectorMockAgent:
    def __init__(self):
        self.llm_model = "phi3"
        self.embed_model = "nomic-embed-text"
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

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="mock_clinical_variants")
        
        self._vectorize_database()

    def _get_embedding(self, text: str):
        """Generates the mathematical vector for a given text using Ollama."""
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    def _vectorize_database(self):
        """Converts the dictionary into vectors and stores them in ChromaDB."""
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for rsid, data in self.clinical_database.items():
            # Create a rich text paragraph for the vector model to understand
            text_to_embed = f"The variant {rsid} in the gene {data['gene']} is associated with the condition {data['condition']}. Details: {data['details']} Clinical classification: {data['significance']}."
            
            documents.append(text_to_embed)
            metadatas.append({"rsid": rsid, "gene": data['gene']})
            ids.append(rsid)
            
            # Call the embedding model
            embeddings.append(self._get_embedding(text_to_embed))

        # Add to the vector database
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully indexed {len(documents)} variants in the Vector Space!")

    def _vector_search(self, query: str) -> str:
        """Finds the most relevant data based on the meaning of the question."""
        query_vector = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=1 # Study THIS
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0]
        return None

    def chat(self, user_query: str) -> str:
        """Orchestrates the RAG process."""
        context = self._vector_search(user_query)
        
        system_prompt = """
        You are a Virtual Genetic Counselor. Answer the user's question using ONLY the provided 'Medical Information'.
        RULES:
        1. If the Medical Information does not answer the question or says [EMPTY], you must answer: "I do not have clinical data regarding this."
        2. Do not invent genetic associations.
        3. Maintain a medical, empathetic, and professional tone.
        """
        
        if context:
            user_prompt = f"Medical Information:\n{context}\n\nQuestion:\n{user_query}"
        else:
            user_prompt = f"Medical Information:\n[EMPTY]\n\nQuestion:\n{user_query}"

        print("\nProcessing the report...")
        
        # Step 3: Let the LLM read the context and write the answer
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    "temperature": 0.1
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Connection error: {e}"
