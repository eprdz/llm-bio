import ollama
import chromadb
import json
import os
import gzip
import re


class GenomicVectorMockAgent:
    def __init__(self):
        self.llm_model = "phi3"
        self.embed_model = "nomic-embed-text"
        self.clinical_database = self.load_db()

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="mock_clinical_variants")
        
        self._vectorize_database()

    def load_db(self, json_file="db.json", clinvar_file="clinvar.vcf.gz"):
        """Loads the database. If it does not exist, it creates it and then loads it."""
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                print(f"{json_file} found")
                json_ = json.load(f)

            return json_
        else:
            print(f"{clinvar_file} found and no json file. Generating database...")
            self.parse_clinvar(clinvar_file)
    

    def parse_clinvar(self, clinvar_file="clinvar.vcf.gz"):
        dict_ = {}
        print(f"Parsing {clinvar_file}...")
        
        # Open the gzip file safely in text mode with utf-8
        with gzip.open(clinvar_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                columns = line.strip().split('\t')
                if len(columns) < 8:
                    continue
                
                var_id = columns[2]
                info_field = columns[7]
                
                def get_info(key):
                    match = re.search(fr'\b{key}=([^;]+)', info_field)
                    return match.group(1) if match else ""

                clndn = get_info("CLNDN")
                clnsig = get_info("CLNSIG")
                geneinfo = get_info("GENEINFO")
                mc = get_info("MC")
                rs = get_info("RS")
                
                dict_[var_id] = {
                    "alleleid": get_info("ALLELEID"),
                    "condition": clndn.split('|')[0].split(',')[0].replace('_', ' ') if clndn else "",
                    "HGVS": get_info("CLNHGVS"),
                    "significance": clnsig.split('|')[0].replace('_', ' ') if clnsig else "",
                    "gene": geneinfo.split(':')[0] if geneinfo else "",
                    "type_of_variation": mc.split('|')[0] if mc else "",
                    "rsid": f"rs{rs}" if rs else ""
                }

        with open("db.json", "w") as f:
            json.dump(dict_, f)
                
        print(f"Successfully loaded {len(dict_)} variants into memory.")
        return dict_


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
        
        for clinvar_id, data in self.clinical_database.items():
            # Create a rich text paragraph for the vector model to understand
            text_to_embed = f"The variant {clinvar_id} in the gene {data['gene']} is associated with the condition {data['condition']}. Type of variation: {data['type_of_variation']}. Rsid: {data['rsid']} Clinical classification: {data['significance']}. HGVS: {data['HGVS']}"
            
            documents.append(text_to_embed)
            metadatas.append({"clinvar_id": clinvar_id, "gene": data['gene']})
            ids.append(clinvar_id)
            
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
            n_results=30 # Study THIS
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
