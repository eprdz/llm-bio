import ollama
import chromadb
import os
import gzip
import re

class GenomicVectorMockAgent:
    def __init__(self):
        self.llm_model = "phi3"
        self.embed_model = "nomic-embed-text"

        self.chroma_client = chromadb.PersistentClient(path="./clinvar_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="clinical_variants")
        
        count = self.collection.count()
        if count == 0:
            print("Base de datos vectorial vacia. Leyendo VCF y generando vectores (esto puede tardar la primera vez)...")
            self._load_and_vectorize("clinvar.vcf.gz")
        else:
            print(f"Base de datos cargada desde disco instantaneamente. {count} variantes disponibles.")

    def parse_clinvar(self, clinvar_file="clinvar.vcf.gz"):
        """Lee el VCF y extrae los campos. Ya no guarda un JSON."""
        dict_ = {}
        print(f"Parsing {clinvar_file}...")
        
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
                
        return dict_

    def _get_embedding(self, text: str, is_query: bool = False):
        """Usa prefijos de Nomic para optimizar la busqueda y generar el vector."""
        prefix = "search_query: " if is_query else "search_document: "
        response = ollama.embeddings(model=self.embed_model, prompt=prefix + text)
        return response["embedding"]

    def _load_and_vectorize(self, clinvar_file):
        """Unifica el parseo y la vectorizacion, insertando en lotes (batches)."""
        temp_database = self.parse_clinvar(clinvar_file)
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        print("Vectorizando datos e insertando en disco...")
        for clinvar_id, data in temp_database.items():
            text_to_embed = f"The variant {clinvar_id} in the gene {data['gene']} is associated with the condition {data['condition']}. Type of variation: {data['type_of_variation']}. Rsid: {data['rsid']} Clinical classification: {data['significance']}. HGVS: {data['HGVS']}"
            
            documents.append(text_to_embed)
            metadatas.append({"clinvar_id": clinvar_id, "gene": data['gene']})
            ids.append(clinvar_id)
            embeddings.append(self._get_embedding(text_to_embed, is_query=False))
            
            # Insertar en bloques de 5000 para no saturar la memoria RAM
            if len(documents) >= 5000:
                self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
                documents, metadatas, ids, embeddings = [], [], [], []

        # Insertar el remanente
        if documents:
            self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            
        print(f"Indexacion completada. {self.collection.count()} variantes almacenadas en disco de forma permanente.")

    def _vector_search(self, query: str) -> str:
        """Busca el vector en el disco."""
        query_vector = self._get_embedding(query, is_query=True)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=5 # Recuperamos 5 contextos (30 es demasiado para el contexto de phi3)
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            print("\n---\n".join(results['documents'][0]))
            return "\n---\n".join(results['documents'][0])
        return None

    def chat(self, user_query: str) -> str:
        """Orquesta el RAG."""
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