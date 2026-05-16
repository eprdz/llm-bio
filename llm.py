import ollama
import chromadb
import gzip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

class GenomicVectorMockAgent:
    def __init__(self):
        self.llm_model = "phi3"
        
        # STRATEGY 3: Native local embeddings. 
        # This loads the Nomic model directly into RAM. No HTTP calls to Ollama.
        # Note: It will download the model weights the very first time you run it.
        print("Loading native embedding model into memory...")
        self.embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

        self.chroma_client = chromadb.PersistentClient(path="./clinvar_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="clinical_variants")
        
        count = self.collection.count()
        if count == 0:
            print("Vector database is empty. Reading VCF and generating vectors...")
            self._stream_and_vectorize("clinvar_patho.vcf.gz")
        else:
            print(f"Database loaded from disk instantly. {count} variants available.")

    def _get_single_embedding(self, text: str, is_query: bool = False):
        """Generates a single vector using the native SentenceTransformer model."""
        prefix = "search_query: " if is_query else "search_document: "
        # .encode() returns a numpy array, we convert it to a standard python list for ChromaDB
        return self.embed_model.encode(prefix + text).tolist()

    def _process_batch_background(self, documents, metadatas, ids):
        """Worker Function: Executes on background threads."""
        try:
            # Generate embeddings locally using all available CPU cores via PyTorch
            embeddings = self.embed_model.encode(documents, show_progress_bar=False).tolist()
            
            # Write to disk asynchronously
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            return len(ids)
        except Exception as e:
            print(f"\nError processing background batch: {e}")
            return 0

    def _stream_and_vectorize(self, clinvar_file):
        """Streaming + Multithreading + Native Embeddings for maximum CPU throughput."""
        print(f"\nStarting parallel vectorization from {clinvar_file}...")
        
        # We can increase the batch size safely now because native embeddings are highly efficient
        batch_size = 1000 
        documents = []
        metadatas = []
        ids = []
        
        # STRATEGY 1: Thread Pool
        # We limit to 3 workers to avoid CPU thrashing, as PyTorch already multithreads internally.
        max_workers = 3
        futures = []
        
        with gzip.open(clinvar_file, "rt", encoding="utf-8") as f, \
             ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            for line in tqdm(f, desc="Reading VCF and Queuing Batches", unit=" lines"):
                if line.startswith('#'):
                    continue
                
                columns = line.strip().split('\t')
                if len(columns) < 8:
                    continue
                
                clinvar_id = columns[2]
                info_field = columns[7]
                
                info_dict = {}
                for item in info_field.split(';'):
                    if '=' in item:
                        k, v = item.split('=', 1)
                        info_dict[k] = v
                    else:
                        info_dict[item] = True

                geneinfo = info_dict.get("GENEINFO", "")
                clndn = info_dict.get("CLNDN", "")
                clnsig = info_dict.get("CLNSIG", "")
                mc = info_dict.get("MC", "")
                rs = info_dict.get("RS", "")
                hgvs = info_dict.get("CLNHGVS", "")
                
                gene = geneinfo.split(':')[0] if geneinfo else ""
                condition = clndn.split('|')[0].split(',')[0].replace('_', ' ') if clndn else ""
                significance = clnsig.split('|')[0].replace('_', ' ') if clnsig else ""
                type_of_variation = mc.split('|')[0] if mc else ""
                rsid = f"rs{rs}" if rs else ""
                
                text_to_embed = f"search_document: The variant {clinvar_id} in the gene {gene} is associated with the condition {condition}. Type of variation: {type_of_variation}. Rsid: {rsid} Clinical classification: {significance}. HGVS: {hgvs}"
                
                documents.append(text_to_embed)
                metadatas.append({
                    "clinvar_id": clinvar_id, 
                    "gene": gene.upper(),
                    "condition": condition.lower(),
                    "significance": significance.lower()
                })
                ids.append(clinvar_id)
                
                if len(documents) >= batch_size:
                    # Submit the batch to the background workers and do NOT wait
                    future = executor.submit(self._process_batch_background, documents, metadatas, ids)
                    futures.append(future)
                    
                    # Clear RAM immediately for the next batch
                    documents, metadatas, ids = [], [], []

            # Submit any remaining data at the end of the file
            if documents:
                future = executor.submit(self._process_batch_background, documents, metadatas, ids)
                futures.append(future)
            
            # Wait for all background threads to finish writing to disk
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing vectors to disk"):
                future.result()
                    
        print(f"\nParallel indexing completed. {self.collection.count()} variants stored.")

    def exact_metadata_search(self, filter_dict: dict) -> str:
        results = self.collection.get(where=filter_dict)
        amount = len(results['ids'])
        
        if amount == 0:
            return f"I have not found any variant that matches the provided filters."
        
        report = f"I found {amount} variants with those criteria:\n"
        report += "-" * 60 + "\n"
        
        limit = min(amount, 10) 
        for i in range(limit):
            var_id = results['ids'][i]
            doc_text = results['documents'][i] 
            clean_text = doc_text.replace("search_document: ", "").split("Clinical classification:")[0].strip()
            report += f"[{i+1}] Variant: {var_id} -> {clean_text}\n"
            
        if amount > 10:
            report += f"... and {amount - 10} more variants hidden to save space.\n"
            
        return report

    def _vector_search(self, query: str) -> str:
        query_vector = self._get_single_embedding(query, is_query=True)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=5 
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            clean_docs = [doc.replace("search_document: ", "") for doc in results['documents'][0]]
            return "\n---\n".join(clean_docs)
        return None

    def chat(self, user_query: str) -> str:
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
            # We still use Ollama here because generating human language requires the LLM
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={"temperature": 0.1}
            )
            return response['message']['content']
        except Exception as e:
            return f"Connection error: {e}"