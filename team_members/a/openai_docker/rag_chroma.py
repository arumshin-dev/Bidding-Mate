from loader import load_documents
from chunker import chunk_text
from embedder import get_embeddings
# from chromadb import PersistentClient
# from vectordb import ChromaVectorDB 
from vectordb_chroma import ChromaVectorDB 
from openai import OpenAI 
import os, unicodedata
import hashlib 

def file_hash(path): 
    # NFC 정규화 
    # path = unicodedata.normalize("NFC", path) 
    if not os.path.exists(path): 
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with open(path, "rb") as f: 
        return hashlib.md5(f.read()).hexdigest()
    # import unicodedata, os, hashlib 
    # path_nfc = unicodedata.normalize("NFC", path) 
    # path_nfd = unicodedata.normalize("NFD", path) 
    # for candidate in [path_nfc, path_nfd]: 
    #     if os.path.exists(candidate): with open(candidate, "rb") as f: 
    #         return hashlib.md5(f.read()).hexdigest() 
    # raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

def safe_text(text: str) -> str: 
    text = unicodedata.normalize("NFC", text) 
    return text.encode("utf-8", "ignore").decode("utf-8")

class RAGPipeline:
    def __init__(self, persist_dir="chroma_db", collection_name="rag_collection"):
        # ChromaDB 초기화
        # self.client = PersistentClient(path=persist_dir)
        # self.collection = self.client.get_or_create_collection(name="rag_collection")
        # self.db = ChromaVectorDB(persist_dir, collection_name) 
        self.db = ChromaVectorDB(persist_dir="/work/chroma_db", collection_name="rag_collection")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build(self, raw_dir="/work/data/raw"):
        print("📄 문서 로딩...")
        docs = load_documents(raw_dir)

        print("✂️ 청킹...")
        chunks = chunk_text(docs)

        # print("🧠 임베딩 생성...")
        # vectors = get_embeddings(chunks)  # ✅ 벡터만 반환
        # DB에 이미 저장된 파일 해시 가져오기 
        existing = self.db.collection.get() 
        existing_hashes = {m.get("file_hash") for m in existing["metadatas"]} 
        print("DEBUG existing_hashes:", existing_hashes) 
        new_chunks = [] 
        new_vectors = [] 
        # DB에 이미 저장된 파일 건너뛰기 
        for chunk in chunks: 
             
            h = file_hash(chunk["filepath"]) 
            if h in existing_hashes: 
                continue # 이미 처리된 문서 건너뛰기 
            chunk["file_hash"] = h 
            new_chunks.append(chunk) 
        if new_chunks: 
            print("🧠 임베딩 생성...")
            vectors = get_embeddings(new_chunks) 
            print("💾 ChromaDB에 저장...")
            self.db.save_incremental(new_chunks, vectors)
        # print("💾 ChromaDB에 저장...")
        # ids = [str(c["chunk_id"]) for c in chunks]
        # texts = [c["text"] for c in chunks]
        # metadatas = [{"project": c["project"], "file": c["file"], "page": c["page"]} for c in chunks]

        # self.collection.add(
        #     ids=ids,
        #     documents=texts,
        #     metadatas=metadatas,
        #     embeddings=vectors
        # )
        # self.db.save(chunks, vectors)
        print("✅ RAG build complete")

    def query(self, question, embedder_model="text-embedding-3-small", top_k=3, where=None):
        # from openai import OpenAI
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("🔍 질문 임베딩 생성...")
        q_vec = self.client.embeddings.create(
            model=embedder_model,
            input=question
        ).data[0].embedding

        print("🔎 ChromaDB 검색...")
        # results = self.collection.query(
        #     query_embeddings=[q_vec],
        #     n_results=3
        # )

        # context 구성
        # context = "\n\n".join(results["documents"][0])

        # prompt = f"""
        # 입찰 전문가입니다. 아래 맥락을 참고해 질문에 답변해 주세요.
        # 맥락: {context}
        # 질문: {question}
        # """
        top_chunks = self.db.search(q_vec, top_k=top_k, where=where)
        print("DEBUG top_chunks:", top_chunks) 
        if not top_chunks: 
            return "관련 맥락을 찾지 못했습니다." 
        
        context = "\n\n".join([safe_text(c["text"]) for c in top_chunks]) 
        prompt = f"""입찰 지원 전문가입니다. 
        아래 맥락을 참고해 질문에 답변해 주세요. 
        맥락: {context} 
        질문: {question}"""
        print("🤖 답변 생성...")
        completion = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = completion.choices[0].message.content
        # 출처 표시 (사람이 읽기 좋게) 
        sources = "\n".join([f"- {c['project']} ({c['file']}, p.{c['page']})" for c in top_chunks])
        
        return f"{answer}\n\n출처:\n{sources}"
