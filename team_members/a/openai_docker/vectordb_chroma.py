import chromadb
from chromadb.config import Settings

class ChromaVectorDB:
    def __init__(self, persist_dir="chroma_db", collection_name="rag_collection"):
        # Chroma 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=persist_dir)
        # 컬렉션 생성 (없으면 새로 생성)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def save(self, chunks, vectors):
        # 기존 데이터 초기화 (중복 방지) 
        # self.collection.delete(where={})# {} 전체 삭제 
        self.client.delete_collection("rag_collection")
        self.collection = self.client.create_collection("rag_collection")

        # ChromaDB에 저장
        ids = [str(c["chunk_id"]) for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"project": c["project"], "file": c["file"], "page": c["page"]} for c in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors  # ✅ 임베더에서 반환한 벡터 리스트
        )
    def save_incremental(self, chunks, vectors):
        ids = [str(c["chunk_id"]) for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"project": c["project"], 
        "file": c["file"], 
        "page": c["page"], 
        "file_hash": c["file_hash"]} for c in chunks]

        self.collection.add(
            ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors
    )

    def search(self, query_vector, top_k=3, where=None):
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            # where=where if where else {}
            where=where if where else None
        )
        # 반환 구조 맞추기
        hits = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hits.append({"text": doc, **meta})
        return hits
