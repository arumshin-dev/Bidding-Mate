from openai import OpenAI
from dotenv import load_dotenv
import os, traceback

load_dotenv() # .env 파일 로드
# print("DEBUG OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")) # 디버깅용
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(chunks):
    vectors = []
    for chunk in chunks:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk["text"] # ✅ 텍스트만 전달
            ).data[0].embedding
            # vectors.append({ 
            #     "project": chunk["project"], 
            #     "file": chunk["file"], 
            #     "page": chunk["page"], 
            #     "chunk_id": chunk["chunk_id"], 
            #     "embedding": emb 
            # })
            vectors.append(emb) # ✅ 벡터만 저장
        except Exception as e: 
            print(f"❌ 임베딩 실패 (chunk_id={chunk.get('chunk_id')}): {e}")
            traceback.print_exc()
    return vectors

