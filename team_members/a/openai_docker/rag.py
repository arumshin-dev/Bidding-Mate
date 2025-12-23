from loader import load_documents
from chunker import chunk_text
from embedder import get_embeddings
from vectordb import VectorDB
from openai import OpenAI
import os
import unicodedata

def safe_text(text: str) -> str:
    # 유니코드 정규화
    text = unicodedata.normalize("NFC", text)
    # surrogate 코드 제거
    return text.encode("utf-8", "ignore").decode("utf-8")

class RAGPipeline:
    def __init__(self, db_path="vectordb.pkl"):
        self.db = VectorDB(db_path)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # 여기서 생성하는 게 안전함

    def build(self, raw_dir="../../../data/raw"):
        print("📄 pdf 문서 로딩...")
        docs = load_documents(raw_dir)
        # print("문서 개수:", len(docs))
        # print("docs[-1]", docs[-1])
        # print("첫 문서 내용 일부:", docs[-1]['text'][:200] if docs else "문서 없음")
        
        print("✂️ Chunking...")
        chunks = chunk_text(docs)
        
        print("🧠 Embedding...")
        print("🔧 임베딩 생성 시작")
        vectors = get_embeddings(chunks)
        print("🔧 임베딩 생성 완료")
        print("💾 Saving to VectorDB...")
        self.db.save(chunks, vectors)
        print("✅ RAG build complete")

    def query(self, question, top_k=3):
        print("🔍 Searching similar chunks...")

        # 1) 질문을 임베딩으로 변환 (문자열을 직접 넘겨야 하므로 따로 처리) 
        q_vec = self.client.embeddings.create( 
            model="text-embedding-3-small", 
            input=safe_text(question )
        ).data[0].embedding

        # 2) 벡터로 검색 
        top_chunks = self.db.search(q_vec, top_k=3)
        print("DEBUG top_chunks:", top_chunks) # 구조 확인용
        if not top_chunks: 
            return "맥락이 부족해 답변하기 어렵습니다. 더 구체적인 질문이나 관련 문서를 제공해 주세요."
        # 3) 프롬프트 구성 
        # dict에서 text만 꺼내서 context 구성
        context = "\n\n".join([c["text"] for c in top_chunks])
        # prompt = f"다음 내용을 참고해서 질문에 답변해줘:\n\n{context}\n\n질문: {question}"
        prompt = f'''
        입찰지원 전문가입니다. 
        입찰공고에 대해 아래에 주어진 맥락을 이용해 질문에 대해 답변해 주세요. 
        주어진 맥락으로 답변이 어렵다면 모른다고 답하세요. 억지로 추론하지 마세요.
        반드시 한국어로 답변해 주세요.

        맥락:
        {context}

        질문:
        {question}
        '''
        print("🤖 Generating answer...")
        # 4) LLM 호출
        completion = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = completion.choices[0].message.content 
        # 4) 출처 표시 (사람이 읽기 좋게) 
        sources = "\n".join([f"- {c['project']} ({c['file']}, p.{c['page']})" for c in top_chunks])
        
        return f"{answer}\n\n출처:\n{sources}"

'''
python3 - << 'EOF'
from rag import RAGPipeline

rag = RAGPipeline("testdb.pkl")
rag.build("../../../data/raw")

answer = rag.query("입찰 공고 조건 요약해줘")
print("답변:", answer)
EOF

'''
