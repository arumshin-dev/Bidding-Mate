from rag_chroma import RAGPipeline, safe_text
import os, traceback
import unicodedata

def main():
    print(">>> inference_chroma.py 시작됨")
    # print(">>> OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    try:
        print(">>> RAGPipeline 초기화+chromadb")
        # 최초 1회만 실행해서 벡터DB 생성 
        # rag = RAGPipeline("pdf4db.pkl")
        # ✅ ChromaDB 기반 RAGPipeline 초기화 
        # rag = RAGPipeline(persist_dir="chroma_db", collection_name="rag_collection")
        rag = RAGPipeline(persist_dir="/work/chroma_db", collection_name="rag_collection")
        print(">>> build 실행")
        rag.build("/work/data/raw")#실제로 데이터를 읽음
        print(">>> build 완료")

        print(">>> 첫 질문 실행")
        question = "입찰 공고 조건 요약해줘"
        answer = rag.query(question)

        print(">>> 답변:", answer)

        # 챗봇 모드 
        while True: 
            question = input("질문: ") 
            question = safe_text(question).strip()
            q_lower = question.lower() 
            if q_lower.startswith("exit") or q_lower.startswith("quit"): 
                break 

            try:
                # 필요하다면 where 조건도 전달 가능 
                # 예: rag.query(question, where={"project":"2025년도 중이온가속기"})
                answer = rag.query(question) 
                print("답변:", answer)
            except Exception as e:
                print("Error during query:", e)
                traceback.print_exc()

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
