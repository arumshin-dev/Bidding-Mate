from rag import RAGPipeline
import os, traceback
import unicodedata

def safe_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8", "ignore").decode("utf-8")

def main():
    print(">>> inference.py 시작됨")
    # print(">>> OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    try:
        print(">>> RAGPipeline 초기화")
        # 최초 1회만 실행해서 벡터DB 생성 
        rag = RAGPipeline("pdf4db.pkl")
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
            if question.lower() in ["exit", "quit"]: 
                break 
            answer = rag.query(question) 
            print("답변:", answer)

        # 챗봇 모드(벡터 DB 생성 없이 바로 챗봇 실행)
        # rag = RAGPipeline("pdf4db.pkl") 
        # rag.build() 
        # while True: 
        #     question = input("질문: ") 
        #     if question.lower() in ["exit", "quit"]: 
        #         break 
        #     answer = rag.query(question) 
        #     print("답변:", answer)
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
