from rag import RAGPipeline

def main():
    print("inference start")

    # 최초 1회만 실행해서 벡터DB 생성 
    rag = RAGPipeline("pdf4db.pkl")
    rag.build("/work/data/raw")#실제로 데이터를 읽음

    question = "입찰 공고 조건 요약해줘"
    answer = rag.query(question)

    print("답변:", answer)

    # 챗봇 모드 
    while True: 
        question = input("질문: ") 
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

if __name__ == "__main__":
    main()
