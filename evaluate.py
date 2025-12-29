import os
from dotenv import load_dotenv
from rag_core import BiddingAgent
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. 환경변수 로드
load_dotenv()

class GPT5ChatOpenAI(ChatOpenAI):
    # 1. 동기 호출 방어
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._fix_temperature(kwargs)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    # 2. 비동기(Async) 호출 방어
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._fix_temperature(kwargs)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    # 온도 수정 함수 (무조건 1.0으로 만듦)
    def _fix_temperature(self, kwargs):
        if "temperature" in kwargs:
            del kwargs["temperature"]
        self.temperature = 1.0
        if self.model_kwargs and "temperature" in self.model_kwargs:
             del self.model_kwargs["temperature"]

# 2. RAG 시스템 불러오기
rag_system = BiddingAgent()

# 3. 채점관 설정
judge_llm = GPT5ChatOpenAI(model="gpt-5") 
judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 테스트 데이터
# 이곳에 테스트 질문과 답변 넣기
test_data = [
    {
        "question": "입찰 자격은 어떻게 돼?",
        "ground_truth": "입찰참가자격은 관련 법령에 따른 경쟁입찰 참가 자격을 갖춘 자로서, 소프트웨어사업자 신고를 필하고 공동수급이 허용됩니다."
    },
    {
        "question": "보안 관련 요구사항을 요약해줘.",
        "ground_truth": "보안 서약서 제출, 자료 유출 금지, 웹 취약점 점검 수행, 시큐어 코딩 적용 등이 요구됩니다."
    }
]

print("시험을 치고 있습니다...")

questions = []
answers = []
contexts = []
ground_truths = []

for item in test_data:
    result = rag_system.ask_with_context(item["question"])
    questions.append(result["question"])
    answers.append(result["answer"])
    contexts.append(result["contexts"])
    ground_truths.append(item["ground_truth"])

# 5. 데이터셋 변환
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# 6. 채점 시작
print("채점 중입니다...")

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

# 7. 결과 출력
print(result)
df = result.to_pandas()
df.to_csv("ragas_score.csv", index=False)
print("상세 결과가 'ragas_score.csv'로 저장되었습니다.")