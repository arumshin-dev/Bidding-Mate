import os
import json
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
from ragas.run_config import RunConfig

# 1. 환경변수 로드
load_dotenv()

# GPT-5 등 미래 모델명 대응을 위한 커스텀 클래스
class GPT5ChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Ragas가 무슨 값을 보내든 상관없이 temperature를 1로 강제 고정
        kwargs["temperature"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # 비동기 호출 시에도 temperature를 1로 강제 고정
        kwargs["temperature"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

# 2. RAG 시스템 불러오기
rag_system = BiddingAgent()

# 3. 채점관 설정 (온도 1로 초기화)
judge_llm = GPT5ChatOpenAI(
    model="gpt-5", 
    temperature=1  # 초기값도 1로 설정
)
judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 테스트 데이터 로드 (JSON 파일 불러오기)
json_file_path = "test_data.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"'{json_file_path}'에서 {len(test_data)}개의 테스트 데이터를 불러왔습니다.")
except FileNotFoundError:
    print(f"오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
    exit()
except json.JSONDecodeError:
    print(f"오류: '{json_file_path}' 파일 형식이 올바르지 않습니다.")
    exit()


print("시험을 치고 있습니다...")

questions = []
answers = []
contexts = []
ground_truths = []

# 데이터셋 구성 루프
for item in test_data:
    # RAG 시스템에 질문 던지기
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

my_run_config = RunConfig(timeout=360)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=my_run_config
)

# 7. 결과 출력 및 저장
print(result)

# DataFrame으로 변환
df = result.to_pandas()

# CSV 저장
df.to_csv("ragas_score.csv", index=False)
print("상세 결과가 'ragas_score.csv'로 저장되었습니다.")

# 오답 노트 출력
print("\n=== 오답 노트 및 상세 확인 ===")

def get_col(row, candidates):
    for col in candidates:
        if col in row and row[col] is not None:
            try:
                import pandas as pd
                if pd.isna(row[col]): continue
            except: pass
            return row[col]
    return "N/A"

for i, row in df.iterrows():
    # 1. 질문 (Ragas가 user_input으로 바꿈)
    q_text = get_col(row, ['user_input', 'question'])
    
    # 2. AI 답변 (Ragas가 response로 바꿈)
    a_text = get_col(row, ['response', 'answer'])
    
    # 3. [핵심 수정] 정답 (Ragas가 reference로 바꿈!!!)
    gt_text = get_col(row, ['reference', 'ground_truth', 'ground_truths'])
    
    # 4. 참고 문서
    ctx_text = get_col(row, ['retrieved_contexts', 'contexts'])
    
    print(f"\nQ{i+1}: {q_text}")
    print(f"정답(GT): {gt_text}")
    print(f"AI 답변: {a_text}")
    
    if isinstance(ctx_text, list) and len(ctx_text) > 0:
        print(f"가져온 문서(Context): {str(ctx_text[0])[:100]}...") 
    elif isinstance(ctx_text, str):
        print(f"가져온 문서(Context): {ctx_text[:100]}...")
    else:
        print("가져온 문서(Context): (없음)")