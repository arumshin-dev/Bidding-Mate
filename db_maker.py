import os
import re
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pdfminer.pdfparser import PDFSyntaxError

# 0. 환경변수 로드
load_dotenv()

# 1. 설정
PDF_FOLDER = "./data/raw/100_PDF"
CSV_PATH = "./data/raw/data_full.csv"
DB_PATH = "./chroma_db_chunk500"

# DB 폴더 초기화
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"기존 DB 폴더({DB_PATH})를 삭제하고 새로 만듭니다.")

# 2. 메타데이터 로드 (파일명 기준 매칭)
print(f"메타데이터 로딩 중... ({CSV_PATH})")
try:
    meta_df = pd.read_csv(CSV_PATH, encoding='utf-8')
    meta_df = meta_df.fillna('')
    
    print(f" -> CSV 컬럼 목록: {list(meta_df.columns)}")
    
    # CSV의 '파일명' 컬럼에서 확장자(.pdf)를 떼고 깨끗하게 다듬어서 인덱스로 만듭니다.
    # 예: "사업명.pdf" -> "사업명"
    meta_df['match_key'] = meta_df['파일명'].astype(str).str.replace(r'\.pdf$', '', regex=True).str.strip()
    
    # 이제 '파일명(match_key)'으로 검색할 수 있게 설정
    meta_df.set_index('match_key', inplace=True)
    
    print(f" -> 총 {len(meta_df)}행의 메타데이터 로드 완료.")
    print(f" -> (참고) 매칭 키 예시 3개: {list(meta_df.index[:3])}")
    
except Exception as e:
    print(f"오류: CSV 파일을 읽을 수 없습니다. ({e})")
    exit()

# 3. 텍스트 청소 함수
def clean_text(text):
    if not text: return ""
    text = text.replace('\r\n', '\n').replace('\t', ' ')
    text = re.sub(r'[\.\-\=_]{3,}', '', text)
    text = re.sub(r'(\b\w+\b)( \1){2,}', r'\1', text)
    text = re.sub(r'(\w{2,})(\1){2,}', r'\1', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n\n', text)
    return text.strip()

# 4. 문서 로드 및 메타데이터 주입
documents = []
print(f"'{PDF_FOLDER}' 폴더에서 PDF 로딩 시작...")

if not os.path.exists(PDF_FOLDER):
    print(f"오류: PDF 폴더를 찾을 수 없습니다.")
    exit()

files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f" -> 대상 파일: {len(files)}개")

success_count = 0

for i, file in enumerate(files):
    file_path = os.path.join(PDF_FOLDER, file)
    
    # 파일명에서 확장자 떼고 공백 제거 (CSV match_key와 똑같이 만듦)
    file_id = os.path.splitext(file)[0].strip()
    
    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        # [디버깅] 처음 3개만 매칭 여부 확인
        if i < 3:
            print(f"[매칭 테스트 {i+1}] 파일명: '{file_id}'")
            if file_id in meta_df.index:
                print(f" ▶ 결과 : ✅ 성공!")
            else:
                print(f" ▶ 결과 : ❌ 실패 (CSV 키 예시: {list(meta_df.index[:1])})")

        # 메타데이터 찾기
        matched_row = None
        if file_id in meta_df.index:
            matched_row = meta_df.loc[file_id]
            success_count += 1
        
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            if "텍스트" in doc.metadata: del doc.metadata["텍스트"]
            doc.metadata["source"] = file 
            
            # 메타데이터 주입
            if matched_row is not None:
                doc.metadata["notice_no"] = str(matched_row.get("공고 번호", "알수없음")).strip()
                doc.metadata["project_name"] = str(matched_row.get("사업명", "알수없음")).strip()
                doc.metadata["budget"] = str(matched_row.get("사업 금액", "0")).strip()
                doc.metadata["agency"] = str(matched_row.get("발주 기관", "알수없음")).strip()
                
        documents.extend(docs)
        
        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(files)}] 진행 중...")
            
    except Exception as e:
        print(f"   [Skip] 오류: {file}")

print(f"\n로드 완료! (메타데이터 매칭 성공: {success_count}/{len(files)})")

# 5. 청킹
print("텍스트 분할 시작...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150, separators=["\n\n", "\n", " ", ""])
split_docs = text_splitter.split_documents(documents)
print(f" -> 총 {len(split_docs)}개의 청크 생성됨")

# 6. 저장
print("벡터 DB 저장 중...")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding_model, persist_directory=DB_PATH)
print(f"\nDB 생성 완료! 경로: {DB_PATH}")