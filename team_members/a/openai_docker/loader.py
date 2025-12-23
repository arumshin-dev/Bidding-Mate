import os, sys, glob, re, unicodedata
import pandas as pd
import fitz # PyMuPDF

# -----------------------------
# 1) PDF 파싱 안정화
# -----------------------------
def safe_get_page_text(page):
    try:
        return page.get_text("text")
    except Exception:
        try:
            blocks = page.get_text("blocks") or []
            return "\n".join(b[4] for b in blocks if len(b) > 4 and b[4])
        except Exception:
            return ""
            
def clean_text_final(text: str, keep_newline=False) -> str:
    # 전처리 
    if not isinstance(text, str):
        return ""
    text = text.strip()# 앞뒤 공백 제거
    # text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")# 줄바꿈, 캐리지리턴-공백, 탭 공백으로 치환
    if not keep_newline: 
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ") 
    else: 
        text = text.replace("\r", " ").replace("\t", " ") # 줄바꿈은 유지
    text = re.sub(r"[-=]{3,}", " ", text)# 구분선 제거 (---, === 등)
    text = re.sub(r"\s*-\s*\d+\s*-\s*", " ", text)# 페이지 번호 패턴 제거 (예: -1-, -23-)
    text = re.sub(r"[·\.]{3,}", " ", text)# 점점점 (··· 또는 ...) 공백으로 대체(예:목차, 목차...)
    text = re.sub(r"\s+", " ", text)# 공백이 2번 이상 연속된 부분-불필요한 공백 압축
    return text.strip()
def clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""

def load_documents(raw_dir):
    # 1. PDF 파일 목록
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    pdf_files_map = {
        unicodedata.normalize("NFC", os.path.basename(f)): f
        for f in pdf_files
    }
    
    # 2. data_list.csv 로드
    csv_path = os.path.join(raw_dir, "data_list.csv")
    df = pd.read_csv(csv_path)
    pdf_df = df[df["파일형식"] == "pdf"].copy()
    pdf_df["파일명_normalized"] = pdf_df["파일명"].apply(
        lambda x: unicodedata.normalize("NFC", x)
    )
    print("PDF 파일 개수:", len(pdf_df))
    print("PDF 파일 목록:", pdf_df["파일명_normalized"].tolist())
    docs = []

    # 3. PDF와 사업명 매핑
    for _, row in pdf_df.iterrows():
        fname_nfc = row["파일명_normalized"]
        if fname_nfc in pdf_files_map:
            pdf_path = pdf_files_map[fname_nfc]
            try:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    # text = clean_text_final(page.get_text())
                    text = clean_text(safe_get_page_text(page))
                    if text:
                        docs.append({
                            "project": row["사업명"],   # 사업명 추가
                            "file": fname_nfc,
                            "page": page_num + 1,
                            "text": text
                        })
                doc.close()
            except Exception as e:
                print(f"❌ PDF 열기 실패: {fname_nfc} ({e})")

    return docs

'''
python3 - << 'EOF'
from loader import load_documents
docs = load_documents("../../../data/raw")
print("문서 개수:", len(docs))
print("docs[0]", docs[0])
print("첫 문서 내용 일부:", docs[0]['text'][:200] if docs else "문서 없음")
EOF
'''