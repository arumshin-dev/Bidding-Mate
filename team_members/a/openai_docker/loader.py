import os, sys, glob, re, fitz, unicodedata
# PyMuPDF 경고 메시지 숨기기 
devnull = open(os.devnull, 'w') 
sys.stderr = devnull

def clean_text_final(text: str) -> str:
    # 전처리 
    if not isinstance(text, str):
        return ""
    text = text.strip()# 앞뒤 공백 제거
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")# 줄바꿈, 캐리지리턴-공백, 탭 공백으로 치환
    text = re.sub(r"[-=]{3,}", " ", text)# 구분선 제거 (---, === 등)
    text = re.sub(r"\s*-\s*\d+\s*-\s*", " ", text)# 페이지 번호 패턴 제거 (예: -1-, -23-)
    text = re.sub(r"[·\.]{3,}", " ", text)# 점점점 (··· 또는 ...) 공백으로 대체(예:목차, 목차...)
    text = re.sub(r"\s+", " ", text)# 공백이 2번 이상 연속된 부분-불필요한 공백 압축
    return text.strip()

def load_documents(raw_dir):
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    pdf_files_map = {
        unicodedata.normalize("NFC", os.path.basename(f)): f
        for f in pdf_files
    }
    docs = []
    for fname_nfc, pdf_path in pdf_files_map.items():
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = clean_text_final(page.get_text())
                if text:
                    docs.append({
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