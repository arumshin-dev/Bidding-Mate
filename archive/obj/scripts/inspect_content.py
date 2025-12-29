import os
from langchain_community.document_loaders import PDFPlumberLoader

PDF_FOLDER = "./data/raw/100_PDF"

def inspect_content():
    print(f"🧐 텍스트 내용 품질 검사를 시작합니다... (경로: {PDF_FOLDER})\n")
    
    if not os.path.exists(PDF_FOLDER):
        print("❌ 폴더가 없습니다.")
        return

    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    suspicious_files = [] # 의심스러운 파일 목록
    
    for i, file in enumerate(files):
        file_path = os.path.join(PDF_FOLDER, file)
        try:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            
            # 전체 텍스트 합치기
            full_text = "".join([doc.page_content for doc in docs])
            cleaned_text = full_text.strip()
            text_len = len(cleaned_text)
            
            # 🚨 기준: 페이지는 있는데 글자가 50자 미만이면 '스캔본' 의심
            if len(docs) > 0 and text_len < 50:
                print(f"⚠️ [스캔 의심] {file}")
                print(f"   ㄴ 페이지: {len(docs)}장 / 글자수: {text_len}자")
                suspicious_files.append(file)
            
            # ✅ 정상 파일 확인 (10개마다 하나씩)
            elif i % 10 == 0:
                # [수정] 백슬래시 에러 방지를 위해 변수에서 먼저 처리
                preview = cleaned_text[:50].replace('\n', ' ')
                
                print(f"✅ [내용 확인] {file[:20]}... ({text_len}자)")
                print(f"   ㄴ 미리보기: {preview}...")

        except Exception as e:
            pass 

    print(f"\n{'='*40}")
    print(f"결과 리포트")
    print(f"총 파일: {len(files)}개")
    print(f"스캔 의심(텍스트 없음): {len(suspicious_files)}개")
    
    if suspicious_files:
        print("\n🗑️ 다음 파일들은 OCR이 필요하거나 내용을 읽을 수 없습니다:")
        for f in suspicious_files:
            print(f" - {f}")

if __name__ == "__main__":
    inspect_content()