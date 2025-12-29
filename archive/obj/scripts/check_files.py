import os
import traceback
from langchain_community.document_loaders import PDFPlumberLoader
from pdfminer.pdfparser import PDFSyntaxError

# ✅ 사용자가 설정한 경로
PDF_FOLDER = "./data/raw/100_PDF"

def check_pdf_health():
    print(f"🏥 PDF 파일 건강검진을 시작합니다... (경로: {PDF_FOLDER})\n")
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ 폴더가 없습니다! 경로를 확인해주세요.")
        return

    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    total = len(files)
    
    if total == 0:
        print("⚠️ PDF 파일이 하나도 없습니다!")
        return

    success_cnt = 0
    fail_cnt = 0
    bad_files = []

    print(f"총 {total}개의 파일을 검사합니다. 잠시만 기다려주세요...\n")

    for i, file in enumerate(files):
        file_path = os.path.join(PDF_FOLDER, file)
        # 진행률 표시 (한 줄에 출력)
        print(f"\r[{i+1}/{total}] 검사 중: {file[:30]}...", end="")

        try:
            # 1. 로드 시도
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()

            # 2. 내용 확인
            if not docs or len(docs) == 0:
                raise ValueError("페이지가 없거나 텍스트가 비어있음")
            
            # 성공
            success_cnt += 1

        except PDFSyntaxError:
            print(f"\n❌ [손상됨] {file}")
            fail_cnt += 1
            bad_files.append(file)
        except ValueError as ve:
            print(f"\n⚠️ [빈 파일] {file} ({ve})")
            fail_cnt += 1
            bad_files.append(file)
        except Exception as e:
            print(f"\n🚫 [에러] {file} : {e}")
            fail_cnt += 1
            bad_files.append(file)

    print(f"\n\n{'='*40}")
    print(f"🎉 검사 완료!")
    print(f"✅ 정상 파일: {success_cnt}개")
    print(f"❌ 문제 파일: {fail_cnt}개")
    
    if bad_files:
        print(f"\n🗑️ 문제 있는 파일 목록 (확인 후 삭제하거나 다시 저장하세요):")
        for bad in bad_files:
            print(f" - {bad}")
    else:
        print("\n✨ 모든 파일이 아주 건강합니다! 바로 DB 구축하셔도 됩니다.")
    print(f"{'='*40}")

if __name__ == "__main__":
    check_pdf_health()