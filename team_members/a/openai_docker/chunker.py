from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_text(docs, chunk_size=300, overlap=50):
    # RecursiveCharacterTextSplitter 정의
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n\n", "\n\n", "\n", "。", ". ", " ", ""]  # 한국어 추가
    )

    # 원본 docs는 [{"text": "...", "project": "...", "file": "...", "page": ...}, ...] 형태라고 가정
    langchain_docs = []
    for doc in docs:
        langchain_docs.append(
            Document(
                page_content=doc["text"],
                metadata={
                    "project": doc["project"],
                    "file": doc["file"],
                    "filepath": doc["filepath"],
                    "page": doc["page"]
                }
            )
        )

    # RecursiveCharacterTextSplitter로 청킹
    chunks = splitter.split_documents(langchain_docs)

    # 반환 형식 맞추기 (dict 리스트)
    results = []
    chunk_id = 0
    for c in chunks:
        # 📌 추가: 섹션 타입 자동 감지
        section = "general"
        if "참가자격" in c.page_content[:100]:
            section = "eligibility"
        elif "제출서류" in c.page_content[:100]:
            section = "documents"
        elif "평가" in c.page_content[:100]:
            section = "evaluation"
        
        results.append({
            "project": c.metadata["project"],
            "file": c.metadata["file"],
            "filepath": c.metadata["filepath"],
            "page": c.metadata["page"],
            "chunk_id": chunk_id,
            "text": c.page_content,
            "length": len(c.page_content),
            "word_count": len(c.page_content.split()),
            "section_type": section,  # 📌 추가
        })
        chunk_id += 1

    return results
