# def chunk_text(docs, chunk_size=300):
def chunk_text(docs, chunk_size=300, overlap=50):
    chunks = []
    for doc in docs:
        text = doc["text"].strip()
        if not text:
            continue
        i = 0
        chunk_id = 0
        while i < len(text):
            chunk_text = text[i:i+chunk_size].strip()
            if chunk_text:
                chunks.append({
                    "project": doc["project"],   # 사업명 추가
                    "file": doc["file"],
                    "page": doc["page"],
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "length": len(chunk_text),
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1
            i += chunk_size - overlap   # 오버랩 적용
    return chunks

'''
python3 - << 'EOF'
from loader import load_documents
from chunker import chunk_text
# import pandas as pd

raw_dir = "../../../data/raw"
docs = load_documents(raw_dir)
chunks = chunk_text(docs, chunk_size=800)

# chunks_df = pd.DataFrame(chunks)
# print("총 청크 수:", len(chunks_df))
# print(chunks_df.head())
chunks = chunk_text(docs, chunk_size=50)
print("청크 개수:", len(chunks))
print("첫 청크:", chunks[0])
EOF
'''
