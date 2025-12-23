import pickle
import numpy as np

class VectorDB:
    def __init__(self, path="vectordb.pkl"):
        self.path = path
        self.entries = [] # [{...chunk..., "embedding": vector}, ...]

        try:
            with open(path, "rb") as f:
                self.entries = pickle.load(f)
        except:
            pass

    def save(self, chunks, vectors):
        # chunks와 vectors를 합쳐서 저장
        self.entries = []
        for chunk, vec in zip(chunks, vectors): 
            # entry = {**chunk, "embedding": vec} 
            # vec을 numpy array로 강제 변환 
            entry = {**chunk, "embedding": np.array(vec, dtype=float)}
            self.entries.append(entry) 
        with open(self.path, "wb") as f: 
            pickle.dump(self.entries, f)

    def search(self, query_vector, top_k=3, project=None, file=None): 
        q = np.array(query_vector, dtype=float) 
        q_norm = np.linalg.norm(q) # 질의 벡터 q의 크기 norm 계산
        if q_norm == 0: return [] 

        filtered = self.entries # 검색 대상 엔트리들을 담는 리스트
        if project: 
            filtered = [e for e in filtered if e.get("project") == project] 
        if file: 
            filtered = [e for e in filtered if e.get("file") == file]
        
        sims = [] # 검색 결과를 담는 리스트
        for entry in filtered: 
            v = entry["embedding"] # 이미 numpy array
            v_norm = np.linalg.norm(v) 
            if v_norm == 0: 
                continue 
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            # sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)) 
            sims.append((sim, entry))
            
        sims.sort(reverse=True, key=lambda x: x[0]) 
        return [c for _, c in sims[:top_k]]

