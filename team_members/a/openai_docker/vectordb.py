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

    def search(self, query_vector, top_k=3): 
        q = np.array(query_vector) 
        sims = [] 
        for entry in self.entries: 
            # v = np.array(entry["embedding"]) 
            v = entry["embedding"] # 이미 numpy array
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)) 
            sims.append((sim, entry))
            
        sims.sort(reverse=True, key=lambda x: x[0]) 
        return [c for _, c in sims[:top_k]]

