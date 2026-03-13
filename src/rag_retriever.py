from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional

class RAGRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """初始化RAG检索器"""
        # 加载SentenceTransformer模型
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.corpus: Optional[List[str]] = None
        self.neigh: Optional[NearestNeighbors] = None
    
    def build_index(self, corpus: List[str]) -> None:
        """构建向量索引"""
        self.corpus = corpus
        # 生成嵌入
        self.embeddings = self.model.encode(corpus)
        # 构建KNN索引
        self.neigh = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.neigh.fit(self.embeddings)
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """检索与查询最相似的文本"""
        if self.neigh is None:
            raise ValueError("Index not built. Please call build_index first.")
        
        # 生成查询嵌入
        query_embedding = self.model.encode([query])
        # 检索最相似的k个文本
        distances, indices = self.neigh.kneighbors(query_embedding, n_neighbors=k)
        
        # 返回检索结果
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.corpus[idx],
                "distance": distances[0][i]
            })
        
        return results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """获取查询的上下文"""
        results = self.retrieve(query, k)
        context = "\n".join([result["text"] for result in results])
        return context
