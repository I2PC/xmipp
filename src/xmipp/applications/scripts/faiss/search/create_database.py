import faiss
import math

def create_database(dim: int, 
                    recipe: str, 
                    metric_type = faiss.METRIC_L2) -> faiss.Index:
    index: faiss.Index = faiss.index_factory(dim, recipe)
    index.metric_type = metric_type
    return index
            