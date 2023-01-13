import faiss

def read_database(path: str) -> faiss.Index:
    return faiss.read_index(path)