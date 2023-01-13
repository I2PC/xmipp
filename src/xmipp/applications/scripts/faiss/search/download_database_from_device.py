import faiss

def download_database_from_device(index: faiss.Index):
    return faiss.index_gpu_to_cpu(index)