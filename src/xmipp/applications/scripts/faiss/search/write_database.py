import faiss

from .download_database_from_device import download_database_from_device

def write_database(index: faiss.Index, path: str) -> faiss.Index:
    if faiss.isGpuIndex(index):
        index = download_database_from_device(index)
    
    faiss.write_index(index, path)
    return index