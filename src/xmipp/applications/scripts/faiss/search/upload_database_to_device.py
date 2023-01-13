import faiss
import torch

def upload_database_to_device(index: faiss.Index, 
                              device: torch.device, 
                              use_f16: bool = False ) -> faiss.Index:
    
    if device.type == 'cuda':
        resources = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = use_f16
        
        index = faiss.index_cpu_to_gpu(
            resources,
            device.index,
            index,
            co
        )
        
    elif device.type != 'cpu':
        raise ValueError('Input device must be CUDA or CPU')

    return index