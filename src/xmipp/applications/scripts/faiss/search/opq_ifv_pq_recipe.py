import math

def opq_ifv_pq_recipe(dim: int, size: int = int(3e6), c: float = 16):
    """ 
    Values selected using:
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    # Determine the parameters
    PQ_BLOCK_SIZE = 4 # Coefficients per PQ partition
    PQ_MAX_BYTES_PER_VECTOR = 48 # For single precision
    PQ_MAX_VECTOR_SIZE = PQ_BLOCK_SIZE * PQ_MAX_BYTES_PER_VECTOR 
    d = min(PQ_MAX_VECTOR_SIZE, dim//PQ_BLOCK_SIZE*PQ_BLOCK_SIZE)
    m = d // PQ_BLOCK_SIZE # Bytes per vector. 48 is the max for GPU and single precision
    k = c * math.sqrt(size) # Number of clusters
    k = 2 ** math.ceil(math.log2(k)) # Use a power of 2
    
    # Elaborate the recipe for the factory
    opq = f'OPQ{m}_{d}' # Only supported on CPU
    ifv = f'IVF{k}' # In order to support GPU do not use HNSW32
    pq = f'PQ{m}'
    recipe = ','.join((opq, ifv, pq))
    
    return recipe