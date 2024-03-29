from tqdm import tqdm
import numpy as np
import lmdb
import math
import faiss
from os import remove
import pickle
from pathlib import Path

dim = 512
DB_descriptors = lmdb.open("./data/features.lmdb", readonly=True)
entries = DB_descriptors.stat()['entries']
nlist = int(math.sqrt(entries))
print(f"entries = {entries}")
print(f"nlist = {nlist}")
#index = faiss.index_factory(dim,f"OPQ64,IVF{nlist},PQ64",faiss.METRIC_L2)
index = faiss.index_factory(dim,f"OPQ256,PQ256x8",faiss.METRIC_L2)
index = faiss.IndexIDMap2(index)
batch_size= min(entries,5_000_000)

pca = None
pca_w_file = Path("./data/pca_w.pkl")
if pca_w_file.is_file():
    with open(pca_w_file, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)
        USE_PCA = True
        print("USING PCA")
if pca is None:
    USE_PCA = False
    print("pca_w.pkl not found. Proceeding without PCA")

if batch_size == 5_000_000: # ~16gb
    features = np.memmap('train.mmap', dtype='float32', mode='w+', shape=(batch_size, dim))
else:
    features = np.zeros((batch_size, dim),dtype='float32')
def get_data():
    with DB_descriptors.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            retrieved = 0
            for data in tqdm(curs.iternext(keys=False, values=True),total=batch_size):
                if retrieved == batch_size:
                    return
                features[retrieved] = np.frombuffer(data,dtype=np.float32)
                retrieved+=1
get_data()
if pca:
    features = pca.transform(features)
print("Training......")
from timeit import default_timer as timer
start = timer()
index.train(features)
end = timer()
print(end - start)
del features
try:
    remove("train.mmap")
except:
    pass    
faiss.write_index(index,"./data/trained.index")