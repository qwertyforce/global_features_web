import pickle
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import lmdb
dim = 768
pca = PCA(n_components=dim, whiten=True, copy=False)

DB_descriptors = lmdb.open("features.lmdb", readonly=True)
batch_size=DB_descriptors.stat()["entries"]
print(f"batch_size = {batch_size}")
features = np.zeros((batch_size, dim),np.float32)
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
pca.fit(features)
with open('pca_w.pkl', 'wb') as handle:
    pickle.dump(pca, handle)