# global_features_web
Faiss + Pytorch + FastAPI + LMDB <br>
Uses BEit ViT B/16, PCAw, aQE <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id <br>

You should install torch yourself https://pytorch.org/get-started/locally/.  
```bash
pip3 install -r requirements.txt
```

```generate_global_features.py ./path_to_img_folder``` -> generates features  
```train.py``` -> trains index  
```add_to_index.py``` -> adds features from lmdb to index  (if ./trained.index is not found, defaults to Flat Index)  
```global_features_web.py``` -> web microservice  


