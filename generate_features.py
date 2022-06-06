import numpy as np
from os import listdir
from tqdm import tqdm
import lmdb
from PIL import Image
import torch
import timm
from torchvision import transforms


DB = lmdb.open('./features.lmdb',map_size=5000*1_000_000) #5000mb
IMAGE_PATH = "./../images"


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def check_if_exists_by_id(id):
    with DB.begin(buffers=True) as txn:
        x = txn.get(int_to_bytes(id),default=False)
        if x:
            return True
        return False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)
model.head=torch.nn.Identity()
model.eval()
model.to(device)

_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def transform(im):
  desired_size = 224
  old_size = im.size  # old_size[0] is in (width, height) format
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  im = im.resize(new_size, Image.ANTIALIAS)
  new_im = Image.new("RGB", (desired_size, desired_size))
  new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
  return _transform(new_im)

  
def get_features(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(image).cpu().numpy()[0]
    feature_vector/=np.linalg.norm(feature_vector)
    return feature_vector

def read_img_file(f):
    img = Image.open(f)
    img=img.convert('L').convert('RGB') #GREYSCALE
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    return img
    
def calc_nn_features(file_name):
    file_id = int(file_name[:file_name.index('.')])
    img_path = IMAGE_PATH+"/"+file_name
    try:
        query_image=read_img_file(img_path)
    except:
        print(f'error reading {img_path}')
        return None
    image_features = get_features(query_image)
    return (int_to_bytes(file_id), image_features.tobytes())


file_names = listdir(IMAGE_PATH)
print(f"images in {IMAGE_PATH} = {len(file_names)}")
new_images = []

for file_name in tqdm(file_names):
    file_id = int(file_name[:file_name.index('.')])
    if check_if_exists_by_id(file_id):
        continue
    new_images.append(file_name)

print(f"new images = {len(new_images)}")
new_images = [new_images[i:i + 100000] for i in range(0, len(new_images), 100000)]
for batch in new_images:
    features=[calc_nn_features(file_name) for file_name in tqdm(batch)]
    features = [i for i in features if i]  # remove None's
    print("pushing data to db")
    with DB.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(features)
