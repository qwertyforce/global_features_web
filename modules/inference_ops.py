import torch
import timm
import numpy as np 

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_device():
    return device

print(f"using {device}")

model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)
model.head=torch.nn.Identity()
# model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
model.eval()
model.to(device)

def get_image_features(images):
    with torch.no_grad():
        feature_vector = model(images)
        feature_vector/=torch.linalg.norm(feature_vector,axis=1).reshape(-1,1)
    return feature_vector.cpu().numpy().astype(np.float32)