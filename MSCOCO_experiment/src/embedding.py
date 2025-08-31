import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/data/modelscope/models/openai-mirror/clip-vit-base-patch32/"
val2014_path = "/home/Masked_Reranker/dataset/MSCOCO/train2014/val2014/"
batch_size = 32

faiss_index_path = "val2014_faiss.index"
image_ids_path = "val2014_image_ids.pt"

model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path,use_fast=True)
model.eval()

image_files = [f for f in os.listdir(val2014_path) if f.endswith(".jpg")]
print(f"一共有 {len(image_files)} 张图片")  # 新增统计行
def extract_image_id(filename):
    return int(filename.split('_')[-1].split('.')[0])

image_ids = [extract_image_id(f) for f in image_files]

all_embeds = []

for i in tqdm(range(0, len(image_files), batch_size), desc="计算图片嵌入"):
    batch_files = image_files[i:i+batch_size]
    images = [Image.open(os.path.join(val2014_path, f)).convert("RGB") for f in batch_files]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
    all_embeds.append(embeds.cpu())

all_embeds = torch.cat(all_embeds, dim=0) 
all_embeds = all_embeds.numpy().astype('float32') 

d = all_embeds.shape[1]

res = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(all_embeds)

faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_index_path)
torch.save({"image_ids": image_ids}, image_ids_path)

print(f"FAISS 索引保存到 {faiss_index_path}")
print(f"image_id 保存到 {image_ids_path}")

