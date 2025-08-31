import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss
from datasets import load_from_disk

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/data/modelscope/models/openai-mirror/clip-vit-base-patch32/"
val2014_path = "/home/Masked_Reranker/dataset/MSCOCO/train2014/val2014/"
batch_size = 32

faiss_index_path = "val2014_faiss_test.index"
image_ids_path = "val2014_image_ids_test.pt"


test_ds_path = "/home/Masked_Reranker/dataset/yerevann/coco-karpathy/test"
ds = load_from_disk(test_ds_path)

test_images = {item['filename']: item['cocoid'] for item in ds}

print(f"找到 test split 图像数: {len(test_images)}") 

available_files = [f for f in os.listdir(val2014_path) if f in test_images]
print(f"实际存在于 val2014 的 test 图片数: {len(available_files)}")

model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path,use_fast=True)
model.eval()

all_embeds = []
image_ids = []

for i in tqdm(range(0, len(available_files), batch_size), desc="计算图片嵌入"):
    batch_files = available_files[i:i+batch_size]
    images = [Image.open(os.path.join(val2014_path, f)).convert("RGB") for f in batch_files]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
    all_embeds.append(embeds.cpu())
    image_ids.extend([test_images[f] for f in batch_files])

all_embeds = torch.cat(all_embeds, dim=0).numpy().astype('float32')
d = all_embeds.shape[1]

res = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.add(all_embeds)

faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_index_path)
torch.save({"image_ids": image_ids}, image_ids_path)

print(f"FAISS 索引保存到 {faiss_index_path}")
print(f"image_id 保存到 {image_ids_path}")
