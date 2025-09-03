import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_from_disk
import numpy as np
import chromadb

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/root/.cache/modelscope/hub/models/AI-ModelScope/clip-vit-large-patch14/"
val2014_path = "/root/Cross_Modal-Retrieve-pre_test/dataset/MSCOCO/val2014/"
batch_size = 20
chroma_persist_dir = "./val2014_chroma_db"
test_ds_path = "/root/Cross_Modal-Retrieve-pre_test/dataset/yerevann/coco-karpathy/test/"

ds = load_from_disk(test_ds_path)
test_images = {item['filename']: item['cocoid'] for item in ds}
print(f"找到 test split 图像数: {len(test_images)}")

available_files = [f for f in os.listdir(val2014_path) if f in test_images]
print(f"实际存在于 val2014 的 test 图片数: {len(available_files)}")

model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
model.eval()


client = chromadb.PersistentClient(path=chroma_persist_dir)


collection = client.get_or_create_collection("val2014_test")

for i in tqdm(range(0, len(available_files), batch_size), desc="计算图片嵌入并写入 Chroma"):
    batch_files = available_files[i:i+batch_size]
    images = [Image.open(os.path.join(val2014_path, f)).convert("RGB") for f in batch_files]
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)

    embeds_np = embeds.cpu().numpy().astype("float32")
    ids = [str(test_images[f]) for f in batch_files]
    metadatas = [{"filename": f} for f in batch_files]

    collection.add(
        embeddings=embeds_np,
        metadatas=metadatas,
        ids=ids
    )


print(f"Chroma 数据库已自动持久化到 {chroma_persist_dir}")

query_vector = embeds_np[0].tolist() 
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
print("查询结果示例：", results)