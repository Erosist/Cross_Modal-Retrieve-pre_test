import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "./dataset/flickr30k_dataset" 
batch_size = 64
save_path = "flickr30k_img_embeds.pt"

dataset = load_from_disk(dataset_path)["test"] 
print(f"数据集条数: {len(dataset)}")

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval() 

class FlickrImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = preprocess(item["image"])  # CLIP预处理
        img_id = item["img_id"]
        return img, img_id

dataloader = DataLoader(FlickrImageDataset(dataset), batch_size=batch_size, shuffle=False)

all_embeds = []
all_ids = []

with torch.no_grad():
    for imgs, img_ids in tqdm(dataloader, desc="Encoding images"):
        imgs = imgs.to(device)
        emb = model.encode_image(imgs)          
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)  
        all_embeds.append(emb.cpu())
        all_ids.extend(img_ids)

all_embeds = torch.cat(all_embeds, dim=0) 
print(f"嵌入生成完成，shape: {all_embeds.shape}")

torch.save({
    "embeddings": all_embeds,
    "img_ids": all_ids
}, save_path)

print(f"保存完成: {save_path}")
