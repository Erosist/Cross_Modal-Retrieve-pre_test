from datasets import load_from_disk
import pandas as pd

dataset_path = "./dataset/flickr30k_dataset"  
dataset = load_from_disk(dataset_path)["test"]
print(f"数据集大小: {len(dataset)}")

records = []

for item in dataset:
    img_id = item["img_id"]
    captions = item["caption"]
    sentids = item["sentids"]
    for cap, sid in zip(captions, sentids):
        records.append({
            "img_id": img_id,  
            "sentid": sid,      
            "caption": cap      
        })

df = pd.DataFrame(records)

df.to_json("flickr30k_captions.json", orient="records", lines=True)

print("保存完成: flickr30k_captions.json")
print(df.head())
