import json

# 选择训练集的 caption 标注文件
ann_file = "/home/Masked_Reranker/dataset/MSCOCO/train2014/annotations/captions_val2014.json"

# 加载 JSON
with open(ann_file, "r") as f:
    data = json.load(f)

# 打印 keys
print("顶层 keys:", data.keys())

# 打印 images 部分前 2 个
print("\n前2个 images:")
for img in data["images"][:2]:
    print(img)

# 打印 annotations 部分前 5 个
print("\n前5个 annotations:")
for ann in data["annotations"][:5]:
    print(ann)
