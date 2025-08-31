import json
import random
import torch
import faiss
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import torch.nn.functional as F

device = "cuda"
data_path = "random_mask.json"
out_path = "random_result.json"
model_path = "/data/modelscope/models/openai-mirror/clip-vit-base-patch32/"

tokenizer = CLIPTokenizer.from_pretrained(model_path)
model = CLIPModel.from_pretrained(model_path).to(device)

index = faiss.read_index("val2014_faiss.index")

img_data = torch.load("val2014_image_ids.pt")
img_ids = img_data["image_ids"]

id2idx = {img_id: idx for idx, img_id in enumerate(img_ids)}

top_k = 40504
batch_size = 50


def load_json(json_path, max_queries=5):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    if max_queries:
        data = data[:max_queries]
    caption = [item["caption"] for item in data]
    golden_id = [item["image_id"] for item in data] 
    L1_mask = [item["L1_mask"] for item in data]
    L2_mask = [item["L2_mask"] for item in data]
    L3_mask = [item["L3_mask"] for item in data]
    L4_mask = [item["L4_mask"] for item in data]
    return caption, golden_id, L1_mask, L2_mask, L3_mask, L4_mask


def find_gold_rank(golden_id, top_k_list):
    try:
        rank = top_k_list.index(golden_id)
    except ValueError:
        rank = -1
    return rank


def encode_texts(texts, batch_size=batch_size):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
            embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds)
    return torch.cat(all_embeds, dim=0)  # [B, d]


def get_topk_batch_faiss(queries, index, top_k=40, batch_size=batch_size):
    text_embeds = encode_texts(queries, batch_size=batch_size)  # [B, d]
    text_embeds = text_embeds.cpu().numpy().astype("float32")
    D, I = index.search(text_embeds, top_k)  # faiss 检索
    results = [[img_ids[idx] for idx in row] for row in I]
    return results


def reorder_with_mask_batch(masked_queries, candidate_results, index, id2idx, img_ids, batch_size=batch_size):

    all_masked_embeds = encode_texts(masked_queries, batch_size=batch_size) 
    reordered_results = []

    for i, candidates in enumerate(candidate_results):
        candidate_indices = [id2idx[img_id] for img_id in candidates]

        try:
            candidate_embeds = index.reconstruct_batch(candidate_indices) 
        except AttributeError:
            import numpy as np
            candidate_embeds = np.stack([index.reconstruct(idx) for idx in candidate_indices], axis=0)

        candidate_embeds = torch.from_numpy(candidate_embeds).to(device)
        sim = F.cosine_similarity(all_masked_embeds[i:i + 1], candidate_embeds)
        sorted_indices = torch.argsort(sim, descending=True)
        reordered = [candidates[idx] for idx in sorted_indices.cpu()]
        reordered_results.append(reordered)

    return reordered_results


def main():
    caption, golden_id, L1_mask, L2_mask, L3_mask, L4_mask = load_json(data_path, max_queries=1000)

    final_results = []

    topk_results = get_topk_batch_faiss(caption, index, top_k=top_k, batch_size=batch_size)

    for i in tqdm(range(len(caption)), desc="查询中..."):
        gold_img_id = golden_id[i]

        masked_queries = [L1_mask[i], L2_mask[i], L3_mask[i], L4_mask[i]]
        reordered_results = reorder_with_mask_batch(
            masked_queries,
            [topk_results[i]] * 4,
            index,
            id2idx,
            img_ids,
            batch_size=batch_size,
        )

        ranks = [find_gold_rank(gold_img_id, topk_results[i])] + \
                [find_gold_rank(gold_img_id, r) for r in reordered_results]

        result_item = {
            "caption": caption[i],
            "gold_indice": gold_img_id,
            "rank_0": ranks[0],
            "rank_1": ranks[1],
            "rank_2": ranks[2],
            "rank_3": ranks[3],
            "rank_4": ranks[4]
        }

        for j in range(1, 5):
            if ranks[j] != -1 and (ranks[0] == -1 or ranks[j] < ranks[0]):
                result_item[f"L{j}_mask"] = masked_queries[j - 1]

        final_results.append(result_item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"完成，结果已写入 {out_path}")


if __name__ == "__main__":
    main()
