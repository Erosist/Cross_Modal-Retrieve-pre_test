import json
import torch
import chromadb
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np

CHROMA_PERSIST_DIR = "./val2014_chroma_db"
MODEL_PATH = "/root/.cache/modelscope/hub/models/AI-ModelScope/clip-vit-large-patch14/"
INPUT_QUERIES_JSON = "masked.json"
OUTPUT_RESULTS_JSON = "result_3.json"

TOP_K_RECALL = 5000
TEXT_BATCH_SIZE = 5    
QUERY_BATCH_SIZE = 10    
MAX_LOAD = 20000  

print("正在加载模型和ChromaDB...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model.eval()

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection("val2014_test")
print(f"模型和ChromaDB加载完成。集合中共有 {collection.count()} 个向量。")

def get_text_embeddings_batch(texts: list, batch_size: int = 32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = processor.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            batch_embeds = model.get_text_features(**inputs)
            batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=-1)
        all_embeddings.append(batch_embeds.cpu().numpy())
    return np.vstack(all_embeddings).astype("float32")

def find_rank_in_candidates(query_embedding: np.ndarray, candidate_ids: list, candidate_embeds: dict, gold_id: str) -> int:
    scores = {}
    for item_id in candidate_ids:
        img_embed = candidate_embeds.get(item_id)
        if img_embed is not None:
            scores[item_id] = np.dot(query_embedding, img_embed)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return -1

def main():
    with open(INPUT_QUERIES_JSON, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    if MAX_LOAD is not None:
        queries_data = queries_data[:MAX_LOAD]

    all_final_results = []

    for batch_start in tqdm(range(0, len(queries_data), QUERY_BATCH_SIZE), desc="Processing Query Batches"):
        batch_queries = queries_data[batch_start:batch_start + QUERY_BATCH_SIZE]

        batch_texts = []
        for query_item in batch_queries:
            batch_texts.extend([
                query_item["caption"],      # rank_0
                query_item["L1_mask"],      # rank_1
                query_item["L2_mask"],      # rank_2
                query_item["L3_mask"],      # rank_3
                query_item["L4_mask"],      # rank_4
            ])
        
        batch_embeddings = get_text_embeddings_batch(batch_texts, batch_size=TEXT_BATCH_SIZE)

        for i, query_item in enumerate(batch_queries):
            gold_id = str(query_item["image_id"])
            embeddings = batch_embeddings[i*5:(i+1)*5]

            recall_results = collection.query(
                query_embeddings=[embeddings[0].tolist()],
                n_results=TOP_K_RECALL,
                include=["distances"]
            )
            recalled_ids = recall_results['ids'][0]

            if gold_id not in recalled_ids:
                recalled_ids.append(gold_id)

            candidate_results = collection.get(ids=recalled_ids, include=["embeddings"])
            candidate_embeds = {item_id: np.array(embed) for item_id, embed in zip(candidate_results['ids'], candidate_results['embeddings'])}

            rank_keys = ["rank_0", "rank_1", "rank_2", "rank_3", "rank_4"]
            ranks = {}
            for idx, rank_key in enumerate(rank_keys):
                ranks[rank_key] = find_rank_in_candidates(embeddings[idx], recalled_ids, candidate_embeds, gold_id)

            all_final_results.append({
                "caption": query_item["caption"],
                "gold_indice": int(gold_id),
                **ranks
            })

    with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as fout:
        json.dump(all_final_results, fout, ensure_ascii=False, indent=2)

    print(f"\n完成！所有查询的排名结果已保存到: {OUTPUT_RESULTS_JSON}")

if __name__ == "__main__":
    main()
