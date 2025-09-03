#!/usr/bin/env python3
import json
import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import chromadb

# ---------- 配置 ----------
CHROMA_PERSIST_DIR = "./val2014_chroma_db"
MODEL_PATH = "/root/.cache/modelscope/hub/models/AI-ModelScope/clip-vit-large-patch14/"
INPUT_QUERIES_JSON = "masked.json"   # 你的 queries 文件 (每条包含 caption, L1_mask..L4_mask, image_id 或 gold_indice)
OUTPUT_RESULTS_JSON = "result_exact_fullrank_fixed.json"

TEXT_EMBED_BATCH = 128   # 文本嵌入批量大小（可根据显存/内存调）
SCORE_BATCH = 256        # 计算 scores 时的批量大小（越大越快但内存占用多）

device = "cuda" if torch.cuda.is_available() else "cpu"

print("加载模型/processor ...")
model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model.eval()

print("打开 ChromaDB 并尝试读取 image embeddings（全库 exact） ...")
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection("val2014_test")

# 尝试一次性把库里所有 embeddings 和 ids 取出来
# 注意：include 中不要写 "ids" —— Chroma 不接受这个选项
try:
    all_items = collection.get(include=["embeddings", "metadatas"])
except Exception as e:
    # 某些环境下 "metadatas" 也可能引发不必要的限制，退回到只取 embeddings
    all_items = collection.get(include=["embeddings"])

# all_items 通常包含 keys: 'ids', 'embeddings', 'metadatas'（ids 虽不可作为 include 项，但会出现在返回值中）
all_ids = [str(i) for i in all_items.get("ids", [])]
all_embeds = np.array(all_items.get("embeddings", []), dtype=np.float32)  # shape (N_images, D)

N_images = all_embeds.shape[0]
print(f"从 Chroma 读取到 {N_images} 个 image embeddings（collection.count()={collection.count()})")

if N_images != collection.count():
    print("⚠️ 警告：从 collection.get() 读取到的 embeddings 数量 与 collection.count() 不一致。")
    print("这可能是 Chroma 的分页/限制导致未一次性返回全量。")
    print("建议：检查 Chroma 版本或将 embeddings 导出为本地 .npy 文件再读取；否则结果可能不全面。")

# 确保 embeddings 被 L2 归一化（若你写入时已经归一化，这里也做一遍以确保）
def l2_normalize_rows(x: np.ndarray):
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

if N_images > 0:
    all_embeds = l2_normalize_rows(all_embeds)

# 建立 id -> index 的映射（collection 存的是 string ids）
id_to_index = {iid: idx for idx, iid in enumerate(all_ids)}

# ---------- 读取 queries 并构造文本列表（5 个 per caption） ----------
print(f"读取 queries: {INPUT_QUERIES_JSON}")
with open(INPUT_QUERIES_JSON, "r", encoding="utf-8") as f:
    queries = json.load(f)

if len(queries) == 0:
    raise ValueError("queries 文件为空，请检查 INPUT_QUERIES_JSON 路径与内容")

# 支持不同 key 名：优先 gold_indice，其次 image_id
gold_key = "gold_indice" if "gold_indice" in queries[0] else ("image_id" if "image_id" in queries[0] else None)
if gold_key is None:
    raise ValueError("无法在 queries 中找到 'gold_indice' 或 'image_id' 字段。")

# 构造 texts_flat, gold_ids_flat, and mapping index->(query_idx, mask_idx)
texts_flat = []
gold_ids_flat = []
meta_map = []  # list of tuples (query_idx, mask_idx) where mask_idx: 0->orig caption, 1..4->mask1..mask4
for qi, q in enumerate(queries):
    gold_id = str(q[gold_key])
    caption = q.get("caption", "")
    masked1 = q.get("L1_mask", "")
    masked2 = q.get("L2_mask", "")
    masked3 = q.get("L3_mask", "")
    masked4 = q.get("L4_mask", "")

    variants = [caption, masked1, masked2, masked3, masked4]
    for mi, txt in enumerate(variants):
        texts_flat.append(txt)
        gold_ids_flat.append(gold_id)
        meta_map.append((qi, mi))  # later use to assemble per-query results

T = len(texts_flat)
print(f"总共文本片段 (should be num_queries * 5) = {T}")

# ---------- 计算所有文本嵌入（分批） ----------
def get_text_embeddings(texts_list, batch_size=TEXT_EMBED_BATCH):
    all_embs = []
    for i in tqdm(range(0, len(texts_list), batch_size), desc="计算 text embeddings"):
        batch = texts_list[i:i+batch_size]
        inputs = processor.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            out = torch.nn.functional.normalize(out, dim=-1)
            all_embs.append(out.cpu().numpy())
    if len(all_embs) == 0:
        return np.zeros((0, model.config.projection_dim if hasattr(model.config, "projection_dim") else out.shape[-1]), dtype=np.float32)
    return np.vstack(all_embs).astype(np.float32)

text_embeds = get_text_embeddings(texts_flat, batch_size=TEXT_EMBED_BATCH)
text_embeds = l2_normalize_rows(text_embeds)

# ---------- 对文本分批计算与所有 images 的精确 dot-product 排名 ----------
print("开始对每个文本与所有 images 做精确相似度并计算 rank（使用向量化批处理）...")
ranks_flat = np.full((T,), -1, dtype=np.int32)  # 初始化为 -1 表示未命中 / gold 不在库内

if N_images == 0:
    raise RuntimeError("图库中没有 embeddings（N_images == 0），无法计算检索，请检查 ChromaDB 内容或重建 embeddings 。")

for bstart in tqdm(range(0, T, SCORE_BATCH), desc="计算 ranks batches"):
    bend = min(T, bstart + SCORE_BATCH)
    batch_emb = text_embeds[bstart:bend]          # shape (B, D)
    B = batch_emb.shape[0]

    # score matrix: (B, N_images)
    scores = batch_emb.dot(all_embeds.T)  # float32

    # gold indices 对应每行
    gold_ids_batch = gold_ids_flat[bstart:bend]
    gold_indices = np.array([id_to_index.get(gid, -1) for gid in gold_ids_batch], dtype=np.int32)

    valid_mask = gold_indices >= 0
    if valid_mask.any():
        # gold_scores：score of gold for each valid row
        gold_scores = np.empty((B,), dtype=np.float32)
        gold_scores.fill(np.nan)
        rows = np.where(valid_mask)[0]
        gold_scores[rows] = scores[rows, gold_indices[rows]]

        # rank = 1 + number of images with score > gold_score
        comp = scores[valid_mask] > gold_scores[valid_mask, None]  # shape (num_valid, N_images)
        ranks_batch = comp.sum(axis=1).astype(np.int32) + 1  # 1-based rank
        ranks_flat[bstart:bend][valid_mask] = ranks_batch

    # those with gold_index == -1 remain -1

# ---------- 将 flat ranks 重新组合为 per-query（每个 query 包含 5 ranks）并保存 ----------
print("组装每个 query 的 rank 结果并写入 JSON ...")
all_final_results = []
for qi, q in enumerate(queries):
    base_idx = qi * 5
    r0 = int(ranks_flat[base_idx + 0])   # original caption
    r1 = int(ranks_flat[base_idx + 1])
    r2 = int(ranks_flat[base_idx + 2])
    r3 = int(ranks_flat[base_idx + 3])
    r4 = int(ranks_flat[base_idx + 4])

    gold_id = str(q[gold_key])
    all_final_results.append({
        "caption": q.get("caption", ""),
        "gold_indice": int(gold_id),
        "rank_0": r0,
        "rank_1": r1,
        "rank_2": r2,
        "rank_3": r3,
        "rank_4": r4
    })

with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as fout:
    json.dump(all_final_results, fout, ensure_ascii=False, indent=2)

print(f"完成，所有排名已保存到: {OUTPUT_RESULTS_JSON}")
print("提示：现在你可以用 caption-level 的统计脚本（分母为 caption 总数）来计算 Recall@K，与 CLIP 论文对齐。")
