import json
import csv
import os

def analyze(json_list):
    total_captions = len(json_list)  # 所有 caption 都要统计
    results = {
        "recall_rank0": {},
        "recall_mask": {}
    }

    # ----------- rank_0 的统计（Caption-level，官方策略） -----------
    for k in [1, 5, 10, 20]:
        hits = sum(
            1 for item in json_list
            if item["rank_0"] != -1 and item["rank_0"] <= k
        )
        results["recall_rank0"][f"recall@{k}"] = (
            f"{(hits / total_captions * 100):.2f}%"
        )

    # ----------- mask (rank_1 到 rank_4) 的统计（Caption-level） -----------
    caption_best_mask = []
    for item in json_list:
        ranks = [item[f"rank_{i}"] for i in range(1, 5) if item[f"rank_{i}"] != -1]
        best_rank = min(ranks) if ranks else None
        caption_best_mask.append(best_rank)

    for k in [1, 5, 10, 20]:
        hits = sum(
            1 for r in caption_best_mask
            if r is not None and r <= k
        )
        results["recall_mask"][f"recall@{k}"] = (
            f"{(hits / total_captions * 100):.2f}%"
        )

    return results


if __name__ == "__main__":
    input_files = [
        "/root/Cross_Modal-Retrieve-pre_test/MSCOCO_experiment/src/result_exact_fullrank_fixed.json",
    ]

    summary = []

    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        with open(input_file, "r", encoding="utf-8") as f:
            json_list = json.load(f)

        results = analyze(json_list)

        print(f"处理完成: {input_file}")

        summary.append({
            "file": base_name,
            "rank0_recall@1": results["recall_rank0"]["recall@1"],
            "rank0_recall@5": results["recall_rank0"]["recall@5"],
            "rank0_recall@10": results["recall_rank0"]["recall@10"],
            "rank0_recall@20": results["recall_rank0"]["recall@20"],
            "mask_recall@1": results["recall_mask"]["recall@1"],
            "mask_recall@5": results["recall_mask"]["recall@5"],
            "mask_recall@10": results["recall_mask"]["recall@10"],
            "mask_recall@20": results["recall_mask"]["recall@20"],
        })

    summary_file = "standard_3.csv"
    with open(summary_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "file",
            "rank0_recall@1", "rank0_recall@5", "rank0_recall@10", "rank0_recall@20",
            "mask_recall@1", "mask_recall@5", "mask_recall@10", "mask_recall@20"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n所有文件的主结果汇总已保存到 {summary_file}")
