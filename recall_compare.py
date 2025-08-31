import json
import csv
import os

def analyze(json_list):
    total = len(json_list)
    valid_rank0 = sum(1 for item in json_list if item["rank_0"] != -1)
    ratio_valid = (valid_rank0 / total * 100) if total > 0 else 0

    results = {
        "ratio_valid_rank0": f"{ratio_valid:.2f}%",
        "recall_rank0": {},
        "recall_mask": {}
    }

    for k in [1, 5, 10, 20]:
        hit_r0 = sum(1 for item in json_list if item["rank_0"] != -1 and item["rank_0"] <= k)
        results["recall_rank0"][f"recall@{k}"] = f"{(hit_r0 / valid_rank0 * 100):.2f}%" if valid_rank0 > 0 else "0.00%"

    valid_mask = 0
    for item in json_list:
        ranks = [item[f"rank_{i}"] for i in range(1, 5) if item[f"rank_{i}"] != -1]
        if ranks:  
            valid_mask += 1

    for k in [1, 5, 10, 20]:
        hit_mask = 0   
        for item in json_list:
            ranks = [item[f"rank_{i}"] for i in range(1, 5) if item[f"rank_{i}"] != -1]
            if ranks:
                best_rank = min(ranks)
                if best_rank <= k:
                    hit_mask += 1
        results["recall_mask"][f"recall@{k}"] = f"{(hit_mask / valid_mask * 100):.2f}%" if valid_mask > 0 else "0.00%"

    return results


if __name__ == "__main__":
    input_files = [
        "/home/Masked_Reranker/MSCOCO_experiment/src/random_result.json",
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
            "ratio_valid_rank0": results["ratio_valid_rank0"],
            "rank0_recall@1": results["recall_rank0"]["recall@1"],  
            "rank0_recall@5": results["recall_rank0"]["recall@5"],
            "rank0_recall@10": results["recall_rank0"]["recall@10"],
            "rank0_recall@20": results["recall_rank0"]["recall@20"],
            "mask_recall@1": results["recall_mask"]["recall@1"],  
            "mask_recall@5": results["recall_mask"]["recall@5"],
            "mask_recall@10": results["recall_mask"]["recall@10"],
            "mask_recall@20": results["recall_mask"]["recall@20"],
        })

    summary_file = "standard_1.csv"
    with open(summary_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "file", "ratio_valid_rank0",
            "rank0_recall@1", "rank0_recall@5", "rank0_recall@10", "rank0_recall@20", 
            "mask_recall@1", "mask_recall@5", "mask_recall@10", "mask_recall@20"    
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n所有文件的主结果汇总已保存到 {summary_file}")
