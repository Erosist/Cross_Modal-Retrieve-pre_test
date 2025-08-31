import json
import csv
import os

def analyze(json_list):
    total = len(json_list)
    valid = sum(1 for item in json_list if item["rank_0"] != -1)
    ratio_valid = (valid / total * 100) if total > 0 else 0

    results = {
        "ratio_valid_rank0": f"{ratio_valid:.2f}%",
        "recall": {}
    }

    changes = []  

    for k in [5, 10, 20]:
        count = 0   
        total_k = 0 

        for item in json_list:
            r0 = item["rank_0"]
            if r0 != -1 and r0 > k:
                total_k += 1
                hit = False
                for i in range(1, 5):
                    ri = item[f"rank_{i}"]
                    if ri <= k and ri < r0:  
                        hit = True
                        mask_key = f"L{i}_mask"
                        changes.append({
                            "caption": item["caption"],
                            "rank_0": r0,
                            "better_rank": ri,
                            "k": k,
                            "mask_used": item.get(mask_key, "")
                        })
                if hit:
                    count += 1

        ratio = (count / total_k * 100) if total_k > 0 else 0
        results["recall"][f"recall@{k}"] = f"{ratio:.2f}%"

    return results, changes


if __name__ == "__main__":
    input_files = [
        "/home/Masked_Reranker/flickr30k_experiment/prompt_2/prompt_2_k20.json",
        "/home/Masked_Reranker/flickr30k_experiment/prompt_2/prompt_2_k30.json",
        "/home/Masked_Reranker/flickr30k_experiment/prompt_2/prompt_2_k40.json",
        "/home/Masked_Reranker/flickr30k_experiment/prompt_2/prompt_2_k50.json",
    ]

    summary = [] 

    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        with open(input_file, "r", encoding="utf-8") as f:
            json_list = json.load(f)

        results, changes = analyze(json_list)

        changes_file_json = f"{base_name}_changes.json"
        with open(changes_file_json, "w", encoding="utf-8") as f:
            json.dump(changes, f, indent=2, ensure_ascii=False)

        changes_file_csv = f"{base_name}_changes.csv"
        with open(changes_file_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["caption", "rank_0", "better_rank", "k", "mask_used"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(changes)

        print(f"处理完成: {input_file}")
        print(f"  变化JSON -> {changes_file_json}")
        print(f"  变化CSV -> {changes_file_csv}")

        summary.append({
            "file": base_name,
            "ratio_valid_rank0": results["ratio_valid_rank0"],
            "recall@5": results["recall"]["recall@5"],
            "recall@10": results["recall"]["recall@10"],
            "recall@20": results["recall"]["recall@20"],
        })

    summary_file = "all_results_summary.csv"
    with open(summary_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["file", "ratio_valid_rank0", "recall@5", "recall@10", "recall@20"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n所有文件的主结果汇总已保存到 {summary_file}")
