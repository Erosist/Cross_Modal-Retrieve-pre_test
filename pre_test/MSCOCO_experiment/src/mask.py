import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed

ARROW_DATASET_PATH = "/root/Cross_Modal-Retrieve-pre_test/dataset/yerevann/coco-karpathy/test/"
OUTPUT_JSON = "masked.json" 
MAX_WORKERS = 10

load_dotenv()
API_KEY = os.getenv("api")
MODEL_NAME = "deepseek-chat"
MAX_TOKENS = 256

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
 )
SYSTEM_PROMPT = """
You are a helpful assistant that must output valid JSON only.

Task:
The user will provide a sentence and an index ID.
Split the sentence into 4 key segments (each segment contains crucial meaning).
For each segment, create a masked version of the original sentence by replacing that segment with underscores ("_").

Requirements:
- Output MUST be a valid JSON object (strict JSON, no extra text).
- Return exactly 4 masked sentences.
- Each of the 4 masked parts should be different (avoid repeating the same masked content).
- You may use syntactic structure (subject, predicate, object, adverbials, etc.) as a guide when choosing mask segments.
- Do not mask too much of the sentence at once. For example, avoid masking the entire main description (bad: "Four casually dressed _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ .").
- When the sentence is long or contains multiple pieces of key information, it is acceptable to mask in smaller consecutive chunks (e.g., 2–3 words at a time), so that different masks still preserve enough context.
- Make sure the 4 masked versions together cover different important parts of the sentence.

The JSON schema should be:
{
  "masked_sentences": [
    "masked sentence 1",
    "masked sentence 2",
    "masked sentence 3",
    "masked sentence 4"
  ]
}

EXAMPLE INPUT:
question: A trendy girl talking on her cellphone while gliding slowly down the street
idx_gold_in_corpus: 21034054

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ _ _ talking on her cellphone while gliding slowly down the street",
    "A trendy girl talking on her cellphone while gliding slowly _ _ _",
    "A trendy girl _ _ _ _ while gliding slowly down the street",
    "A trendy girl talking on her cellphone while _ _ slowly down the street"
  ]
}

EXAMPLE INPUT:
question: Two young guys with shaggy hair look at their hands while hanging out in the yard
idx_gold_in_corpus: 21035002

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ _ _ with shaggy hair look at their hands while hanging out in the yard",
    "Two young guys with _ _ look at their hands while hanging out in the yard",
    "Two young guys with shaggy hair _ _ _ _ while hanging out in the yard",
    "Two young guys with shaggy hair look at their hands while _ _ _ _ _"
  ]
}

EXAMPLE INPUT:
question: what is the job of the whip in congress
idx_gold_in_corpus: 21033795

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "what is the _ of the whip in congress",
    "what is the job of the _ in congress",
    "what is the job of the whip _ _",
    "_ _ the job of the whip in congress"
  ]
}
""".strip()

SYSTEM_PROMPT_2 = """
You are a helpful assistant that must output valid JSON only.

Task:
The user will provide a sentence and an index ID.
Split the sentence into 4 key segments (each segment contains crucial meaning).
For each segment, create a masked version of the original sentence by replacing that segment with underscores ("_").

Requirements:
- Output MUST be a valid JSON object (strict JSON, no extra text).
- Return exactly 4 masked sentences.
- The JSON schema should be:
{
  "masked_sentences": [
    "masked sentence 1",
    "masked sentence 2",
    "masked sentence 3",
    "masked sentence 4"
  ]
}

EXAMPLE INPUT:
question: A trendy girl talking on her cellphone while gliding slowly down the street
idx_gold_in_corpus: 21034054

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ _ _ talking on her cellphone while gliding slowly down the street",
    "A trendy girl talking on her cellphone while gliding slowly _ _ _",
    "A trendy girl _ _ _ _ while gliding slowly down the street",
    "A trendy girl talking on her cellphone while _ _ slowly down the street"
  ]
}

EXAMPLE INPUT:
question: Two young guys with shaggy hair look at their hands while hanging out in the yard
idx_gold_in_corpus: 21035002

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ _ _ with shaggy hair look at their hands while hanging out in the yard",
    "Two young guys with _ _ look at their hands while hanging out in the yard",
    "Two young guys with shaggy hair _ _ _ _ while hanging out in the yard",
    "Two young guys with shaggy hair look at their hands while _ _ _ _ _"
  ]
}

EXAMPLE INPUT:
question: what is the job of the whip in congress
idx_gold_in_corpus: 21033795

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "what is the _ of the whip in congress",
    "what is the job of the _ in congress",
    "what is the job of the whip _ _",
    "_ _ the job of the whip in congress"
  ]
}
""".strip()

SYSTEM_PROMPT_RANDOM_MASK_STRONG = """
You are a helpful assistant that must output valid JSON only.

Task:
The user will provide a sentence and an index ID.
Randomly mask 4 different segments of the sentence by replacing consecutive words with underscores ("_").
- Each masked segment should start at a random position in the sentence.
- Each masked segment should have a random length (at least 1 word, up to half of the sentence).
- Each masked sentence should have its own independent random mask positions; do not follow a fixed pattern.
- The 4 masked sentences should be different from each other and cover different parts of the sentence.
- Masks can appear anywhere in the sentence, even multiple times if needed.
- The goal is to maximize randomness while still keeping the sentence understandable.

Requirements:
- Output MUST be a valid JSON object (strict JSON, no extra text).
- Return exactly 4 masked sentences.
- The JSON schema is:
{
  "masked_sentences": [
    "masked sentence 1",
    "masked sentence 2",
    "masked sentence 3",
    "masked sentence 4"
  ]
}

EXAMPLE INPUT:
question: A trendy girl talking on her cellphone while gliding slowly down the street
idx_gold_in_corpus: 21034054

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "A _ _ _ _ her cellphone while _ slowly down the street",
    "_ trendy girl _ _ her _ while gliding _ down the street",
    "A trendy _ _ on her _ _ _ slowly _",
    "A _ girl _ on _ cellphone while _ _ _ _ street"
  ]
}

EXAMPLE INPUT:
question: Two young guys with shaggy hair look at their hands while hanging out in the yard
idx_gold_in_corpus: 21035002

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ young guys _ shaggy _ _ _ their hands while hanging out in _ yard",
    "Two _ _ with shaggy _ look at their _ while hanging _ in the yard",
    "_ young _ with _ hair _ _ _ hands _ hanging out in the yard",
    "Two young guys _ shaggy hair _ at their _ while _ _ _ _ _"
  ]
}

EXAMPLE INPUT:
question: what is the job of the whip in congress
idx_gold_in_corpus: 21033795

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "what _ the job _ whip _ congress",
    "_ is the _ of the whip in _",
    "what _ _ job of the _ in congress",
    "_ is _ job _ the _ _ _"
  ]
}
""".strip()

def build_user_prompt(question: str, idx: str) -> str:
    return f"""question: {question}
idx_gold_in_corpus: {idx}"""

def call_deepseek_mask(question: str, idx: str, retries: int = 3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, idx)},
    ]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={'type': 'json_object'},
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                last_err = ValueError("Empty content from API")
                time.sleep(0.7 * attempt)
                continue
            data = json.loads(content)
            masked = data.get("masked_sentences", [])
            if not isinstance(masked, list) or len(masked) != 4:
                last_err = ValueError(f"Bad JSON shape: {data}")
                time.sleep(0.7 * attempt)
                continue
            masked = [m.strip() for m in masked]
            return masked
        except Exception as e:
            last_err = e
            time.sleep(0.7 * attempt)
    return None

def process_single_caption(caption: str, sentid: str, img_id: int):
    masked_sentences = call_deepseek_mask(
        question=caption,
        idx=sentid,
    )

    if masked_sentences:
        return {
            "image_id": img_id,
            "sentid": sentid,
            "caption": caption,
            "L1_mask": masked_sentences[0],
            "L2_mask": masked_sentences[1],
            "L3_mask": masked_sentences[2],
            "L4_mask": masked_sentences[3],
        }
    else:
        print(f"跳过 sentid={sentid}，未成功生成 masked sentences。")
        return None


def main(num_samples: int = 1000):
    print(f"正在从 {ARROW_DATASET_PATH} 加载数据集...")
    dataset = load_from_disk(ARROW_DATASET_PATH)
    print(f"数据集加载完成，共 {len(dataset)} 条记录。")

    samples_to_process = dataset.select(range(min(num_samples, len(dataset))))
    
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        print("正在提交所有 caption 任务到线程池...")
        for item in samples_to_process:
            img_id = item["cocoid"]
            sentences = item["sentences"]
            sentids = item["sentids"]

            for i, caption in enumerate(sentences):
                if i < len(sentids):
                    sentid = str(sentids[i])
                else:
                    sentid = f"{img_id}_sent_{i}"
                
                future = executor.submit(process_single_caption, caption, sentid, img_id)
                futures.append(future)
        
        print("任务提交完成，开始处理...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Captions"):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: (x['image_id'], x['sentid']))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"完成，结果已写入：{OUTPUT_JSON}。共处理 {len(results)} 条 caption。")


if __name__ == "__main__":
    main(num_samples=5000)
