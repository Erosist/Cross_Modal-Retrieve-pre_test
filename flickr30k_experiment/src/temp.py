import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv


INPUT_JSONL = "flickr30k_captions.json"        
OUTPUT_JSON = "flickr30k_captions_masked.json"

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
question: who had the most governmental power under the articles of confederation
idx_gold_in_corpus: 21034054

EXAMPLE JSON OUTPUT:
{
  "masked_sentences": [
    "_ had the most governmental power under the articles of confederation",
    "who had _ _ _ _ under the articles of confederation",
    "who had the most governmental power _ _ _ _ _",
    "_ had the _ governmental power under the articles of confederation"
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

    raise None


def main(num_samples: int = 10):
    results = []

    data = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    for item in tqdm(data, desc="Processing"):
        img_id = item.get("img_id")
        sentid = item.get("sentid")
        caption = item.get("caption")

        masked_sentences = call_deepseek_mask(
        question=caption,
        idx=str(sentid),
        )

        if masked_sentences is None:
            print(f"跳过 sentid={sentid}，未生成 masked sentences。")
            continue

        out = {
            "img_id": img_id,
            "sentid": sentid,
            "caption": caption,
            "L1_mask": masked_sentences[0],
            "L2_mask": masked_sentences[1],
            "L3_mask": masked_sentences[2],
            "L4_mask": masked_sentences[3],
        }
        results.append(out)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"完成，结果已写入：{OUTPUT_JSON}。共处理 {len(results)} 条。")

if __name__ == "__main__":
    main(num_samples=1000)

