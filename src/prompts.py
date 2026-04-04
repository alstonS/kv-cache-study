from transformers import AutoTokenizer

BASE_TEXT = (
    "Large language models use key value caches to speed up autoregressive decoding. "
    "This project studies memory efficiency, latency, and throughput under long contexts. "
)

def build_prompt_to_length(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    text = BASE_TEXT
    while True:
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]
        if ids.shape[0] >= target_tokens:
            trimmed_ids = ids[:target_tokens]
            return tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        text += " " + BASE_TEXT
