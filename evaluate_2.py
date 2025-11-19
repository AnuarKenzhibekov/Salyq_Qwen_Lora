#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Оценка нескольких Qwen-совместимых embedding-моделей на eval.jsonl формата:
{"query": "...", "pos": ["позитивный"], "neg": ["негативный1", "негативный2", ...]}
или
{"query": "...", "pos": "позитивный", "neg": ["негативный1", ...]}

Метрика: MRR и Recall@k по рангу ПЕРВОГО pos в списке corpus = [pos, *neg].
"""

import json
import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


# ======== CONFIG ========

HF_MODEL_PATHS = {
    # имя модели в отчёте : путь/ID модели
    "Fine-Tuned (Qwen3-SalyqAI)": "./qwen3_lora_salyqai",          # твой output_dir
    "Base (Qwen3-Embedding-8B)": "Qwen/Qwen3-Embedding-8B",        # базовая
    # можешь добавить ещё любые HF-модели
}

EVAL_FILE = "eval.jsonl"
BATCH_SIZE = 16            # для батчинга внутри одной модели
MAX_QUERY_LEN = 128
MAX_PASSAGE_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ======== DATA LOADING ========

def load_eval_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data


# ======== POOLING ========

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Как в Qwen3-Embedding: берём hidden state последнего НЕ-PAD токена.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# ======== MODEL WRAPPER ========

def build_tokenizer_and_model(model_path: str):
    print(f"\n📦 Загружаем модель: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if DEVICE == "cuda":
        # для GPU можно bfloat16 / float16
        try:
            dtype = torch.bfloat16
        except TypeError:
            dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        trust_remote_code=True,
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def encode_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int,
) -> Tensor:
    """
    Кодирует список текстов в батче, возвращает тензор [N, D] (на CPU).
    """
    all_embs = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = model(**encoded)
        emb = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # [N, D]


# ======== METRICS (pos + neg) ========

def evaluate_qwen_model(model_name: str, model_path: str, data: List[Dict[str, Any]]):
    print("\n" + "=" * 60)
    print(f"📊 Evaluating HF model: {model_name} ({model_path})")

    tokenizer, model = build_tokenizer_and_model(model_path)

    recalls = {k: 0 for k in [1, 5, 10]}
    mrr_sum = 0.0
    total_samples = 0

    for item in tqdm(data, desc=f"Evaluating {model_name}"):
        query = str(item.get("query", "")).strip()
        pos = item.get("pos", "")
        neg = item.get("neg", [])

        # --- нормализуем формат POS ---
        if isinstance(pos, list):
            if len(pos) == 0:
                continue
            pos_text = str(pos[0]).strip()
        else:
            pos_text = str(pos).strip()

        # --- нормализуем формат NEG ---
        if isinstance(neg, list):
            neg_texts = [str(x).strip() for x in neg if str(x).strip()]
        else:
            n = str(neg).strip()
            neg_texts = [n] if n else []

        if not query or not pos_text or len(neg_texts) == 0:
            # для этой метрики нужен хотя бы 1 негатив
            continue

        # corpus = [POS, NEG1, NEG2, ...]
        corpus = [pos_text] + neg_texts

        # --- эмбеддинги ---
        query_emb = encode_batch([query], tokenizer, model, max_length=MAX_QUERY_LEN)[0:1]  # [1, D]
        corpus_embs = encode_batch(corpus, tokenizer, model, max_length=MAX_PASSAGE_LEN)    # [C, D]

        # cos_sim: [C]
        cos_scores = F.cosine_similarity(query_emb, corpus_embs)  # broadcasting: [1,D] vs [C,D] -> [C]

        # ранжируем (по убыванию)
        topk = torch.topk(cos_scores, k=len(corpus))
        # ищем позицию 0-го (первого) элемента = POS
        rank = (topk.indices == 0).nonzero(as_tuple=True)[0].item() + 1  # rank ∈ [1..C]

        mrr_sum += 1.0 / rank
        for k in recalls:
            if rank <= k:
                recalls[k] += 1

        total_samples += 1

    if total_samples == 0:
        print("⚠️  Нет валидных примеров для оценки (нужны pos + хотя бы один neg).")
        return None, None

    mrr = mrr_sum / total_samples
    final_recalls = {k: v / total_samples for k, v in recalls.items()}
    return mrr, final_recalls


def print_results(model_name: str, mrr: float, recalls: Dict[int, float]):
    print(f"\n--- Results for {model_name} ---")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  Accuracy / Recall@1:          {recalls[1]:.4f}")
    print(f"  Recall@5:                     {recalls[5]:.4f}")
    print(f"  Recall@10:                    {recalls[10]:.4f}")
    print("=" * 60)


# ======== MAIN ========

def main():
    if not os.path.exists(EVAL_FILE):
        print(f"❌ Error: Evaluation file '{EVAL_FILE}' not found!")
        return

    data = load_eval_data(EVAL_FILE)
    print(f"Loaded {len(data)} samples from {EVAL_FILE} for evaluation.")

    for model_name, model_path in HF_MODEL_PATHS.items():
        mrr, recalls = evaluate_qwen_model(model_name, model_path, data)
        if mrr is not None:
            print_results(model_name, mrr, recalls)


if __name__ == "__main__":
    main()
