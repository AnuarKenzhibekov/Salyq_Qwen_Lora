import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, TaskType

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =ф===================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
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


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Пуллинг, как в примерах Qwen3-Embedding:
    берём hidden state последнего НЕ-PAD токена.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def build_hf_dataset(jsonl_path: str, add_instructions: bool = True) -> Dataset:
    """
    Ожидаемый формат строк:
        {"query": "...", "pos": ["..."], "neg": ["..."]}
    или
        {"query": "...", "pos": "...", "neg": "..."}
    """
    raw = load_jsonl(jsonl_path)

    queries = []
    pos_passages = []
    neg_passages = []

    for item in raw:
        q = str(item.get("query", "")).strip()
        pos = item.get("pos", "")
        neg = item.get("neg", "")

        # POS
        if isinstance(pos, list):
            if len(pos) == 0:
                continue
            p = str(pos[0]).strip()
        else:
            p = str(pos).strip()

        # NEG
        if isinstance(neg, list):
            if len(neg) == 0:
                continue
            n = str(neg[0]).strip()
        else:
            n = str(neg).strip()

        if not q or not p or not n:
            continue

        if add_instructions:
            q = "Instruct: " + q

        queries.append(q)
        pos_passages.append(p)
        neg_passages.append(n)

    print(f"[DATA] Загружено {len(queries)} примеров из {jsonl_path}")
    return Dataset.from_dict({
        "query": queries,
        "pos": pos_passages,
        "neg": neg_passages,
    })


# ==================== DATA COLLATOR ====================

@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_query_len: int = 128
    max_passage_len: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        queries = [f["query"] for f in features]
        pos_passages = [f["pos"] for f in features]
        neg_passages = [f["neg"] for f in features]

        q_batch = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt",
        )
        pos_batch = self.tokenizer(
            pos_passages,
            padding=True,
            truncation=True,
            max_length=self.max_passage_len,
            return_tensors="pt",
        )
        neg_batch = self.tokenizer(
            neg_passages,
            padding=True,
            truncation=True,
            max_length=self.max_passage_len,
            return_tensors="pt",
        )

        batch = {
            "query_input_ids": q_batch["input_ids"],
            "query_attention_mask": q_batch["attention_mask"],

            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],

            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }
        return batch


# ==================== КОНТРАСТИВНАЯ МОДЕЛЬ ====================

class QwenEmbeddingContrastiveModel(nn.Module):
    def __init__(self, base_model: AutoModel, temperature: float = 0.02):
        super().__init__()
        self.model = base_model
        self.temperature = temperature

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(outputs.last_hidden_state, attention_mask)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(
        self,
        query_input_ids: Tensor,
        query_attention_mask: Tensor,
        pos_input_ids: Tensor,
        pos_attention_mask: Tensor,
        neg_input_ids: Tensor,
        neg_attention_mask: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Улучшенный лосс:
        - используем и in-batch negatives, и explicit negatives
        - каждый query сравнивается со всеми pos и всеми neg из батча
        """
        # [B, D]
        q_emb = self.encode(query_input_ids, query_attention_mask)
        pos_emb = self.encode(pos_input_ids, pos_attention_mask)
        neg_emb = self.encode(neg_input_ids, neg_attention_mask)

        B = q_emb.size(0)

        # Собираем все кандидаты-документы:
        # сначала все pos, потом все neg → [2B, D]
        all_passages = torch.cat([pos_emb, neg_emb], dim=0)  # [2B, D]

        # Similarity: [B, D] @ [D, 2B] → [B, 2B]
        logits = (q_emb @ all_passages.t()) / self.temperature

        # Правильный ответ для i-го запроса — i-й pos в первой половине all_passages
        labels = torch.arange(B, device=q_emb.device)

        loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "query_emb": q_emb,
            # Для retrieval-метрик считаем, что "правильный" документ = pos
            "passage_emb": pos_emb,
        }


# ==================== МЕТРИКИ ДЛЯ RETRIEVAL ====================

def compute_retrieval_metrics(query_embs: Tensor, passage_embs: Tensor) -> Dict[str, float]:
    """
    Простая in-batch оценка:
    считаем, что i-й query должен найти i-й passage.
    """
    similarity = query_embs @ passage_embs.t()  # (B, B)
    batch_size = similarity.size(0)
    labels = torch.arange(batch_size, device=similarity.device)

    sorted_indices = torch.argsort(similarity, dim=1, descending=True)

    ranks = []
    for i in range(batch_size):
        rank = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    mrr = float(np.mean(1.0 / ranks))
    recall_1 = float(np.mean(ranks <= 1))
    recall_5 = float(np.mean(ranks <= 5))
    recall_10 = float(np.mean(ranks <= 10))

    return {
        "mrr": mrr,
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
    }


class RetrievalMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """
        После evaluation считаем MRR и Recall@k на всём eval наборе.
        """
        model.eval()
        all_query_embs = []
        all_passage_embs = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                all_query_embs.append(outputs["query_emb"].cpu())
                all_passage_embs.append(outputs["passage_emb"].cpu())

        query_embs = torch.cat(all_query_embs, dim=0)
        passage_embs = torch.cat(all_passage_embs, dim=0)

        metrics = compute_retrieval_metrics(query_embs, passage_embs)

        print("\n" + "=" * 50)
        print("Retrieval Metrics on eval:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 50 + "\n")


# ==================== КАСТОМНЫЙ TRAINER ====================

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Совместимо с Trainer, который может передавать доп. аргументы
        (например, num_items_in_batch).
        """
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss



# ==================== MAIN ====================

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-файнтюнинг Qwen3-Embedding-8B (query/pos/neg)")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-8B",
                        help="Имя базовой модели на HuggingFace")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Путь к train.jsonl (query/pos/neg)")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Путь к eval.jsonl (query/pos/neg)")
    parser.add_argument("--output_dir", type=str, default="qwen3_embedding_lora_salyqai",
                        help="Директория для сохранения модели")

    # гиперпараметры
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.02)

    parser.add_argument("--max_query_len", type=int, default=128)
    parser.add_argument("--max_passage_len", type=int, default=512)

    parser.add_argument("--bf16", action="store_true", help="Использовать bfloat16 (если GPU умеет)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== ПАРАМЕТРЫ ТРЕНИРОВКИ ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    # ===== ЖЕСТКО ВЫБИРАЕМ GPU ===== # <-- тут меняешь номер карты

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    device = torch.device("cuda")

    print(f"[INFO] Используем device: {device}")
    print("current_device():", torch.cuda.current_device())
    print("device_name:", torch.cuda.get_device_name())
    # ---------- TOKENIZER ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- BASE MODEL ----------
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    base_model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)

    if base_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # ---------- LoRA ----------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()
    base_model.to(device)

    model = QwenEmbeddingContrastiveModel(base_model, temperature=args.temperature).to(device)

    # ---------- DATA ----------
    train_ds = build_hf_dataset(args.train_file, add_instructions=True)
    eval_ds = build_hf_dataset(args.eval_file, add_instructions=True)

    print(f"[DATA] Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # ---------- DATA COLLATOR ----------
    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len,
    )

    # ---------- TRAINING ARGS ----------
    # ---------- TRAINING ARGS ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,

        bf16=args.bf16,
        fp16=not args.bf16,

        logging_steps=10,
        logging_first_step=True,
        save_total_limit=3,

        load_best_model_at_end=False,
        report_to=[],  # без wandb и т.п.
        remove_unused_columns=False,
    )

    # ---------- TRAINER ----------
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[RetrievalMetricsCallback()],
    )

    effective_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps
    approx_steps_per_epoch = len(train_ds) // effective_bs if effective_bs > 0 else 0
    print("\n🚀 Начинаем обучение...")
    print(f"  Эффективный batch size: {effective_bs}")
    print(f"  Приблизительно шагов на эпоху: {approx_steps_per_epoch}")
    print(f"  Temperature: {args.temperature}\n")

    trainer.train()

    # ---------- SAVE ----------
    os.makedirs(args.output_dir, exist_ok=True)

    trainer.model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": args.model_name,
        "temperature": args.temperature,
        "max_query_len": args.max_query_len,
        "max_passage_len": args.max_passage_len,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово! Модель и токенайзер сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()
