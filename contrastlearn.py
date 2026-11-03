import os
import time
import random
import json
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)

from my_data_augmentation import apply_c2c_augmentation_tokens
from retriever_utils import _join_codexglue_tokens, _parse_codexglue_record_with_tokens


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairTextDataset(Dataset):
    def __init__(self, dataset_root: str, dataset_name: str, split: str, is_training: bool = False):
        self.is_training = is_training
        self.split = split
        self.languages = []

        jsonl_path = os.path.join(dataset_root, dataset_name, f"{split}.jsonl") if dataset_name else os.path.join(dataset_root, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"数据文件不存在: {jsonl_path}")

        self.code_tokens_list = []
        self.texts = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    js = json.loads(line.strip())
                    code_tokens, doc_tokens = _parse_codexglue_record_with_tokens(js)
                    nl = _join_codexglue_tokens(doc_tokens)
                    lang = js.get('language', 'python')

                    if not code_tokens or not nl:
                        continue
                    self.code_tokens_list.append(code_tokens)
                    self.texts.append(nl)
                    self.languages.append(lang)
                except Exception as e:
                    print(f"  警告: 跳过第 {idx} 行，解析失败: {e}")

        print(f"数据集 {dataset_name}/{split}: {len(self.code_tokens_list)} 样本 (is_training={is_training})")

    def __len__(self) -> int:
        return len(self.code_tokens_list)

    def __getitem__(self, idx: int) -> Tuple[list, str, str]:
        return self.code_tokens_list[idx], self.texts[idx], self.languages[idx]


def collate_fn(examples, tokenizer, max_source_length: int, max_target_length: int,
               is_training: bool = False):
    """
    Build a 2N batch: [original_code x N, augmented_code x N] with paired docstrings.
    View A: original code_tokens (100% unchanged).
    View B: one weighted augmentation (rename 50%, insert/swap/delete 30% each),
            guaranteed to differ literally from view A.
    """
    code_tokens_batch, texts, langs = zip(*examples)

    orig_codes = []
    aug_codes = []
    for code_tokens, lang in zip(code_tokens_batch, langs):
        orig_tokens = list(code_tokens)
        orig_codes.append(_join_codexglue_tokens(orig_tokens))
        if is_training:
            aug_tokens = apply_c2c_augmentation_tokens(orig_tokens, lang)
        else:
            aug_tokens = orig_tokens
        aug_codes.append(_join_codexglue_tokens(aug_tokens))

    codes = orig_codes + aug_codes

    code_batch = tokenizer(
        codes, padding=True, truncation=True, max_length=max_source_length, return_tensors='pt'
    )
    text_batch = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_target_length, return_tensors='pt'
    )
    return {
        'input_ids': code_batch['input_ids'],
        'labels': text_batch['input_ids'],
    }


class ContrastiveGenerator(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer

    def mean_pool_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool + L2 norm (shared by code and docstring)."""
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return F.normalize(pooled, p=2, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.05,
    ):
        batch = input_ids.size(0)
        half = batch // 2
        if half == 0 or batch != half * 2:
            raise ValueError(f"C2C batch must be even (2N layout), got batch_size={batch}")
        if labels.size(0) != half:
            raise ValueError(f"labels must be N={half}, got {labels.size(0)}")

        code_feature = self.mean_pool_emb(input_ids)
        nl_feature = self.mean_pool_emb(labels)

        ce = nn.CrossEntropyLoss()

        orig_feat = code_feature[:half]
        aug_feat = code_feature[half:]

        # c2c: orig_i <-> aug_i only (N-by-N cross-view; avoids diagonal self-similarity).
        c2c_orig2aug = torch.einsum("bd,cd->bc", orig_feat, aug_feat)
        c2c_aug2orig = torch.einsum("bd,cd->bc", aug_feat, orig_feat)
        target_c2c = torch.arange(half, device=input_ids.device)
        code2code_loss = (
            ce(c2c_orig2aug / temperature, target_c2c)
            + ce(c2c_aug2orig / temperature, target_c2c)
        ) / 2

        # c2nl: original code views align to their corresponding docstrings.
        c2nl_scores = torch.einsum("bd,cd->bc", orig_feat, nl_feature)
        target_c2nl = torch.arange(half, device=input_ids.device)
        code2nl_loss = ce(c2nl_scores / temperature, target_c2nl)

        return code2code_loss, code2nl_loss


def _log(msg: str):
    print(msg, flush=True)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, log_interval: int):
    model.train()
    running = {'c2c': 0.0, 'c2nl': 0.0, 'steps': 0}
    start = time.time()
    steps_per_epoch = len(dataloader)

    _log(f"Epoch steps: {steps_per_epoch}; logging every {log_interval} steps (step 1 included)")

    for step, batch in enumerate(dataloader, 1):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        code2code_loss, code2nl_loss = model(
            input_ids=input_ids,
            labels=labels,
            temperature=model.temperature,
        )
        loss = code2code_loss + code2nl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running['c2c'] += float(code2code_loss.detach().item())
        running['c2nl'] += float(code2nl_loss.detach().item())
        running['steps'] += 1

        if step == 1 or step % log_interval == 0:
            n = running['steps']
            _log(
                f"[train] step={step}/{steps_per_epoch} "
                f"avg_c2c={running['c2c']/n:.4f} "
                f"avg_c2nl={running['c2nl']/n:.4f} "
                f"loss={loss.item():.4f} "
                f"elapsed={time.time()-start:.1f}s"
            )

    n = max(1, running['steps'])
    return running['c2c'] / n, running['c2nl'] / n


def _save_checkpoint(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _freeze_decoder(model):
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False


def train_contrastive(args):
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print(f"Tokenizer: {type(tokenizer).__name__}, vocab={len(tokenizer)}")

    config = T5Config.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    base = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )

    model = ContrastiveGenerator(config, tokenizer)
    model.load_state_dict(base.state_dict(), strict=False)
    model.temperature = getattr(args, 'temperature', 0.05)
    _freeze_decoder(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Encoder-only fine-tuning; trainable parameters: {trainable:,}")

    dataset = PairTextDataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
        split=args.split,
        is_training=True,
    )

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda ex: collate_fn(
            ex, tokenizer, args.max_source_length, args.max_target_length,
            is_training=True,
        ),
        drop_last=True,
    )

    no_decay = ['bias', 'LayerNorm.weight']
    encoder_params = list(model.encoder.named_parameters())
    grouped = [
        {
            'params': [p for n, p in encoder_params if p.requires_grad and not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in encoder_params if p.requires_grad and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=args.lr)

    total_steps = args.epochs * len(dl)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    _log(f"\nTraining: epochs={args.epochs} batch_size={args.batch_size} (2N={args.batch_size * 2} codes/step) "
         f"steps/epoch={len(dl)} total_steps={total_steps} warmup={warmup_steps}")
    _log("Loss: MeanPool(code)↔MeanPool(aug_code) (c2c) + "
         "MeanPool(code)↔MeanPool(docstring) (c2nl)")
    _log("Code/docstring embeddings use mean pooling + L2 normalization; "
         "collate layout: 2N code + N docstring")
    _log(f"temperature={model.temperature}, log_interval={args.log_interval}")
    _log("=" * 60)

    for epoch in range(1, args.epochs + 1):
        _log(f"\nEPOCH {epoch}/{args.epochs}")
        _log("-" * 40)

        avg_c2c, avg_c2nl = train_one_epoch(
            model, dl, optimizer, scheduler, device, args.log_interval
        )
        print(f"\n[EPOCH {epoch} complete] avg_c2c={avg_c2c:.4f} avg_c2nl={avg_c2nl:.4f}")

        if getattr(args, 'output_dir', None):
            ckpt_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
            _save_checkpoint(model, tokenizer, ckpt_dir)
            print(f"  Checkpoint saved: {ckpt_dir}")

    print("\n" + "=" * 60)
    print("Training complete!")

    if getattr(args, 'output_dir', None):
        _save_checkpoint(model, tokenizer, args.output_dir)
        print(f"Final model saved: {args.output_dir}")

    return model, tokenizer
