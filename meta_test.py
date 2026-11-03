import argparse
import copy
import json
import os
import warnings
import time

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from retriever_utils import create_frozen_retriever, _load_codexglue_split
import torch.optim as optim

from eval_retriever import bleu_corpus, bleu_sentence, tokenize, stratified_subsample


def _load_meta_generator(output_dir, codet5_path, device):
    """Load Meta-trained weights from output_dir; tokenizer always from CodeT5-base."""
    output_dir = os.path.abspath(output_dir)
    codet5_path = os.path.abspath(codet5_path)

    weight_names = ('pytorch_model.bin', 'model.safetensors')
    if not any(os.path.exists(os.path.join(output_dir, n)) for n in weight_names):
        raise FileNotFoundError(
            f"No generator weights in {output_dir} (expected pytorch_model.bin or model.safetensors).\n"
            "Point --output_dir to the folder where meta_train called model.save_pretrained()."
        )

    tokenizer = AutoTokenizer.from_pretrained(codet5_path)
    model = T5ForConditionalGeneration.from_pretrained(output_dir).to(device)
    return tokenizer, model


def _save_test_artifacts(save_dir, detail_records, metrics, config):
    os.makedirs(save_dir, exist_ok=True)

    gold_path = os.path.join(save_dir, 'gold.txt')
    pred_path = os.path.join(save_dir, 'predictions.txt')
    with open(gold_path, 'w', encoding='utf-8') as fg, open(pred_path, 'w', encoding='utf-8') as fp:
        for row in detail_records:
            fg.write(row['gold_summary'].replace('\n', ' ').strip() + '\n')
            fp.write(row['prediction'].replace('\n', ' ').strip() + '\n')

    gold_tsv = os.path.join(save_dir, 'gold.codexglue.tsv')
    pred_tsv = os.path.join(save_dir, 'predictions.codexglue.tsv')
    with open(gold_tsv, 'w', encoding='utf-8') as fg, open(pred_tsv, 'w', encoding='utf-8') as fp:
        for row in detail_records:
            fg.write(f"{row['idx']}\t{row['gold_summary'].replace(chr(10), ' ').strip()}\n")
            fp.write(f"{row['idx']}\t{row['prediction'].replace(chr(10), ' ').strip()}\n")

    details_path = os.path.join(save_dir, 'meta_test_details.jsonl')
    with open(details_path, 'w', encoding='utf-8') as f:
        for row in detail_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    metrics_path = os.path.join(save_dir, 'meta_test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'config': config}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved test artifacts -> {save_dir}")
    print(f"  gold.txt / predictions.txt")
    print(f"  gold.codexglue.tsv / predictions.codexglue.tsv")
    print(f"  meta_test_details.jsonl / meta_test_metrics.json")


def generate_summary_internal(model, tokenizer, code, device, max_length, num_beams, length_penalty):
    """测试期代码生成的通用函数"""
    inputs = tokenizer([code], return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_high_quality_draft(model, tokenizer, code, device):
    """生成高质量草稿，用作 MMR 的相关性锚点"""
    inputs = tokenizer([code], return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=64,
            num_beams=5,
            length_penalty=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def mmr_rerank(candidate_pool, draft_tokens, top_k, lambda_mmr):
    """MMR 重排序逻辑：基于候选摘要与草稿的 BLEU 得分"""
    if len(candidate_pool) == 0:
        return []

    for cand in candidate_pool:
        cand['summary_tokens'] = tokenize(cand['summary'])
        cand['relevance'] = bleu_sentence([draft_tokens], [cand['summary_tokens']])

    selected_indices = []
    remaining_indices = list(range(len(candidate_pool)))

    first_idx = max(remaining_indices, key=lambda i: candidate_pool[i]['relevance'])
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < min(top_k, len(candidate_pool)):
        best_mmr_score = -1.0
        best_cand_idx = -1

        for i in remaining_indices:
            cand_i = candidate_pool[i]
            max_redundancy = max([
                bleu_sentence([cand_i['summary_tokens']], [candidate_pool[j]['summary_tokens']])
                for j in selected_indices
            ])
            mmr_score = lambda_mmr * cand_i['relevance'] - (1 - lambda_mmr) * max_redundancy

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_cand_idx = i

        if best_cand_idx == -1:
            break

        selected_indices.append(best_cand_idx)
        remaining_indices.remove(best_cand_idx)

    return [candidate_pool[idx] for idx in selected_indices]


# ==============================================================================
# Core test flow with retrieval and latency metrics.
# ==============================================================================

def run_meta_test(args, retriever=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_split = getattr(args, 'test_split', 'test')
    max_test_samples = getattr(args, 'max_test_samples', None)
    save_dir = getattr(args, 'test_output_dir', None) or os.path.join(args.output_dir, f'meta_test_{args.ablation_mode}')

    warnings.filterwarnings('ignore', message='.*past_key_values.*')

    print(f"Generator weights : {os.path.abspath(args.output_dir)}")
    print(f"Tokenizer         : {os.path.abspath(args.codet5_path)}")
    print(f"Results dir       : {os.path.abspath(save_dir)}")
    print(f"Ablation Mode     : {args.ablation_mode}")

    tokenizer, model = _load_meta_generator(args.output_dir, args.codet5_path, device)
    model.eval()

    if retriever is None:
        retriever = create_frozen_retriever(args, device)

    test_records = _load_codexglue_split(retriever.data_dir, retriever.languages, test_split)
    if max_test_samples is not None and len(test_records) > max_test_samples:
        test_records = stratified_subsample(test_records, max_test_samples)

    beam_size = getattr(args, 'beam_size', 5)
    length_penalty = getattr(args, 'length_penalty', 1.0)
    max_gen_length = getattr(args, 'max_gen_length', 64)

    print(f"Evaluating {len(test_records)} {test_split} samples...")

    global_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    detail_records = []
    gold_tokens = []
    pred_tokens = []

    inference_latencies = []

    for idx, item in enumerate(tqdm(test_records, desc="Generating", unit="sample")):
        sample_start_time = time.time()
        target_code = item['code']
        target_lang = item.get('language', 'ruby')
        gold_summary = item['summary']

        model.load_state_dict({k: v.to(device) for k, v in global_state_dict.items()})

        support_set = []

        # =====================================================================
        # Prepare the support set for the selected ablation.
        # =====================================================================
        if args.ablation_mode == 'zero_shot':
            pass


        elif args.ablation_mode == 'standard':

            # Select a retrieval rank range for the ablation.

            rank_offset = getattr(args, 'rank_offset', 0)

            # Expand the retrieval pool to account for the rank offset.

            fetch_k = args.top_k_test + rank_offset

            raw_hits = retriever.search_top_k(

                target_code, top_k=fetch_k,

                same_language=target_lang, threshold=args.sim_threshold_test

            )

            # Keep only the requested rank range.

            if len(raw_hits) >= fetch_k:

                selected_hits = raw_hits[rank_offset: fetch_k]

            else:

                # Fall back to the last available top-k hits.

                selected_hits = raw_hits[-args.top_k_test:] if len(raw_hits) > args.top_k_test else raw_hits

            # Preserve dense similarity scores for reporting.

            for sample_dict, score in selected_hits:
                sample_dict['dense_similarity'] = float(score)

            support_set = [hit[0] for hit in selected_hits]

        elif args.ablation_mode == 'draft_mmr':
            model.eval()
            draft_text = generate_high_quality_draft(model, tokenizer, target_code, device)
            draft_tokens = tokenize(draft_text)

            pool_size = max(args.top_k_test * 4, 20)
            raw_hits_20 = retriever.search_top_k(
                target_code, top_k=pool_size,
                same_language=target_lang, threshold=args.sim_threshold_test
            )
            # Preserve dense similarity scores for the MMR candidate pool.
            for sample_dict, score in raw_hits_20:
                sample_dict['dense_similarity'] = float(score)
            candidate_pool = [hit[0] for hit in raw_hits_20]

            support_set = mmr_rerank(candidate_pool, draft_tokens, args.top_k_test, args.mmr_lambda)

        elif args.ablation_mode == 'random':
            # Use same-language random retrieval with a reproducible seed.
            support_set = retriever.random_search(
                top_k=args.top_k_test,
                same_language=target_lang,
                seed=idx
            )
            # Randomly selected samples have no dense retrieval score.
            for sample_dict in support_set:
                sample_dict['dense_similarity'] = 0.0

        num_support = len(support_set)

        # Compute the average similarity of the selected support set.
        if num_support > 0:
            avg_similarity = sum(s.get('dense_similarity', 0.0) for s in support_set) / num_support
        else:
            avg_similarity = 0.0

        # ==========================================
        # Adapt on the support set when it is non-empty.
        # ==========================================
        if num_support > 0:
            model.train()
            eval_lr = getattr(args, 'eval_inner_lr', 1e-5)
            test_optimizer = optim.AdamW(model.parameters(), lr=eval_lr)

            inputs = [s['code'] for s in support_set]
            labels_text = [s['summary'] for s in support_set]

            model_inputs = tokenizer(
                inputs, max_length=args.max_source_length, padding=True,
                truncation=True, return_tensors="pt"
            ).to(device)

            labels_tensors = tokenizer(
                labels_text, max_length=64, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels_tensors.input_ids[labels_tensors.input_ids == tokenizer.pad_token_id] = -100

            for _ in range(args.inner_steps_test):
                test_optimizer.zero_grad()
                outputs = model(**model_inputs, labels=labels_tensors.input_ids)
                loss = outputs.loss
                loss.backward()
                test_optimizer.step()

        # ==========================================
        # Generate the summary and record metrics.
        # ==========================================
        model.eval()
        prediction = generate_summary_internal(
            model, tokenizer, target_code, device, max_gen_length, beam_size, length_penalty
        )

        gold_t = tokenize(gold_summary)
        pred_t = tokenize(prediction)
        gold_tokens.append(gold_t)
        pred_tokens.append(pred_t)
        sent_bleu = bleu_sentence([gold_t], [pred_t])

        latency = time.time() - sample_start_time
        inference_latencies.append(latency)

        # Store the average retrieval similarity in the per-sample record.
        detail_records.append({
            'idx': idx,
            'language': target_lang,
            'gold_summary': gold_summary,
            'prediction': prediction,
            'sentence_bleu': sent_bleu,
            'num_support': num_support,
            'avg_similarity': avg_similarity,  # Key retrieval metric.
            'latency_sec': latency,
            'query_code': target_code[:500],
        })

    # ==========================================
    # Aggregate global metrics and save artifacts.
    # ==========================================
    bleu_c = bleu_corpus(gold_tokens, pred_tokens)
    bleu_s = bleu_sentence(gold_tokens, pred_tokens)
    avg_support = sum(r['num_support'] for r in detail_records) / max(len(detail_records), 1)
    zero_support = sum(1 for r in detail_records if r['num_support'] == 0)

    metrics = {
        'bleu_corpus': bleu_c,
        'bleu_sentence': bleu_s,
        'num_samples': len(test_records),
        'split': test_split,
        'avg_support_size': avg_support,
        'zero_support_rate': zero_support / max(len(detail_records), 1),
        'avg_inference_latency_sec': sum(inference_latencies) / len(inference_latencies)
    }

    config = {
        'ablation_mode': args.ablation_mode,
        'mmr_lambda': getattr(args, 'mmr_lambda', 0.6),
        'output_dir': args.output_dir,
        'codet5_path': args.codet5_path,
        'retriever_path': args.retriever_path,
        'top_k_test': args.top_k_test,
        'sim_threshold_test': args.sim_threshold_test,
        'inner_steps_test': args.inner_steps_test,
        'inner_lr': args.inner_lr,
        'eval_inner_lr': getattr(args, 'eval_inner_lr', 2e-3),
        'max_test_samples': max_test_samples,
        'languages': args.languages,
        'beam_size': beam_size,
        'length_penalty': length_penalty,
        'max_gen_length': max_gen_length,
    }

    print("\n" + "=" * 70)
    print(f"Meta Test Results (Mode: {args.ablation_mode.upper()})")
    print("=" * 70)
    print(f"  BLEU-4 (corpus)      : {bleu_c * 100:.2f}")
    print(f"  BLEU-4 (sentence)    : {bleu_s * 100:.2f}")
    print("=" * 70)

    _save_test_artifacts(save_dir, detail_records, metrics, config)

    legacy_pred = os.path.join(args.output_dir, f'predictions_{args.ablation_mode}.txt')
    with open(legacy_pred, 'w', encoding='utf-8') as f:
        for row in detail_records:
            f.write(row['prediction'].replace('\n', ' ').strip() + '\n')

    return metrics