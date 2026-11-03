"""Launch contrastive fine-tuning of the CodeT5 encoder."""
import os
import json
import argparse
from contrastlearn import train_contrastive


def merge_multilingual_data(dataset_root, languages, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    merged_train_path = os.path.join(cache_dir, 'train.jsonl')

    total_samples = 0
    print("Merging multilingual training data...")
    with open(merged_train_path, 'w', encoding='utf-8') as out_f:
        for lang in languages:
            data_path = os.path.join(dataset_root, lang, 'train.jsonl')
            if not os.path.exists(data_path):
                print(f"  Warning: {data_path} does not exist; skipping {lang}.")
                continue

            count = 0
            with open(data_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        js = json.loads(line)
                        if js.get('code_tokens') or js.get('code'):
                            if js.get('docstring_tokens'):
                                out_f.write(line + '\n')
                                count += 1
                    except json.JSONDecodeError:
                        continue

            total_samples += count
            print(f"  {lang}: {count} samples")

    print(f"  Total: {total_samples} samples -> {merged_train_path}\n")
    return cache_dir


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning of the CodeT5 encoder")
    parser.add_argument('--model_name_or_path', type=str, default='/mnt/proj/exp/model/CodeT5-base/')
    parser.add_argument('--output_dir', type=str, default='/mnt/proj/exp/saved_model/finetuned_retriver/')
    parser.add_argument('--local_files_only', action='store_true', default=False)
    parser.add_argument('--dataset_root', type=str, default='/mnt/proj/exp/dataset/', help='CodeXGlue dataset root')
    parser.add_argument('--cache_dir', type=str, default='/mnt/proj/exp/saved_model/muti_cache',
                        help='Cache directory for merged multilingual data')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=['python', 'java', 'ruby', 'javascript', 'php', 'go'])
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval; step 1 is always logged')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')

    args = parser.parse_args()


    print("=" * 70)
    print("Multilingual contrastive fine-tuning - CodeT5 encoder")
    print("=" * 70)
    print(f"Model: {args.model_name_or_path}")
    print(f"Languages: {args.languages}")
    print(f"Output directory: {args.output_dir}")
    print()

    cache_root = merge_multilingual_data(args.dataset_root, args.languages, args.cache_dir)
    args.dataset_root = cache_root
    args.dataset = ''
    args.split = 'train'

    train_contrastive(args)

    print("\n" + "=" * 70)
    print("Fine-tuning complete. Next steps:")
    print(f"  1. Final model: {args.output_dir}")
    print(f"  2. Epoch checkpoints: {args.output_dir}/checkpoint-epoch-N")
    print("  3. Evaluate: python eval_retriever.py --retriever_path", args.output_dir)
    print("  4. Meta-training: python run_pipeline.py --do_train")
    print("=" * 70)


if __name__ == '__main__':
    main()
