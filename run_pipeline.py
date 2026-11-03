import argparse
import os
import torch
from meta_train import run_meta_train
from meta_test import run_meta_test
from retriever_utils import create_frozen_retriever


def get_args():
    parser = argparse.ArgumentParser(description="CodeT5 Reptile-RAG Meta Learning")

    # Basic path parameters
    parser.add_argument("--data_dir", type=str, default="/mnt/proj/exp/dataset/", help="CodeXGlue 数据集路径")
    parser.add_argument("--codet5_path", type=str, default="/mnt/proj/exp/model/CodeT5-base/",
                        help="Path to the CodeT5 pretrained model (or Salesforce/codet5-base)")
    parser.add_argument("--retriever_path", type=str,
                        default="/mnt/proj/exp/saved_model/finetuned_retriver/checkpoint-epoch-2/",
                        help="Directory containing the contrastive-finetuned retriever")
    parser.add_argument("--output_dir", type=str, default="/mnt/proj/exp/saved_model/checkpoint/ruby/maml/",
                        help="Directory for saving model checkpoints")

    # Task control
    parser.add_argument("--do_train", action="store_true", help="Run Reptile meta-training")
    parser.add_argument("--do_test", action="store_true", help="Run retrieval-based test-time adaptation")
    parser.add_argument("--languages", nargs='+', default=["ruby"], help="Languages in the training/test pool")

    # Retriever parameters (filtering and limits)
    parser.add_argument("--top_k_train", type=int, default=5, help="Maximum training support-set size (K-shot)")
    parser.add_argument("--top_k_test", type=int, default=5, help="Maximum test-time support-set size")
    parser.add_argument("--sim_threshold_train", type=float, default=0,
                        help="Minimum training cosine similarity for support examples")
    parser.add_argument("--sim_threshold_test", type=float, default=0,
                        help="Minimum test-time similarity for support examples")
    parser.add_argument('--rank_offset', type=int, default=0, help='Retrieval rank offset for ablation study')

    # Test-time ablation parameters
    parser.add_argument('--ablation_mode', type=str, default='standard',
                        choices=['zero_shot', 'standard', 'draft_mmr', 'random'],
                        help="Test-time ablation: zero_shot, standard, draft_mmr, or random")
    parser.add_argument('--mmr_lambda', type=float, default=0.5,
                        help="MMR relevance weight; higher values favor relevance")
    parser.add_argument('--algo_mode', type=str, default='reptile',
                        choices=['reptile', 'maml'])

    # Meta-learning parameters
    parser.add_argument("--inner_steps_train", type=int, default=3, help="Number of training inner-loop steps")
    parser.add_argument("--inner_steps_test", type=int, default=5, help="Number of test-time adaptation steps")
    parser.add_argument("--inner_lr", type=float, default=1e-5, help="Inner-loop SGD learning rate")
    parser.add_argument("--probe_lr", type=float, default=1e-4, help="Lookahead probing SGD learning rate")
    parser.add_argument("--eval_inner_lr", type=float, default=1e-5,
                        help="Validation/test-time adaptation learning rate")
    parser.add_argument("--outer_lr", type=float, default=5e-6, help="Outer-loop AdamW learning rate")
    parser.add_argument("--max_episodes", type=int, default=25000, help="Maximum number of episodes")
    parser.add_argument("--sample_temperature", type=float, default=2.0,
                        help="Language-aware target sampling temperature")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Run meta-validation every N successful updates")
    parser.add_argument("--patience", type=int, default=4,
                        help="Number of evaluations without improvement before early stopping")
    parser.add_argument("--val_sample_size", type=int, default=500,
                        help="Fixed validation subset size for early stopping")

    # System and memory settings
    parser.add_argument("--max_source_length", type=int, default=256, help="Maximum source sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_faiss_gpu", action="store_true",
                        help="Use GPU for Faiss indexing and retrieval (requires faiss-gpu)")
    parser.add_argument("--index_cache_dir", type=str, default="/mnt/proj/exp/saved_model/retriever_index_cache/ruby"
                                                               "/",
                        help="Retriever vector cache directory")
    parser.add_argument("--max_index_samples", type=int, default=None,
                        help="Optional number of training samples to index")
    parser.add_argument("--encode_batch_size", type=int, default=128,
                        help="Batch size for index encoding")
    parser.add_argument("--test_split", type=str, default="test", choices=["valid", "test"],
                        help="Split used for generation evaluation")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Optional limit on evaluated test samples")
    parser.add_argument("--test_output_dir", type=str, default=None,
                        help="Test results directory")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam-search width passed to model.generate")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Generation length penalty")
    parser.add_argument("--max_gen_length", type=int, default=64,
                        help="Maximum generated summary length")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.do_train and not args.do_test:
        print("Neither --do_train nor --do_test was specified; running meta-training by default.")
        args.do_train = True
    os.makedirs(args.output_dir, exist_ok=True)
    if args.index_cache_dir:
        os.makedirs(args.index_cache_dir, exist_ok=True)

    retriever = None
    if args.do_train or args.do_test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Building or loading the shared train/test retriever index...")
        retriever = create_frozen_retriever(args, device)

    if args.do_train:
        print("Starting Reptile meta-training...")
        run_meta_train(args, retriever=retriever)

    if args.do_test:
        print("Starting retrieval-based test-time adaptation...")
        run_meta_test(args, retriever=retriever)
