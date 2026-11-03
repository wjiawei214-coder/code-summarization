import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
import random


def _index_cache_fingerprint(encoder_path, data_dir, languages, max_index_samples,
                             encode_batch_size, max_seq_length):
    return {
        'encoder_path': os.path.abspath(encoder_path),
        'data_dir': os.path.abspath(data_dir),
        'languages': sorted(languages),
        'max_index_samples': max_index_samples,
        'encode_batch_size': encode_batch_size,
        'max_seq_length': max_seq_length,
        'split': 'train',
    }


def _index_cache_paths(cache_dir):
    cache_dir = os.path.abspath(cache_dir)
    return {
        'meta': os.path.join(cache_dir, 'index_meta.json'),
        'vectors': os.path.join(cache_dir, 'vectors.npy'),
        'corpus': os.path.join(cache_dir, 'corpus.jsonl'),
    }


def _load_index_cache(cache_dir, fingerprint):
    paths = _index_cache_paths(cache_dir)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    with open(paths['meta'], 'r', encoding='utf-8') as f:
        saved = json.load(f)
    if saved != fingerprint:
        return None
    vectors = np.load(paths['vectors'])
    corpus = []
    with open(paths['corpus'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return vectors, corpus


def _save_index_cache(cache_dir, fingerprint, vectors, corpus):
    os.makedirs(cache_dir, exist_ok=True)
    paths = _index_cache_paths(cache_dir)
    np.save(paths['vectors'], vectors.astype(np.float32))
    with open(paths['corpus'], 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(paths['meta'], 'w', encoding='utf-8') as f:
        json.dump(fingerprint, f, ensure_ascii=False, indent=2)
    print(f"  Saved retriever index cache -> {cache_dir} ({len(corpus)} vectors)")


def create_frozen_retriever(args, device):
    """Build or load shared FrozenRetrieverIndex (used by meta train/test in one run)."""
    return FrozenRetrieverIndex(
        encoder_path=args.retriever_path,
        data_dir=args.data_dir,
        languages=args.languages,
        device=device,
        base_model_path=args.codet5_path,
        max_index_samples=getattr(args, 'max_index_samples', None),
        encode_batch_size=getattr(args, 'encode_batch_size', 32),
        max_seq_length=args.max_source_length,
        use_faiss_gpu=getattr(args, 'use_faiss_gpu', False),
        index_cache_dir=getattr(args, 'index_cache_dir', None),
    )


class FaissVectorIndex:
    """Dense retrieval index via Faiss inner-product search (L2-normalized vectors = cosine)."""

    def __init__(self, vectors: np.ndarray, embed_dim: int = None, use_gpu: bool = False):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "Faiss is not installed. Use: pip install faiss-cpu  or  pip install faiss-gpu"
            ) from exc

        self._faiss = faiss
        vectors = vectors.astype(np.float32)

        if vectors.size == 0:
            if embed_dim is None:
                raise ValueError("embed_dim is required when building an empty Faiss index.")
            self.embed_dim = embed_dim
            self.n_vectors = 0
            self.use_gpu = False
            self.index = faiss.IndexFlatIP(self.embed_dim)
            return

        self.n_vectors, self.embed_dim = vectors.shape
        self.use_gpu = use_gpu and torch.cuda.is_available()

        index = faiss.IndexFlatIP(self.embed_dim)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(vectors)
        self.index = index
        print(f"  Faiss IndexFlatIP: {self.n_vectors} vectors, dim={self.embed_dim}")

    def __len__(self):
        return self.n_vectors

    def search_scores(self, query: np.ndarray, top_k: int):
        """Return (scores, indices) sorted by descending inner product."""
        if self.n_vectors == 0:
            return np.array([]), np.array([], dtype=np.int64)

        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        fetch_k = min(top_k, self.n_vectors)
        scores, indices = self.index.search(query, fetch_k)
        return scores[0], indices[0]


def _load_frozen_encoder(encoder_path, device, base_model_path=None):
    """Load contrastive-finetuned CodeT5 encoder for retrieval."""
    encoder_path = os.path.abspath(encoder_path)
    base_model_path = os.path.abspath(base_model_path or encoder_path)

    config_path = os.path.join(encoder_path, 'config.json')
    weights_path = os.path.join(encoder_path, 'pytorch_model.bin')
    safetensors_path = os.path.join(encoder_path, 'model.safetensors')

    if not os.path.exists(config_path):
        base_config_path = os.path.join(base_model_path, 'config.json')
        if os.path.exists(base_config_path):
            print(f"Warning: {config_path} missing, fallback to config from {base_model_path}")
            config = T5Config.from_pretrained(base_model_path)
        else:
            raise FileNotFoundError(
                f"Invalid retriever checkpoint: {encoder_path}\n"
                f"Missing config.json. Re-run contrastive finetune, or pass --codet5_path "
                f"pointing to a valid CodeT5 base model."
            )
    else:
        config = T5Config.from_pretrained(encoder_path)

    tokenizer_dir = encoder_path if os.path.exists(os.path.join(encoder_path, 'tokenizer_config.json')) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    model = T5ForConditionalGeneration(config)
    if os.path.exists(weights_path) or os.path.exists(safetensors_path):
        try:
            loaded = T5ForConditionalGeneration.from_pretrained(encoder_path)
            model.load_state_dict(loaded.state_dict(), strict=False)
        except (OSError, RuntimeError, ValueError):
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Cannot load retriever weights from {encoder_path}. "
                    "Expected pytorch_model.bin or model.safetensors."
                )
            print(f"Warning: loading legacy state_dict from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"No model weights found in {encoder_path}. "
            "Run run_contrastive_finetune.py first."
        )

    encoder = model.encoder
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder


def _join_codexglue_tokens(tokens):
    """Join CodeXGlue token list to text (same as official CodeXGlue run.py)."""
    if not tokens:
        return ''
    text = ' '.join(tokens).replace('\n', ' ')
    return ' '.join(text.strip().split())


def _parse_codexglue_record(js):
    """Parse one CodeXGlue Code-to-Text jsonl record into (code, summary)."""
    code = _join_codexglue_tokens(js.get('code_tokens'))
    summary = _join_codexglue_tokens(js.get('docstring_tokens'))
    return code, summary


def _parse_codexglue_record_with_tokens(js):
    """Parse record and also return raw token lists for token-level augmentation."""
    code_tokens = list(js.get('code_tokens') or [])
    doc_tokens = list(js.get('docstring_tokens') or [])
    return code_tokens, doc_tokens


def _load_codexglue_split(data_dir, languages, split):
    records = []
    for lang in languages:
        jsonl_path = os.path.join(data_dir, lang, f'{split}.jsonl')
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {lang}.")
            continue

        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                    code, summary = _parse_codexglue_record(js)
                    if not code or not summary:
                        continue
                    records.append({
                        'code': code,
                        'summary': summary,
                        'language': js.get('language', lang),
                    })
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: skip {lang}/{split}.jsonl line {idx + 1}: {e}")
        print(f"Loaded {count} samples from {lang}/{split}.jsonl")

    return records


class FrozenRetrieverIndex:
    def __init__(self, encoder_path, data_dir, languages, device, base_model_path=None,
                 max_index_samples=None, encode_batch_size=32, max_seq_length=256,
                 use_faiss_gpu=False, index_cache_dir=None):
        self.device = device
        self.data_dir = data_dir
        self.languages = languages
        self.encode_batch_size = encode_batch_size
        self.max_seq_length = max_seq_length
        self.use_faiss_gpu = use_faiss_gpu
        self.encoder_path = encoder_path
        self.index_cache_dir = index_cache_dir

        self.tokenizer, self.encoder = _load_frozen_encoder(
            encoder_path, device, base_model_path=base_model_path
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.embed_dim = self.encoder.config.d_model

        self.corpus = []
        self.index = FaissVectorIndex(
            np.zeros((0, self.embed_dim), dtype='float32'),
            embed_dim=self.embed_dim,
            use_gpu=self.use_faiss_gpu,
        )

        gpu_note = "GPU" if self.use_faiss_gpu and torch.cuda.is_available() else "CPU"
        enc_note = "cuda" if self.device.type == "cuda" else "cpu"
        fingerprint = _index_cache_fingerprint(
            encoder_path, data_dir, languages, max_index_samples,
            encode_batch_size, max_seq_length,
        )
        cached = None
        if index_cache_dir:
            cached = _load_index_cache(index_cache_dir, fingerprint)

        if cached is not None:
            vectors, self.corpus = cached
            self.index = FaissVectorIndex(vectors, use_gpu=self.use_faiss_gpu)
            print(
                f"Loaded retriever index from cache ({len(self.corpus)} vectors, "
                f"encoder={enc_note}, Faiss={gpu_note})"
            )
        else:
            print(
                f"正在构建密集检索库 (encoder={enc_note}, Faiss IndexFlatIP={gpu_note})..."
            )
            self.build_index(data_dir, languages, max_samples=max_index_samples)
            if index_cache_dir and len(self.corpus) > 0 and getattr(self, '_index_vectors', None) is not None:
                _save_index_cache(index_cache_dir, fingerprint, self._index_vectors, self.corpus)

        self._build_lang_corpus()

    @torch.no_grad()
    def mean_pool_emb(self, input_ids, attention_mask):
        """Mean-pool + L2 norm (same as contrastlearn.mean_pool_emb)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return F.normalize(pooled, p=2, dim=1)

    @torch.no_grad()
    def encode_code(self, code_text):
        """Encode a single code string → (1, dim) numpy vector."""
        inputs = self.tokenizer(
            code_text, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.device)
        attention_mask = inputs['input_ids'].ne(self.pad_token_id)
        vec = self.mean_pool_emb(inputs['input_ids'], attention_mask).cpu().numpy().astype('float32')
        return vec

    @torch.no_grad()
    def encode_codes_batch(self, code_texts):
        """Encode code list → (N, dim) numpy matrix for index building."""
        vectors = []
        for start in range(0, len(code_texts), self.encode_batch_size):
            batch = code_texts[start:start + self.encode_batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors='pt'
            ).to(self.device)
            attention_mask = inputs['input_ids'].ne(self.pad_token_id)
            vec = self.mean_pool_emb(inputs['input_ids'], attention_mask).cpu().numpy().astype('float32')
            vectors.append(vec)

        if not vectors:
            return np.zeros((0, self.embed_dim), dtype='float32')
        return np.vstack(vectors).astype('float32')

    @torch.no_grad()
    def build_index(self, data_dir, languages, max_samples=None):
        self.corpus = _load_codexglue_split(data_dir, languages, 'train')

        if len(self.corpus) == 0:
            print("Warning: no training samples loaded from CodeXGlue jsonl files.")
            self.index = FaissVectorIndex(
                np.zeros((0, self.embed_dim), dtype='float32'),
                embed_dim=self.embed_dim,
                use_gpu=self.use_faiss_gpu,
            )
            return

        if max_samples is not None and len(self.corpus) > max_samples:
            rng = random.Random(42)
            self.corpus = rng.sample(self.corpus, max_samples)
            print(f"Subsampled index corpus to {len(self.corpus)} samples.")

        codes = [item['code'] for item in self.corpus]
        print(f"Encoding {len(codes)} codes with mean-pool embedding...")
        vectors = self.encode_codes_batch(codes)
        self._index_vectors = vectors

        self.index = FaissVectorIndex(vectors, use_gpu=self.use_faiss_gpu)
        print(f"Faiss index ready: {vectors.shape[0]} vectors, dim={vectors.shape[1]}")

    def _build_lang_corpus(self):
        """Group train corpus by language for temperature-smoothed sampling."""
        self.lang_corpus = {}
        for item in self.corpus:
            lang = item.get('language', 'unknown')
            self.lang_corpus.setdefault(lang, []).append(item)
        self.lang_names = sorted(self.lang_corpus.keys())
        if self.lang_names:
            counts = {lang: len(self.lang_corpus[lang]) for lang in self.lang_names}
            print(f"Language corpus sizes: {counts}")

    def _compute_lang_probs(self, sample_temperature=2.0):
        """P(L_i) ∝ N_i^(1/tau); tau=1 → proportional; tau→∞ → uniform."""
        if not self.lang_names:
            return np.array([], dtype=np.float64)
        if sample_temperature <= 0:
            raise ValueError(f"sample_temperature must be > 0, got {sample_temperature}")

        counts = np.array(
            [len(self.lang_corpus[lang]) for lang in self.lang_names],
            dtype=np.float64,
        )
        if np.all(counts <= 0):
            return np.ones(len(self.lang_names), dtype=np.float64) / len(self.lang_names)

        weights = np.power(counts, 1.0 / sample_temperature)
        return weights / weights.sum()

    def sample_random_target(self, sample_temperature=2.0):
        if not self.corpus:
            raise RuntimeError("Corpus is empty; cannot sample a target.")
        if not self.lang_corpus:
            self._build_lang_corpus()

        lang_probs = self._compute_lang_probs(sample_temperature)
        lang = np.random.choice(self.lang_names, p=lang_probs)
        target = random.choice(self.lang_corpus[lang])
        return target['code'], target['summary']

    @torch.no_grad()
    def search(self, target_code, top_k=5, threshold=0.75):
        if not self.corpus or not hasattr(self, 'index'):
            return []
        vec = self.encode_code(target_code)
        scores, idxs = self.index.search_scores(vec, top_k + 1)

        support_set = []
        for score, idx in zip(scores, idxs):
            if score >= threshold and self.corpus[idx]['code'] != target_code:
                support_set.append(self.corpus[idx])
            if len(support_set) == top_k:
                break
        return support_set

    @torch.no_grad()
    def search_top_k(self, target_code, top_k=5, exclude_self=True, same_language=None, threshold=-1.0):
        """Return top-k hits with C2C cosine scores (code mean-pool + Faiss)."""
        if not self.corpus or not hasattr(self, 'index'):
            return []

        vec = self.encode_code(target_code)

        fetch_k = top_k + (1 if exclude_self else 0)
        if same_language:
            fetch_k = max(fetch_k * 100, 200)
        else:
            fetch_k = max(fetch_k + 10, top_k * 5)
        fetch_k = min(fetch_k, len(self.corpus))

        scores, idxs = self.index.search_scores(vec, fetch_k)

        results = []
        for score, idx in zip(scores, idxs):
            if exclude_self and self.corpus[idx]['code'] == target_code:
                continue
            if same_language and self.corpus[idx].get('language') != same_language:
                continue
            if score < threshold:
                continue
            results.append((self.corpus[idx], float(score)))
            if len(results) == top_k:
                break
        return results

    @torch.no_grad()
    def search_top1(self, target_code, threshold=-1.0, same_language=None):
        hits = self.search_top_k(
            target_code, top_k=1, exclude_self=True,
            same_language=same_language, threshold=threshold,
        )
        if not hits:
            return None, None
        return hits[0]

    def load_test_data(self, split='test'):
        records = _load_codexglue_split(self.data_dir, self.languages, split)
        return [item['code'] for item in records]

    def load_eval_records(self, split='test'):
        """Load full records (code, summary, language) for generation eval."""
        return _load_codexglue_split(self.data_dir, self.languages, split)
