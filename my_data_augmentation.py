import random
import re
import string

# Extra vocabulary used for identifier renaming.
EXTRA_VOCAB = ["temp_var", "helper_obj", "val_x", "idx_cnt", "data_buf", "node_item", "result_out"]

CODE_KEYWORDS = {
    "if", "else", "elif", "for", "while", "return", "def", "class", "public", "private",
    "function", "func", "var", "let", "echo", "pass", "break", "continue", "import",
    "from", "as", "try", "except", "finally", "raise", "with", "lambda", "yield",
    "true", "false", "null", "None", "self", "this", "new", "void", "int", "float",
    "str", "bool", "package", "func", "struct", "interface", "type", "const",
}


def get_dummy_tokens(language: str):
    """Return a short dummy token sequence for the given language."""
    dummies = {
        "python": [["dummy_var", "=", "0"], ["pass"]],
        "ruby": [["dummy_var", "=", "0"], ["nil"]],
        "java": [["int", "dummyVar", "=", "0", ";"]],
        "javascript": [["let", "dummyVar", "=", "0", ";"]],
        "php": [["$dummyVar", "=", "0", ";"]],
        "go": [["var", "dummyVar", "int", "=", "0"]],
    }
    return random.choice(dummies.get(language, [["dummy_var", "=", "0"]]))


def augment_rename_identifiers_tokens(tokens):
    """Rename identifier tokens in a CodeXGlue code_tokens list."""
    candidates = [
        t for t in set(tokens)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t) and t not in CODE_KEYWORDS and len(t) > 2
    ]
    if not candidates:
        return tokens

    out = list(tokens)
    num_to_replace = random.randint(1, min(3, len(candidates)))
    for target in random.sample(candidates, num_to_replace):
        new_name = random.choice(EXTRA_VOCAB) + "_" + "".join(random.choices(string.ascii_lowercase, k=3))
        out = [new_name if tok == target else tok for tok in out]
    return out


def augment_insert_dummy_tokens(tokens, language: str):
    """Insert a dummy token sub-sequence at a random position."""
    if not tokens:
        return tokens
    dummy = get_dummy_tokens(language)
    pos = random.randint(0, len(tokens))
    return tokens[:pos] + dummy + tokens[pos:]


def _statement_spans(tokens):
    """Split code_tokens into spans separated by ';' tokens."""
    spans = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok == ";":
            if i > start:
                spans.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        spans.append((start, len(tokens)))
    return spans


def augment_swap_independent_statements_tokens(tokens):
    """Swap two token spans that share no identifier overlap."""
    spans = _statement_spans(tokens)
    if len(spans) < 2:
        return tokens

    random.shuffle(spans)
    for i in range(len(spans) - 1):
        s1, e1 = spans[i]
        s2, e2 = spans[i + 1]
        ids1 = {t for t in tokens[s1:e1] if re.fullmatch(r"[A-Za-z_]\w*", t) and t not in CODE_KEYWORDS}
        ids2 = {t for t in tokens[s2:e2] if re.fullmatch(r"[A-Za-z_]\w*", t) and t not in CODE_KEYWORDS}
        if ids1.isdisjoint(ids2):
            seg1 = tokens[s1:e1]
            seg2 = tokens[s2:e2]
            return tokens[:s1] + seg2 + tokens[e1:s2] + seg1 + tokens[e2:]
    return tokens


def augment_sample_statement_tokens(tokens):
    """Remove one statement span (between ';' boundaries)."""
    spans = _statement_spans(tokens)
    if len(spans) <= 1:
        return tokens
    start, end = random.choice(spans[1:])
    if end - start <= 1:
        return tokens
    return tokens[:start] + tokens[end:]


def apply_c2c_augmentation_tokens(tokens, language: str):
    """
    C2C view-B augmentation: pick exactly one operator by weight, then guarantee
    the result differs from the original token list.

    Operator weights: rename 50%, insert/swap/delete 30% each.
    If the chosen operator makes no change, force identifier rename; if still
    unchanged, force dummy insertion.
    """
    orig = list(tokens)
    if not orig:
        return orig

    operators = (
        (augment_rename_identifiers_tokens, 50),
        (lambda t: augment_insert_dummy_tokens(t, language), 30),
        (augment_swap_independent_statements_tokens, 30),
        (augment_sample_statement_tokens, 30),
    )
    funcs, weights = zip(*operators)
    aug_fn = random.choices(funcs, weights=weights, k=1)[0]
    aug = aug_fn(list(orig))

    if aug == orig:
        aug = augment_rename_identifiers_tokens(list(orig))
    if aug == orig:
        aug = augment_insert_dummy_tokens(list(orig), language)
    return aug


def apply_online_augmentation_tokens(tokens, language: str):
    """Legacy multi-operator probabilistic augmentation (deprecated for C2C training)."""
    return apply_c2c_augmentation_tokens(tokens, language)


# ---- legacy string-based augmentation (for raw source code, not code_tokens) ----

def get_dummy_statement(language: str) -> str:
    dummies = {
        "python": ["dummy_var = 0", "pass", "temp_noop = None"],
        "ruby": ["dummy_var = 0", "nil", "temp_noop = 1"],
        "java": ["int dummyVar = 0;", "String tempNoop = \"\";"],
        "javascript": ["let dummyVar = 0;", "const tempNoop = null;"],
        "php": ["$dummyVar = 0;", "$tempNoop = null;"],
        "go": ["var dummyVar int = 0", "_ = dummyVar"],
    }
    return random.choice(dummies.get(language, ["dummy_var = 0;"]))


def augment_rename_identifiers(code_str: str) -> str:
    identifiers = list(set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code_str)))
    keywords = {"if", "else", "for", "while", "return", "def", "class", "public", "private", "function", "func", "var", "let", "echo"}
    candidates = [ident for ident in identifiers if ident not in keywords and len(ident) > 2]
    if not candidates:
        return code_str
    num_to_replace = random.randint(1, min(3, len(candidates)))
    to_replace = random.sample(candidates, num_to_replace)
    augmented_code = code_str
    for target in to_replace:
        new_name = random.choice(EXTRA_VOCAB) + "_" + "".join(random.choices(string.ascii_lowercase, k=3))
        augmented_code = re.sub(rf'\b{target}\b', new_name, augmented_code)
    return augmented_code


def augment_insert_dummy(code_str: str, language: str) -> str:
    lines = code_str.split('\n')
    if len(lines) < 2:
        return code_str
    dummy = get_dummy_statement(language)
    insert_pos = random.randint(1, len(lines) - 1)
    indent_match = re.match(r'^\s*', lines[insert_pos - 1])
    indent = indent_match.group(0) if indent_match else ""
    lines.insert(insert_pos, indent + dummy)
    return '\n'.join(lines)


def augment_swap_independent_statements(code_str: str) -> str:
    lines = code_str.split('\n')
    if len(lines) < 3:
        return code_str
    body_indices = list(range(1, len(lines) - 1))
    random.shuffle(body_indices)
    for i in range(len(body_indices) - 1):
        idx1 = body_indices[i]
        idx2 = body_indices[i + 1]
        words1 = set(re.findall(r'\b[A-Za-z_]+\b', lines[idx1]))
        words2 = set(re.findall(r'\b[A-Za-z_]+\b', lines[idx2]))
        if not words1.intersection(words2):
            lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
            break
    return '\n'.join(lines)


def augment_sample_statement(code_str: str) -> str:
    lines = code_str.split('\n')
    valid_indices = [
        i for i, line in enumerate(lines)
        if i > 0 and i < len(lines) - 1 and len(line.strip()) > 3 and line.strip() not in ['{', '}']
    ]
    if not valid_indices:
        return code_str
    remove_idx = random.choice(valid_indices)
    del lines[remove_idx]
    return '\n'.join(lines)


def apply_online_augmentation(code_str: str, language: str) -> str:
    """Legacy augmentation for multi-line raw source code strings."""
    aug_code = code_str
    if random.random() < 0.3:
        aug_code = augment_rename_identifiers(aug_code)
    if random.random() < 0.3:
        aug_code = augment_insert_dummy(aug_code, language)
    if random.random() < 0.3:
        aug_code = augment_swap_independent_statements(aug_code)
    if random.random() < 0.3:
        aug_code = augment_sample_statement(aug_code)
    return aug_code
