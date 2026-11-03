import os
import random
import time
import higher

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from retriever_utils import create_frozen_retriever
import torch.optim as optim


def _build_val_subset(retriever, val_sample_size, seed=42):
    valid_records = retriever.load_eval_records(split='valid')
    if not valid_records:
        print("Warning: valid split is empty; early stopping disabled.")
        return []
    rng = random.Random(seed)
    if len(valid_records) <= val_sample_size:
        subset = list(valid_records)
    else:
        subset = rng.sample(valid_records, val_sample_size)
    print(f"Fixed validation subset: {len(subset)} samples (from {len(valid_records)} valid)")
    return subset


def evaluate_meta_loss(model, tokenizer, val_subset, retriever, args, device):
    """Meta-adapted forward loss: inner-loop adapt on support, then target loss."""
    if not val_subset:
        return None

    theta_0 = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    total_loss = 0.0
    valid_count = 0

    model.train()

    for item in val_subset:
        support_set = retriever.search(
            item['code'], top_k=args.top_k_train, threshold=args.sim_threshold_train
        )
        if len(support_set) < 2:
            continue

        valid_count += 1
        num_support = len(support_set)
        eval_lr = getattr(args, 'eval_inner_lr', 3e-4)
        inner_optimizer = optim.SGD(model.parameters(), lr=eval_lr)

        inputs = [s['code'] for s in support_set]
        label_texts = [s['summary'] for s in support_set]

        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding=True,
            truncation=True, return_tensors='pt'
        ).to(device)

        labels_tensors = tokenizer(
            label_texts, max_length=64, padding=True, truncation=True,
            return_tensors='pt'
        ).to(device)

        labels = labels_tensors.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        for _ in range(args.inner_steps_train):
            inner_optimizer.zero_grad()
            outputs = model(**model_inputs, labels=labels)
            logits = outputs.logits

            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            flat_loss = loss_fct(flat_logits, flat_labels)

            token_loss = flat_loss.view(num_support, -1)
            non_pad_mask = (labels.view(num_support, -1) != -100).float()
            sample_loss = token_loss.sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)

            adapt_loss = sample_loss.mean()
            adapt_loss.backward()
            inner_optimizer.step()

        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                [item['code']], max_length=args.max_source_length, padding=True,
                truncation=True, return_tensors='pt'
            ).to(device)
            labels = tokenizer(
                [item['summary']], max_length=64, padding=True,
                truncation=True, return_tensors='pt'
            ).to(device)
            labels.input_ids[labels.input_ids == tokenizer.pad_token_id] = -100
            outputs = model(**inputs, labels=labels.input_ids)
            total_loss += outputs.loss.item()

        model.train()
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(theta_0[name])
                param.grad = None

    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad = None

    if valid_count == 0:
        return None
    return total_loss / valid_count


def run_meta_train(args, retriever=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.codet5_path)
    model = T5ForConditionalGeneration.from_pretrained(args.codet5_path).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    if retriever is None:
        retriever = create_frozen_retriever(args, device)

    outer_optimizer = optim.AdamW(model.parameters(), lr=args.outer_lr)

    sample_temperature = getattr(args, 'sample_temperature', 2.0)
    eval_steps = getattr(args, 'eval_steps', getattr(args, 'eval_interval', 1000))
    patience = getattr(args, 'patience', 3)
    val_sample_size = getattr(args, 'val_sample_size', 200)
    grad_align_tau = getattr(args, 'grad_align_tau', 1.0)

    val_subset = _build_val_subset(retriever, val_sample_size)
    early_stop_enabled = len(val_subset) > 0 and eval_steps > 0

    best_dir = os.path.join(args.output_dir, 'best_meta_model')
    best_val_loss = float('inf')
    patience_counter = 0
    stopped_early = False

    lang_probs = retriever._compute_lang_probs(sample_temperature)
    if retriever.lang_names:
        prob_str = ', '.join(
            f"{lang}={p:.3f}" for lang, p in zip(retriever.lang_names, lang_probs)
        )
        print(f"Language sampling (tau={sample_temperature}): {prob_str}")

    loss = None
    successful_episodes = 0
    skipped_episodes = 0

    # Initialize timing and memory probes.
    episode_times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for episode in range(args.max_episodes):
        # Record the episode start time.
        ep_start_time = time.time()
        target_code, target_summary = retriever.sample_random_target(
            sample_temperature=sample_temperature
        )
        support_set = retriever.search(
            target_code, top_k=args.top_k_train, threshold=args.sim_threshold_train
        )

        if len(support_set) < 2:
            skipped_episodes += 1
            continue

        successful_episodes += 1
        num_support = len(support_set)

        # Use a list for faster parameter snapshots.
        theta_0 = [p.clone().detach() for p in trainable_params]

        # Tokenize the target once.
        target_inputs = tokenizer([target_code], max_length=args.max_source_length, padding=True, truncation=True,
                                  return_tensors="pt").to(device)
        target_labels_tensors = tokenizer([target_summary], max_length=64, padding=True, truncation=True,
                                          return_tensors="pt").to(device)
        t_labels = target_labels_tensors.input_ids.clone()
        t_labels[t_labels == tokenizer.pad_token_id] = -100

        # Tokenize the support set once instead of inside the inner loop.
        s_inputs_all = [item['code'] for item in support_set]
        s_labels_all = [item['summary'] for item in support_set]

        model_inputs_all = tokenizer(s_inputs_all, max_length=args.max_source_length, padding=True, truncation=True,
                                     return_tensors="pt").to(device)
        labels_tensors_all = tokenizer(s_labels_all, max_length=64, padding=True, truncation=True,
                                       return_tensors="pt").to(device)
        labels_all = labels_tensors_all.input_ids.clone()
        labels_all[labels_all == tokenizer.pad_token_id] = -100

        # =====================================================================
        # Select the meta-learning algorithm.
        # =====================================================================
        is_pure_reptile = getattr(args, 'algo_mode', 'meta_rag') == 'reptile'
        is_maml = getattr(args, 'algo_mode', 'meta_rag') == 'maml'

        if is_maml:
            # =====================================================================
            # Full second-order MAML implementation using higher.
            # =====================================================================
            model.train()
            inner_optimizer = optim.SGD(trainable_params, lr=args.inner_lr)

            # Avoid an additional initial parameter copy.
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                # Inner loop: adapt on the support set while retaining the graph.
                for _ in range(args.inner_steps_train):
                    s_outputs = fmodel(**model_inputs_all, labels=labels_all)
                    s_loss = s_outputs.loss
                    # This step extends the second-order computation graph.
                    diffopt.step(s_loss)

                # Outer loop: compute the target meta-loss.
                t_outputs = fmodel(**target_inputs, labels=t_labels)
                t_loss = t_outputs.loss

                # Backpropagate through the inner loop.
                t_loss.backward()

            # Update the global parameters.
            outer_optimizer.step()
            outer_optimizer.zero_grad()

            loss = t_loss  # Keep the shared logging path compatible.

        elif not is_pure_reptile:
            # =====================================================================
            # First-order lookahead probing with dynamic weights.
            # =====================================================================
            target_losses = []
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            temp_optimizer = optim.SGD(
                trainable_params, lr=getattr(args, 'probe_lr', 1e-4)
            )

            for i in range(num_support):
                for p, p_0 in zip(trainable_params, theta_0):
                    p.data.copy_(p_0)

                s_input_single = {k: v[i:i + 1] for k, v in model_inputs_all.items()}
                s_lbl_single = labels_all[i:i + 1]

                model.train()
                temp_optimizer.zero_grad()
                s_out = model(**s_input_single, labels=s_lbl_single)
                s_out.loss.backward()
                temp_optimizer.step()

                model.eval()
                with torch.no_grad():
                    t_out = model(**target_inputs, labels=t_labels)
                    target_losses.append(t_out.loss.item())

            loss_tensor = torch.tensor(target_losses, device=device)
            weight_tau = getattr(args, 'weight_tau', 1.0)
            dynamic_weights = F.softmax(-loss_tensor / weight_tau, dim=0) * num_support

            for p, p_0 in zip(trainable_params, theta_0):
                p.data.copy_(p_0)
        else:
            # =====================================================================
            # Pure Reptile uses uniform weights.
            # =====================================================================
            dynamic_weights = torch.ones(num_support, device=device)

        # =====================================================================
        # Run the first-order inner loop for Meta-RAG and Reptile.
        # =====================================================================
        if not is_maml:
            model.train()
            inner_optimizer = optim.SGD(trainable_params, lr=args.inner_lr)

            for _ in range(args.inner_steps_train):
                inner_optimizer.zero_grad()
                outputs = model(**model_inputs_all, labels=labels_all)

                logits = outputs.logits
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_labels = labels_all.reshape(-1)
                flat_loss = loss_fct(flat_logits, flat_labels)

                token_loss = flat_loss.view(num_support, -1)
                non_pad_mask = (labels_all.view(num_support, -1) != -100).float()
                sample_loss = token_loss.sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)

                loss = (sample_loss * dynamic_weights).mean()
                loss.backward()
                inner_optimizer.step()

            outer_optimizer.zero_grad()
            for p, p_0 in zip(trainable_params, theta_0):
                pseudo_grad = p_0 - p.data
                p.grad = pseudo_grad
                p.data.copy_(p_0)

            outer_optimizer.step()

        # =====================================================================
        # Validation and logging.
        # =====================================================================
        if early_stop_enabled and successful_episodes > 0 and successful_episodes % eval_steps == 0:
            val_loss = evaluate_meta_loss(
                model, tokenizer, val_subset, retriever, args, device
            )
            if val_loss is None:
                print(f"[Eval @ update {successful_episodes}] skipped.")
                model.train()
                continue

            print(
                f"[Eval @ update {successful_episodes}, raw episode {episode + 1}] meta_val_loss={val_loss:.4f} (best={best_val_loss:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                AutoTokenizer.from_pretrained(args.codet5_path).save_pretrained(best_dir)
                print(f"  -> New best saved to {best_dir}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at update {successful_episodes} (raw episode {episode + 1})")
                    stopped_early = True
                    break
            model.train()

        # Record the episode duration.
        episode_times.append(time.time() - ep_start_time)

        # Keep the 100-update progress output focused.
        if successful_episodes > 0 and successful_episodes % 100 == 0:
            avg_ep_time = sum(episode_times[-100:]) / 100

            mem_info = ""
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                mem_info = f" | Peak Mem: {peak_mem:.0f}MB"

            print(
                f"Update {successful_episodes} (raw {episode + 1}/{args.max_episodes}). "
                f"Loss: {loss.item():.4f} | Time/Ep: {avg_ep_time:.3f}s{mem_info}"
            )
        # =========================================================================
        # ===== End of the episode loop. =====
        # =========================================================================

    print(
        f"Meta-train done: successful_episodes={successful_episodes}, "
        f"skipped={skipped_episodes}, stopped_early={stopped_early}"
    )

    save_dir = args.output_dir
    if os.path.isdir(best_dir):
        print(
            f"Loading best checkpoint from {best_dir} "
            f"(best_val_loss={best_val_loss:.4f}) for final export."
        )
        model = T5ForConditionalGeneration.from_pretrained(best_dir).to(device)
    else:
        print("No best checkpoint found; exporting final episode weights.")

    print(f"Saving meta generator to {save_dir} ...")
    model.save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(args.codet5_path).save_pretrained(save_dir)