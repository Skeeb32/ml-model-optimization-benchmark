"""Forward-only timing and a small, explicitly diagnostic quality probe."""

import hashlib
import time

import numpy as np
import torch


def make_inputs(vocab_size, batch_size, length, seed):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, vocab_size, (batch_size, length), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def time_forward(model, inputs, warmup, iterations):
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(**inputs)
        for _ in range(iterations):
            start = time.perf_counter_ns()
            output = model(**inputs)
            timings.append((time.perf_counter_ns() - start) / 1e9)
            del output
    return timings


def quality_probe(model, tokenizer, texts, seed):
    generator = torch.Generator().manual_seed(seed)
    correct, total, loss_sum, predictions, digest = 0, 0, 0.0, [], hashlib.sha256()
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                return_special_tokens_mask=True,
            )
            special = inputs.pop("special_tokens_mask").bool()
            positions = torch.nonzero(~special[0], as_tuple=False).flatten()
            if not len(positions):
                continue
            n = max(1, round(len(positions) * 0.15))
            chosen = positions[torch.randperm(len(positions), generator=generator)[:n]]
            targets = inputs["input_ids"][0, chosen].clone()
            inputs["input_ids"][0, chosen] = tokenizer.mask_token_id
            digest.update(inputs["input_ids"].numpy().tobytes())
            digest.update(targets.numpy().tobytes())
            logits = model(**inputs).logits[0, chosen]
            predicted = logits.argmax(-1)
            correct += int((predicted == targets).sum())
            total += n
            loss_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
            predictions.extend(predicted.tolist())
    if total == 0:
        raise ValueError("Quality corpus contains no eligible tokens")
    return {
        "masked_token_accuracy": correct / total,
        "masked_token_nll": loss_sum / total,
        "quality_tokens": total,
        "quality_input_sha256": digest.hexdigest(),
        "quality_predictions": predictions,
    }


def agreement(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Prediction lists must have matching, nonzero length")
    return float(np.mean(np.asarray(left) == np.asarray(right)))
