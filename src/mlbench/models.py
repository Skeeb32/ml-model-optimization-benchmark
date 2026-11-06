"""Identical task heads and pinned checkpoint revisions."""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForMaskedLM


def load_model(spec, smoke=False):
    if smoke:
        config = BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=256,
        )
        return BertForMaskedLM(config).eval(), None
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=False
    )
    model = (
        AutoModelForMaskedLM.from_pretrained(
            spec["id"],
            revision=spec["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        .cpu()
        .eval()
    )
    return model, tokenizer
