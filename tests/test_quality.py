from types import SimpleNamespace

import pytest
import torch

from mlbench.inference import quality_probe


class Tokenizer:
    mask_token_id = 4

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[0, 2, 3, 1]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1]]),
        }


class Model:
    def __call__(self, input_ids, attention_mask):
        assert input_ids[0, 0] == 0 and input_ids[0, -1] == 1
        assert (input_ids == 4).sum() == 1
        return SimpleNamespace(logits=torch.zeros((1, 4, 8)))


def test_quality_masks_only_eligible_tokens_and_normalizes_loss():
    result = quality_probe(Model(), Tokenizer(), ["first", "second"], 42)
    assert result["quality_tokens"] == 2
    assert result["masked_token_accuracy"] == 0
    assert result["masked_token_nll"] == pytest.approx(torch.log(torch.tensor(8.0)).item())
    assert len(result["quality_predictions"]) == 2
    assert result == quality_probe(Model(), Tokenizer(), ["first", "second"], 42)


def test_empty_quality_corpus_is_rejected():
    with pytest.raises(ValueError, match="no eligible tokens"):
        quality_probe(Model(), Tokenizer(), [], 42)
