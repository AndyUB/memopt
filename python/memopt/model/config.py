from memopt.data.tokenizer import DEFAULT_TOKENIZER_VOCAB_SIZE
from memopt.model.transformer import Transformer
from memopt.model.ckpt_transformer import (
    AttnCheckpointedTransformer,
    BlockwiseCheckpointedTransformer,
    FFNCheckpointedTransformer,
)
from typing import Type, Any

DEFAULT_ADAMW_ARGS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
}

MLP_TINY = {
    "input_dim": 10,
    "hidden_dim": 20,
    "output_dim": 5,
}

TRANSFORMER_DEFAULT_BATCH_SIZE = 4
TRANSFORMER_SMALL = {
    "vocab_size": DEFAULT_TOKENIZER_VOCAB_SIZE,
    "num_layers": 12,
    "d_model": 768,
    "num_heads": 12,
    "d_ff": 3072,
}
TRANSFORMER_SMALL_SINGLE_PROCESS_CONTEXT_LENGTH = 1024
TRANSFORMER_LARGE = {
    "vocab_size": DEFAULT_TOKENIZER_VOCAB_SIZE,
    "num_layers": 36,
    "d_model": 1280,
    "num_heads": 20,
    "d_ff": 5120,
}
TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH = 256
TRANSFORMER_LARGE_DDP_CONTEXT_LENGTH = 128
TRANSFOMRER_XLARGE = {
    "vocab_size": DEFAULT_TOKENIZER_VOCAB_SIZE,
    "num_layers": 48,
    "d_model": 1536,
    "num_heads": 24,
    "d_ff": 6144,
}

TRAINABLE_TRANSFORMER_CONFIGS = {
    "small": TRANSFORMER_SMALL,
    "large": TRANSFORMER_LARGE,
}


def clone_and_update(obj: dict[str, Any], key: str, new_value: Any) -> dict[str, Any]:
    new_obj = obj.copy()
    new_obj[key] = new_value
    return new_obj


OOM_TRANSFORMER_CONFIGS = {
    **{
        f"large/{n}layers": clone_and_update(TRANSFORMER_LARGE, "num_layers", n)
        for n in range(40, 52)
    },
    "xlarge": TRANSFOMRER_XLARGE,
}


CKPT_STRATEGIES: dict[str, Type[Transformer]] = {
    "Attention": AttnCheckpointedTransformer,
    "FFN": FFNCheckpointedTransformer,
    "Blockwise": BlockwiseCheckpointedTransformer,
    "None": Transformer,
}
