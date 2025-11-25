from memopt.data.tokenizer import DEFAULT_TOKENIZER_VOCAB_SIZE

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
