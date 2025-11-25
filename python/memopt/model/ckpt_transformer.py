import torch

from memopt.actckpt import checkpoint
from memopt.model.transformer import Transformer


class BlockwiseCheckpointedTransformer(Transformer):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = checkpoint(layer, x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
