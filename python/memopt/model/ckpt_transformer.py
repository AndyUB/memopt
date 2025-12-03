import functools
import torch

from memopt.actckpt import checkpoint
from memopt.model.transformer import Transformer, TransformerBlock


class BlockwiseCheckpointedTransformer(Transformer):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = checkpoint(layer, x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


def attn_fwd_in_block(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    n1 = block.ln1(x)
    attn_output = block.attn(n1)
    y = x + attn_output
    return y


def ffn_fwd_in_block(block: TransformerBlock, y: torch.Tensor) -> torch.Tensor:
    n2 = block.ln2(y)
    z = block.ffn(n2)
    out = y + z
    return out


def attn_ckpted_block_fwd(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    y = checkpoint(functools.partial(attn_fwd_in_block, block), x)
    out = ffn_fwd_in_block(block, y)
    return out


def ffn_ckpted_block_fwd(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    y = attn_fwd_in_block(block, x)
    out = checkpoint(functools.partial(ffn_fwd_in_block, block), y)
    return out


class AttnCheckpointedTransformer(Transformer):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = attn_ckpted_block_fwd(layer, x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


class FFNCheckpointedTransformer(Transformer):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = ffn_ckpted_block_fwd(layer, x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
