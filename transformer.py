"""Implementation of GPT models."""
import torch
import math
from torch import nn

class CausalSelfAttention(nn.Module):
    """Naive implementation of multi-head self-attention in a transformer decoder.

    This was just implemented for some testing. Use the FlashAttention class below which is much
    faster.
    """

    def __init__(self, in_channels: int, n_heads: int, context_window: int):
        """Initialize."""
        super().__init__()
        assert in_channels%n_heads == 0, "The state dimensionality should be divisible by the number of attention heads."

        # It is faster to make this one big Linear transform.
        self._query_transf = nn.Linear(in_channels, in_channels, bias=False)
        self._key_transf = nn.Linear(in_channels, in_channels, bias=False)
        self._value_transf = nn.Linear(in_channels, in_channels, bias=False)
        self._n_heads = n_heads
        self._view_dim = (n_heads, in_channels//n_heads)
        self.register_buffer("mask", torch.tril(torch.ones(context_window, context_window)))


    def forward(self, x: torch.Tensor):
        """Forward pass."""
        B, T, C = x.shape
        queries = self._query_transf(x).view(B, self._view_dim[0], T, self._view_dim[1])
        keys = self._key_transf(x).view(B, self._view_dim[0], T, self._view_dim[1])
        values = self._value_transf(x).view(B, self._view_dim[0], T, self._view_dim[1])
        
        attention = queries @ keys.transpose(-2, -1)
        attention = attention / (self._view_dim[1]**0.5)
        attention = attention.masked_fill(self.mask==0, float("-inf"))
        attention = nn.functional.softmax(attention, dim=-1)
        
        output = attention @ values
        return output.view(B, T, C)


class CausalFlashSelfAttention(nn.Module):
    """Multi-head self-attention in transformer decoder using flash attention https://arxiv.org/abs/2205.14135.

    This is significantly faster than the naive implementation above and should be used instead for
    model training.
    """
    def __init__(self, in_channels: int, n_heads: int, dropout_p: float = 0.0):
        """Initialize.

        In this case we do not need the sequence length since the pytorch flash attention
        implementation is dynamic w.r.t. the sequence length.
        """
        super().__init__()
        assert in_channels%n_heads == 0, "The state dimensionality should be divisible by the number of attention heads."

        # One large transform is faster than three separate ones for keys, queries and values.
        self._att_transf = nn.Linear(in_channels, 3 * in_channels, bias=False)

        # The original transformer paper (https://arxiv.org/abs/1706.03762) also does a projection
        # as a part of the attention block. WE apply this as well.
        self._out_transf = nn.Linear(in_channels, in_channels, bias=False)
        self._n_heads = n_heads
        self._head_dim = in_channels//n_heads
        self._in_channels = in_channels
        self._dropout_p = dropout_p

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        B, T, C = x.shape

        # First we apply one big transformation, then we split the result in three separate vectors.
        # Note that this is equivalent to applying three separate linear transforms, but is faster.
        # The same applies to all the attention heads, we can do their respective transforms in one
        # big one.
        # The split happens along the last dimension.
        queries, keys, values = self._att_transf(x).split(self._in_channels, dim=-1)

        # We need to transform the keys queries and values to be able to apply multiple attention
        # heads.
        queries = queries.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
        keys = keys.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)
        values = values.view(B, T, self._n_heads, self._head_dim).transpose(1, 2)

        # Torch implementation of flash attention.
        # is_causal is crucial here, since otherwise the model will be able to look at tokens into
        # the future and cheat.
        out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self._dropout_p if self.training else 0, is_causal=True)

        # Concatenate the channels from the multiple heads before applying the final linear
        # projection.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self._out_transf(out)



def _ln(x: torch.Tensor):
    """Making layernorm easier to call."""
    return torch.nn.functional.layer_norm(x, x.shape[-1])


class TransformerBlock(nn.Module):
    """Implementation of a transformer block, which contains an attention layer."""
    def __init__(self, in_channels: int, n_heads: int, expansion_factor: int = 4, dropout_p: float = 0.0):
        """Initialize the transformer block.
    
        After the attention layer a two-layer MLP is applied. Both the attention operation and the
        MLP have a parallel residual connection.

        Layernorm will be applied before attention and before the MLP that follows. This is not how
        it is done in the original transformer paper, but it makes the training procedure of the
        transformative much less sensitive. Using a normal optimizer without a specially finetuned
        learning rate schedule will converge with this architecture. Achieving congergence with the
        original layernorm placement is much harder. See https://arxiv.org/abs/2002.04745 for more
        information.

        Args:
            ...mostly obvious...
            expansion_factor: This determines the number of output channels of the first projection
                              in the two layer MLP. By default this is 4 times the dimensionality
                              of the internal state of the attention layers since this is what GPT
                              uses. My speculation is that the motivation for this is that making
                              the dimensionality higher before applying ReLU prevents information
                              loss. This mechanism is explained in a lot of detail in the paper
                              introducing MobileNetV2: https://arxiv.org/abs/1801.04381.
        """
        super().__init__()
        self._attention_layer = CausalFlashSelfAttention(in_channels=in_channels, n_heads=n_heads, dropout_p=dropout_p)
        self._linear_1 = nn.Linear(in_channels, in_channels*expansion_factor, bias=True)
        self._linear_2 = nn.Linear(in_channels*expansion_factor, in_channels, bias=True)

        def _do(x: torch.Tensor):
            """Dropout."""
            return torch.nn.functional.dropout(x, p=dropout_p, training=self.training)

    def forward(self, x):
        """Forward pass."""
        x = x + self._attention_layer(_ln(x))

        # TODO Does it matter to apply GeLU here instead of ReLU like in GPT papers?
        # I expect ReLU to be faster, but should test it explicitly.
        x = x + self._do(self._linear_2(torch.nn.functional.relu(self._linear_1(_ln(x)))))
        return x


def _init_weights(module: nn.Module):
    """Initialize the weights in a transformer model.

    This is taken from the nanoGPT repo https://github.com/karpathy/nanoGPT. I should investigate
    what the exact motivation for this is since it does not seem to be explained.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerModel(nn.Module):

    def __init__(self, vocab_size: int, context_window: int, n_layers: int, n_heads: int, dim: float, expansion_factor: int, dropout_p: float):
        """Create the full transformer model."""
        super().__init__()

        # First embed all the tokens into vectors.
        self._embed = torch.nn.Embedding(vocab_size, dim)

        # We also need a positional encoding since the attention operations in a tranmsformer are
        # symmetric, and without this the model will not be able to use positional information,
        # which is highly important in language modeling.
        # Since our maximum input size is 'context_window', we need one embedded vector for each
        # possible position in the input.
        self._positional_encoder = torch.nn.Embedding(context_window, dim)

        # Stack a number of transformer blocks.
        self._transformer_blocks = nn.Sequential(
            *[TransformerBlock(in_channels=dim, n_heads=n_heads, expansion_factor=expansion_factor, dropout_p=dropout_p) for _ in range(n_layers)]
        )

        # The output needs to be transformed back to token space for the final classification.
        self._head = nn.Linear(dim, vocab_size, bias=False)

        # The initial embedding and the final output should have the same weights since a token is
        # predicted for each position and this embedding maps between our vector and token spaces,
        # so there is no need to learn a separate transformation.
        self._embed.weight = self._head.weight
        self._context_window = context_window

        # Apply the mystery weight initialization to all layers.
        self.apply(_init_weights)

        # The following weight initialization is explained in the GPT 2 paper
        # https://paperswithcode.com/paper/language-models-are-unsupervised-multitask) in which
        # they write "We scale the weights of residual layers at initialization by a factor of
        # 1/sqrt(N) where N is the number of residual layers."
        for pn, p in self.named_parameters():
            if pn.endswith('_out_transf.weight') or pn.endswith("_linear_2.weight"):
                # Factor 2 here because each transformer block has one attention and one feedforward
                # block, while n_layers is the number of transformer blocks.
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))


    # TODO Do we need this?
    #def context_window(self):
    #    return self._context_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        tokens = self._embed(x)
        pos = self._positional_encoder(torch.arange(x.shape[1], device=tokens.device).unsqueeze(0))
        x = self._transformer_blocks(tokens + pos)
        return self._head(_ln(x))


class GPT1Model(TransformerModel):
    """GPT 1 model defined in https://paperswithcode.com/paper/improving-language-understanding-by.
    
    The original implementation of GPT 1 probably does not have the layernorms in the places where
    we do, but the current placement is better if you want the model to converge.
    """
    def __init__(self, vocab_size: int, context_window: int):
        super().__init__(vocab_size=vocab_size, context_window=context_window, n_layers=12, n_heads=12, dim=768, dropout_p=0.1)


class GPT2SModel(GPT1Model):
    """GPT 2 Small model defined in https://paperswithcode.com/paper/language-models-are-unsupervised-multitask.

    This is equivalent to GPT 1, except for the layernorm placement, which we anyway changed for GPT 1.
    """

class GPT2MModel(TransformerModel):
    """GPT 2 Medium model.
    
    I made a guess about the number of heads that it scales in the same way as the internal
    dimensionality of the transformer. This is never clearly defined in the papers.
    """
    def __init__(self, vocab_size, context_window):
        super().__init__(vocab_size=vocab_size, context_window=context_window, n_layers=24, n_heads=16, dim=1024, dropout_p=0.1)



class GPT2LModel(TransformerModel):
    """GPT 2 Large model."""
    def __init__(self, vocab_size, context_window):
        super().__init__(vocab_size=vocab_size, context_window=context_window, n_layers=36, n_heads=20, dim=1280, dropout_p=0.1)


class GPT2Model(TransformerModel):
    """GPT 2 XL model, also known as GPT 2."""
    def __init__(self, vocab_size, context_window):
        super().__init__(vocab_size=vocab_size, context_window=context_window, n_layers=48, n_heads=25, dim=1600, dropout_p=0.1)
