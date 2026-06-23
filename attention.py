import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout,
                  num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x, kv_cache=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        new_keys = self.W_key(x)
        new_values = self.W_value(x)

        new_keys = new_keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        new_values = new_values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_keys, cached_values = kv_cache
            keys = torch.cat([cached_keys, new_keys], dim=2)
            values = torch.cat([cached_values, new_values], dim=2)
        else:
            keys = new_keys
            values = new_values

        updated_cache = (keys, values)

        total_len = keys.shape[2]
        attn_scores = queries @ keys.transpose(2, 3)

        # only mask the rows corresponding to the new query tokens
        mask_bool = self.mask.bool()[:num_tokens, :total_len]
        # shift mask for cached positions
        if kv_cache is not None:
            cache_len = total_len - num_tokens
            mask_bool = self.mask.bool()[cache_len:cache_len + num_tokens, :total_len]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, updated_cache
