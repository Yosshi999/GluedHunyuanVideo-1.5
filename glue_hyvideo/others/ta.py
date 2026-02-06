from typing import Optional, Tuple
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
import numpy as np

compiled_flex_attention = torch.compile(flex_attention)

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int) -> Tuple[torch.Tensor, int]:
    """Pad the input tensor along a specific dimension to make its size a multiple of a given number.

    Args:
        x (torch.Tensor): input tensor.
        multiple (int): the multiple to pad to.
        dim (int): the dimension along which to pad.
    Returns:
        torch.Tensor: padded tensor.
        int: amount of padding added.
    """
    size = x.shape[dim]
    pad_size = (multiple - size % multiple) % multiple
    if pad_size > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=dim)
    return x, pad_size

def tile_mat(tokens: torch.Tensor, sequence_dim: int, canvas_dims: Tuple[int, int, int], tile_dims: Tuple[int, int, int]) -> torch.Tensor:
    """Rearrange 3D tokens into tiles.

    Args:
        tokens (torch.Tensor): input tensor of shape (B, H, L, E).
        sequence_dim (int): target dimension to reorder.
        canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
        tile_dims (Tuple[int, int, int]): dimensions of each tile.
    Returns:
        torch.Tensor: tiled tensor of shape (B, H, L, E).
    """
    L = tokens.shape[sequence_dim]
    C1, C2, C3 = canvas_dims
    T1, T2, T3 = tile_dims
    assert C1 * C2 * C3 == L, "canvas_dims do not match token length"
    assert C1 % T1 == 0 and C2 % T2 == 0 and C3 % T3 == 0, "canvas_dims must be divisible by tile_dims"
    D1, D2, D3 = C1 // T1, C2 // T2, C3 // T3
    # rearrange (D1 T1 D2 T2 D3 T3) -> (D1 D2 D3 T1 T2 T3) along sequence_dim
    perm = list(range(sequence_dim)) + [
        sequence_dim, sequence_dim + 2, sequence_dim + 4,
        sequence_dim + 1, sequence_dim + 3, sequence_dim + 5,
    ] + list(range(sequence_dim + 6, tokens.ndim + 5))
    tokens = (tokens
        .unflatten(sequence_dim, (D1, T1, D2, T2, D3, T3))
        .permute(*perm)
        .flatten(sequence_dim, sequence_dim + 5)
    )
    return tokens

def untile_mat(tokens: torch.Tensor, sequence_dim: int, canvas_dims: Tuple[int, int, int], tile_dims: Tuple[int, int, int]) -> torch.Tensor:
    """Rearrange tiled 3D tokens back to original format.

    Args:
        tokens (torch.Tensor): tiled tensor of shape (B, H, L, E).
        sequence_dim (int): target dimension to reorder.
        canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
        tile_dims (Tuple[int, int, int]): dimensions of each tile.
    Returns:
        torch.Tensor: untiled tensor of shape (B, H, L, E).
    """
    L = tokens.shape[sequence_dim]
    C1, C2, C3 = canvas_dims
    T1, T2, T3 = tile_dims
    assert C1 * C2 * C3 == L, "canvas_dims do not match token length"
    assert C1 % T1 == 0 and C2 % T2 == 0 and C3 % T3 == 0, "canvas_dims must be divisible by tile_dims"
    D1, D2, D3 = C1 // T1, C2 // T2, C3 // T3
    # rearrange (D1 D2 D3 T1 T2 T3) -> (D1 T1 D2 T2 D3 T3) along sequence_dim
    perm = list(range(sequence_dim)) + [
        sequence_dim, sequence_dim + 3,
        sequence_dim + 1, sequence_dim + 4,
        sequence_dim + 2, sequence_dim + 5,
    ] + list(range(sequence_dim + 6, tokens.ndim + 5))
    tokens = (tokens
        .unflatten(sequence_dim, (D1, D2, D3, T1, T2, T3))
        .permute(*perm)
        .flatten(sequence_dim, sequence_dim + 5)
    )
    return tokens

def untile_indices(
    indices: torch.Tensor,
    canvas_dims: Tuple[int, int, int],
    tile_dims: Tuple[int, int, int],
) -> torch.Tensor:
    """Convert tiled indices back to original coordinates.

    Args:
        indices (torch.Tensor): tiled indices of shape (*).
        canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
        tile_dims (Tuple[int, int, int]): dimensions of each tile.
    Returns:
        torch.Tensor: untiled dim1-index of shape (*).
        torch.Tensor: untiled dim2-index of shape (*).
        torch.Tensor: untiled dim3-index of shape (*).
    """
    C1, C2, C3 = canvas_dims
    T1, T2, T3 = tile_dims
    D1, D2, D3 = C1 // T1, C2 // T2, C3 // T3

    # indices are in the tiled format: (D1 D2 D3 T1 T2 T3)
    # convert back to original format: (D1 T1) (D2 T2) (D3 T3)
    t3 = indices % T3
    t2 = (indices // T3) % T2
    t1 = (indices // (T3 * T2)) % T1
    d3 = (indices // (T3 * T2 * T1)) % D3
    d2 = (indices // (T3 * T2 * T1 * D3)) % D2
    d1 = (indices // (T3 * T2 * T1 * D3 * D2)) % D1
    return (
        d1 * T1 + t1,
        d2 * T2 + t2,
        d3 * T3 + t3,
    )

class TiledFlexAttnProcessor:
    def __init__(
        self,
        canvas_dims: Tuple[int, int, int],
        tile_dims: Tuple[int, int, int] = (4, 8, 8),
    ):
        """3-D Tile Attention.

        Args:
            canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
            tile_dims (Tuple[int, int, int]): dimensions of each tile.
        """
        self.canvas_dims = canvas_dims
        self.tile_dims = tile_dims
        self.block_mask: Optional[BlockMask] = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        
        # 3.5 Tiling
        image_len = query.shape[1]
        query = query.unflatten(1, self.canvas_dims)
        key = key.unflatten(1, self.canvas_dims)
        value = value.unflatten(1, self.canvas_dims)
        query_pads = []
        for i in range(3):
            query, qp = pad_to_multiple(query, self.tile_dims[i], i + 1)
            key, _ = pad_to_multiple(key, self.tile_dims[i], i + 1)
            value, _ = pad_to_multiple(value, self.tile_dims[i], i + 1)
            query_pads.append(qp)
        query = query.flatten(1, 3)
        key = key.flatten(1, 3)
        value = value.flatten(1, 3)
        query_canvas_with_pad = tuple(c + p for c, p in zip(self.canvas_dims, query_pads))
        query = tile_mat(query, 1, query_canvas_with_pad, self.tile_dims)
        key = tile_mat(key, 1, query_canvas_with_pad, self.tile_dims)
        value = tile_mat(value, 1, query_canvas_with_pad, self.tile_dims)
        image_len_withpad = query.shape[1]

        # 4. Encoder condition QKV projection and normalization
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
            text_len = encoder_hidden_states.shape[1]
            text_attn_len = torch.clamp(attention_mask.bool().sum(1), min=text_len)  # (B,)
        block_size = int(np.prod(self.tile_dims))
        query, _ = pad_to_multiple(query, block_size, 1)
        key, _ = pad_to_multiple(key, block_size, 1)
        value, _ = pad_to_multiple(value, block_size, 1)

        batch_size, seq_len, heads, dim = query.shape
        if self.block_mask is None:
            canvas_dims = self.canvas_dims
            tile_dims = self.tile_dims
            def mask_mod(b, h, q_idx, kv_idx):
                q0, q1, q2 = untile_indices(q_idx, query_canvas_with_pad, tile_dims)
                k0, k1, k2 = untile_indices(kv_idx, query_canvas_with_pad, tile_dims)
                return torch.where(
                    q_idx < image_len_withpad,
                    (q0 < canvas_dims[0]) & (q1 < canvas_dims[1]) & (q2 < canvas_dims[2]),
                    q_idx - image_len_withpad < text_attn_len[b],
                ) & torch.where(
                    kv_idx < image_len_withpad,
                    (k0 < canvas_dims[0]) & (k1 < canvas_dims[1]) & (k2 < canvas_dims[2]),
                    kv_idx - image_len_withpad < text_attn_len[b],
                )

            # self.block_mask = create_block_mask(  # OOM
            #     mask_mod,
            #     batch_size,
            #     heads,
            #     seq_len,
            #     seq_len,
            #     device=query.device,
            #     BLOCK_SIZE=block_size,
            # )
            # self.block_mask = torch.compile(create_block_mask)(  # OK
            #     mask_mod,
            #     1,
            #     1,
            #     seq_len,
            #     seq_len,
            #     device=query.device,
            #     BLOCK_SIZE=block_size,
            # )
            full_block_mask = torch.ones((seq_len // block_size, seq_len // block_size), dtype=bool)
            kv_indices = torch.argsort(full_block_mask, dim=1, descending=True).int()
            kv_num_blocks = torch.sum(full_block_mask, dim=1).int()
            self.block_mask = BlockMask.from_kv_blocks(kv_num_blocks[None,None].to("cuda"), kv_indices[None,None].to("cuda"), BLOCK_SIZE=block_size, mask_mod=mask_mod)
            print(self.block_mask)

        # 5. Attention
        query = query.transpose(1, 2)  # (B, heads, L, dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hidden_states = compiled_flex_attention(
            query,
            key,
            value,
            block_mask=self.block_mask,
        )
        hidden_states = hidden_states.transpose(1, 2)  # (B, L, heads, dim)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)  # (B, L, C)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :image_len_withpad],
                hidden_states[:, image_len_withpad : image_len_withpad + text_len],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = untile_mat(hidden_states, 1, query_canvas_with_pad, self.tile_dims)
        hidden_states = hidden_states.unflatten(1, query_canvas_with_pad)
        hidden_states = hidden_states[:, :self.canvas_dims[0], :self.canvas_dims[1], :self.canvas_dims[2]].flatten(1, 3)
        return hidden_states, encoder_hidden_states