from typing import Optional, Tuple
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
import numpy as np

compiled_flex_attention = torch.compile(flex_attention)

def glue_boundary(x: torch.Tensor, left_glue_size: int, right_glue_size: int, dim: int, temporal_rotation: int = 0, temporal_dim: Optional[int] = None) -> torch.Tensor:
    """Glue the boundaries of the input tensor along a specific dimension.

    Args:
        x (torch.Tensor): input tensor.
        left_glue_size (int): size of the left glue.
        right_glue_size (int): size of the right glue.
        dim (int): the dimension along which to glue.
    Returns:
        torch.Tensor: glued tensor.
    """
    dim_size = x.shape[dim]
    left_glue = x.narrow(dim, dim_size - left_glue_size, left_glue_size)
    right_glue = x.narrow(dim, 0, right_glue_size)
    if temporal_rotation != 0:
        assert temporal_dim is not None, "temporal_dim must be specified for temporal rotation"
        right_glue = torch.roll(right_glue, shifts=temporal_rotation, dims=temporal_dim)
        left_glue = torch.roll(left_glue, shifts=-temporal_rotation, dims=temporal_dim)

    x = torch.cat([left_glue, x, right_glue], dim=dim)
    return x

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

class GluedSlidingTiledFlexAttnProcessor(nn.Module):
    def __init__(
        self,
        canvas_dims: Tuple[int, int, int],
        tile_dims: Tuple[int, int, int] = (4, 8, 8),
        kernel_dims: Tuple[int, int, int] = (1, 1, 1),
        glued_dims: Tuple[bool, bool, bool] = (False, False, False),
        rope_dim_list: Tuple[int, int, int] = (16, 56, 56),
        rope_theta: int = 256,
        temporal_rotation: int = 0,
    ):
        """3-D Sliding Tile Attention.

        Args:
            canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
            tile_dims (Tuple[int, int, int]): dimensions of each tile.
            kernel_dims (Tuple[int, int, int]): dimensions of the sliding kernel.
            glued_dims (Tuple[bool, bool, bool]): whether to glue boundaries along each dimension.
            rope_dim_list (Tuple[int, int, int]): RoPE dimensions for image tokens.
            rope_theta (int): RoPE theta parameter.
            temporal_rotation (int): frames to rotate along temporal dimension on gluing right boundary.
        """
        super().__init__()
        self.canvas_dims = canvas_dims
        self.tile_dims = tile_dims
        self.kernel_dims = kernel_dims
        self.glued_dims = glued_dims
        self.temporal_rotation = temporal_rotation
        if temporal_rotation != 0:
            assert glued_dims[0], "temporal gluing must be enabled if temporal_rotation is set"
        self.block_mask: Optional[BlockMask] = None

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        axes_grids_k = []
        axes_grids_q = []
        for i in range(3):
            if glued_dims[i]:
                left_glue_size = tile_dims[i] * (kernel_dims[i] // 2)
                right_glue_size = tile_dims[i] * ((kernel_dims[i]+1) // 2)
                rope_start = -left_glue_size
                rope_end = canvas_dims[i] + right_glue_size
            else:
                rope_start = 0
                rope_end = canvas_dims[i]
            grid = torch.arange(rope_start, rope_end, dtype=torch.float32)
            axes_grids_k.append(grid)
            axes_grids_q.append(torch.arange(0, canvas_dims[i], dtype=torch.float32))
        grid_k = torch.stack(torch.meshgrid(*axes_grids_k, indexing="ij"), dim=0)
        grid_q = torch.stack(torch.meshgrid(*axes_grids_q, indexing="ij"), dim=0)

        freqs_k = []
        freqs_q = []
        for i in range(3):
            freq_k = get_1d_rotary_pos_embed(rope_dim_list[i], grid_k[i].reshape(-1), rope_theta, use_real=True)
            freq_q = get_1d_rotary_pos_embed(rope_dim_list[i], grid_q[i].reshape(-1), rope_theta, use_real=True)
            freqs_k.append(freq_k)
            freqs_q.append(freq_q)
        freqs_cos_k = torch.cat([f[0] for f in freqs_k], dim=1)  # (W * H * T, D / 2)
        freqs_sin_k = torch.cat([f[1] for f in freqs_k], dim=1)  # (W * H * T, D / 2)
        freqs_cos_q = torch.cat([f[0] for f in freqs_q], dim=1)  # (W * H * T, D / 2)
        freqs_sin_q = torch.cat([f[1] for f in freqs_q], dim=1)  # (W * H * T, D / 2)
        self.register_buffer("freqs_cos_k", freqs_cos_k)
        self.register_buffer("freqs_sin_k", freqs_sin_k)
        self.register_buffer("freqs_cos_q", freqs_cos_q)
        self.register_buffer("freqs_sin_q", freqs_sin_q)


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

        # glue boundaries of KV
        # note that glue padding is even to tile size
        query = query.unflatten(1, self.canvas_dims)
        key = key.unflatten(1, self.canvas_dims)
        value = value.unflatten(1, self.canvas_dims)
        glued_offsets = []
        glued_kv_dims = []
        for i in range(3):
            if self.glued_dims[i]:
                # glue along dimension i
                left_glue_size = self.tile_dims[i] * (self.kernel_dims[i] // 2)
                right_glue_size = self.tile_dims[i] * ((self.kernel_dims[i]+1) // 2)
                if i > 0 and self.temporal_rotation != 0:
                    key = glue_boundary(key, left_glue_size, right_glue_size, 1 + i, self.temporal_rotation, temporal_dim=1)
                    value = glue_boundary(value, left_glue_size, right_glue_size, 1 + i, self.temporal_rotation, temporal_dim=1)
                else:
                    key = glue_boundary(key, left_glue_size, right_glue_size, 1 + i)
                    value = glue_boundary(value, left_glue_size, right_glue_size, 1 + i)
                glued_offsets.append(left_glue_size)
                glued_kv_dims.append(self.canvas_dims[i] + left_glue_size + right_glue_size)
            else:
                glued_offsets.append(0)
                glued_kv_dims.append(self.canvas_dims[i])
        query = query.flatten(1, 3)
        key = key.flatten(1, 3)
        value = value.flatten(1, 3)

        # 3. Rotational positional embeddings applied to QK
        query = apply_rotary_emb(query, (self.freqs_cos_q, self.freqs_sin_q), sequence_dim=1)
        key = apply_rotary_emb(key, (self.freqs_cos_k, self.freqs_sin_k), sequence_dim=1)
        
        # 3.5 Tiling
        query = query.unflatten(1, self.canvas_dims)
        key = key.unflatten(1, glued_kv_dims)
        value = value.unflatten(1, glued_kv_dims)
        query_pads = []
        kv_pads = []
        for i in range(3):
            query, qp = pad_to_multiple(query, self.tile_dims[i], i + 1)
            key, kp = pad_to_multiple(key, self.tile_dims[i], i + 1)
            value, _ = pad_to_multiple(value, self.tile_dims[i], i + 1)
            query_pads.append(qp)
            kv_pads.append(kp)
        query = query.flatten(1, 3)
        key = key.flatten(1, 3)
        value = value.flatten(1, 3)
        query_canvas_with_pad = tuple(c + p for c, p in zip(self.canvas_dims, query_pads))
        kv_canvas_with_pad = tuple(c + p for c, p in zip(glued_kv_dims, kv_pads))
        query = tile_mat(query, 1, query_canvas_with_pad, self.tile_dims)
        key = tile_mat(key, 1, kv_canvas_with_pad, self.tile_dims)
        value = tile_mat(value, 1, kv_canvas_with_pad, self.tile_dims)
        image_len_withpad = query.shape[1]
        kvimage_len_withpad = key.shape[1]

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

        seq_len = query.shape[1]
        kvseq_len = key.shape[1]
        if self.block_mask is None:
            canvas_dims = self.canvas_dims
            tile_dims = self.tile_dims
            def mask_mod(b, h, q_idx, kv_idx):
                q0, q1, q2 = untile_indices(q_idx, query_canvas_with_pad, tile_dims)
                k0, k1, k2 = untile_indices(kv_idx, kv_canvas_with_pad, tile_dims)
                tq0, tq1, tq2 = (q0 + glued_offsets[0]) // tile_dims[0], (q1 + glued_offsets[1]) // tile_dims[1], (q2 + glued_offsets[2]) // tile_dims[2]
                tk0, tk1, tk2 = k0 // tile_dims[0], k1 // tile_dims[1], k2 // tile_dims[2]
                return torch.where(
                    q_idx < image_len_withpad,
                    (q0 < canvas_dims[0]) & (q1 < canvas_dims[1]) & (q2 < canvas_dims[2]),
                    q_idx - image_len_withpad < text_attn_len[b],
                ) & torch.where(
                    kv_idx < kvimage_len_withpad,
                    (k0 < glued_kv_dims[0]) & (k1 < glued_kv_dims[1]) & (k2 < glued_kv_dims[2]),
                    kv_idx - kvimage_len_withpad < text_attn_len[b],
                ) & torch.where(
                    (q_idx < image_len_withpad) & (kv_idx < kvimage_len_withpad),
                    (tq0 - tk0 >= -self.kernel_dims[0]//2) & (tq0 - tk0 < (self.kernel_dims[0]+1)//2) &
                    (tq1 - tk1 >= -self.kernel_dims[1]//2) & (tq1 - tk1 < (self.kernel_dims[1]+1)//2) &
                    (tq2 - tk2 >= -self.kernel_dims[2]//2) & (tq2 - tk2 < (self.kernel_dims[2]+1)//2),
                    True,
                )

            self.block_mask = torch.compile(create_block_mask)(  # OK
                mask_mod,
                1,
                1,
                seq_len,
                kvseq_len,
                device=query.device,
                BLOCK_SIZE=block_size,
            )
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
