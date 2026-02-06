"""
Sliding Tile Attention based on PyTorch's FlexAttention.

References:
- FlexAttention:
    - https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention
    - https://pytorch.org/blog/flexattention/
- Sliding Tile Attention:
    - https://www.arxiv.org/abs/2502.04507
- Existing Implementation:
    - https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/hyvideo/models/transformers/modules/ssta_attention.py
        - Note that it depends on [flex-block-attn](https://github.com/Tencent-Hunyuan/flex-block-attn), which supports only SM90.
"""

from functools import lru_cache
from typing import Tuple
from collections.abc import Sequence
import torch
from torch.nn.attention.flex_attention import flex_attention, BlockMask
import numpy as np
from einops import rearrange
# from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed

compiled_flex_attention = torch.compile(flex_attention)

@lru_cache(maxsize=128)
def _create_sta_nd_mask(canvas_dims, padding, tile_dims, kernel_dims) -> torch.Tensor:
    """Create N-D sliding tile attention mask.
    Key-Value pairs can be padded on each dimension.

    Args:
        canvas_dims (Tuple[int, ...]): dimensions of the canvas.
        padding (Tuple[Tuple[int, int], ...]): left and right padding along each dimension.
        tile_dims (Tuple[int, ...]): dimensions of each tile. Recommend to make its prod be power of 2.
        kernel_dims (Tuple[int, ...]): dimensions of the attention kernel.
            Attention window along dimension i will be tile_dims[i] * kernel_dims[i].
    Returns:
        torch.Tensor: the attention mask with shape (prod(canvas_dims)//prod(tile_dims), prod(canvas_dims + padding)//prod(tile_dims))
    """
    q_len = np.prod(canvas_dims)
    kv_canvas_dims = [cdim + pad[0] + pad[1] for cdim, pad in zip(canvas_dims, padding)]
    kv_len = np.prod(kv_canvas_dims)
    
    block_size = np.prod(tile_dims)
    qblock_num = q_len // block_size
    kvblock_num = kv_len // block_size
    block_mask = np.ones((qblock_num, kvblock_num), dtype=bool)

    # flattened indices
    q_indices = np.arange(qblock_num)
    kv_indices = np.arange(kvblock_num)
    q_grid, kv_grid = np.meshgrid(q_indices, kv_indices, indexing='ij')

    q_stride, kv_stride = 1, 1
    for qdim, kvdim, tdim, kdim, pad in zip(reversed(canvas_dims), reversed(kv_canvas_dims), reversed(tile_dims), reversed(kernel_dims), reversed(padding)):
        q_block_size = qdim // tdim
        q_block_coords = (q_grid // q_stride) % q_block_size
        padded_q_block_coords = q_block_coords + (pad[0] // tdim)

        kv_block_size = kvdim // tdim
        kv_block_coords = (kv_grid // kv_stride) % kv_block_size
        kernel_center_coords = np.clip(padded_q_block_coords, kdim // 2, kv_block_size - 1 - (kdim // 2))
        relative = np.abs(kv_block_coords - kernel_center_coords)
        block_mask &= (relative <= (kdim // 2))
        q_stride *= q_block_size
        kv_stride *= kv_block_size

    return block_mask

def create_sta_nd_mask(
    canvas_dims: Sequence[int],
    padding: Sequence[Tuple[int, int]],
    tile_dims: Sequence[int],
    kernel_dims: Sequence[int],
) -> torch.Tensor:
    """Create N-D sliding tile attention mask.
    Key-Value pairs can be padded on each dimension.

    Args:
        canvas_dims (Tuple[int, ...]): dimensions of the canvas.
        padding (Tuple[Tuple[int, int], ...]): left and right padding along each dimension.
        tile_dims (Tuple[int, ...]): dimensions of each tile. Recommend to make its prod be power of 2.
        kernel_dims (Tuple[int, ...]): dimensions of the attention kernel.
            Attention window along dimension i will be tile_dims[i] * kernel_dims[i].
    Returns:
        torch.Tensor: the attention mask with shape (prod(canvas_dims)//prod(tile_dims), prod(canvas_dims + padding)//prod(tile_dims))
    """
    assert len(canvas_dims) == len(padding) == len(tile_dims) == len(kernel_dims)
    rank = len(canvas_dims)
    assert all(canvas_dims[i] % tile_dims[i] == 0 for i in range(rank))
    assert all(padding[i][0] % tile_dims[i] == 0 and padding[i][1] % tile_dims[i] == 0 for i in range(rank))
    assert all(kernel_dims[i] % 2 == 1 for i in range(rank))
    assert all(canvas_dims[i] % (tile_dims[i] * kernel_dims[i]) == 0 for i in range(rank))
    return _create_sta_nd_mask(
        canvas_dims=tuple(canvas_dims),
        padding=tuple([tuple(p) for p in padding]),
        tile_dims=tuple(tile_dims),
        kernel_dims=tuple(kernel_dims),
    )

def tile_mat(tokens: torch.Tensor, canvas_dims: Tuple[int, int, int], tile_dims: Tuple[int, int, int]) -> torch.Tensor:
    """Rearrange tokens into tiles.

    Args:
        tokens (torch.Tensor): input tensor of shape (B, H, L, E).
        canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
        tile_dims (Tuple[int, int, int]): dimensions of each tile.
    Returns:
        torch.Tensor: tiled tensor of shape (B, H, L, E).
    """
    B, H, L, E = tokens.shape
    C1, C2, C3 = canvas_dims
    T1, T2, T3 = tile_dims
    assert C1 * C2 * C3 == L, "canvas_dims do not match token length"
    assert C1 % T1 == 0 and C2 % T2 == 0 and C3 % T3 == 0, "canvas_dims must be divisible by tile_dims"
    D1, D2, D3 = C1 // T1, C2 // T2, C3 // T3
    tokens = rearrange(
        tokens,
        "B H (D1 T1 D2 T2 D3 T3) E -> B H (D1 D2 D3 T1 T2 T3) E",
        D1=D1, D2=D2, D3=D3,
        T1=T1, T2=T2, T3=T3,
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

@torch.compiler.disable
def glued_attention_3d(
    query_img_txt: Tuple[torch.Tensor, torch.Tensor],
    key_img_txt: Tuple[torch.Tensor, torch.Tensor],
    value_img_txt: Tuple[torch.Tensor, torch.Tensor],
    rope_dim_list: Tuple[int, int, int],
    rope_theta: int,
    canvas_dims: Tuple[int, int, int],
    glued_dims: Tuple[bool, bool, bool] = (False, False, False),
    tile_dims: Tuple[int, int, int] = (4, 8, 8),
    kernel_dims: Tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """3-D Sliding Tile Attention with glued boundaries.
    Tokens are aligned (B, L, H, E)

    Args:
        query_img_txt (Tuple[torch.Tensor, torch.Tensor]): query tensors for image and text.
        key_img_txt (Tuple[torch.Tensor, torch.Tensor]): key tensors for image and text.
        value_img_txt (Tuple[torch.Tensor, torch.Tensor]): value tensors for image and text.
        rope_dim_list (Tuple[int, int, int]): RoPE dimensions for image tokens.
        rope_theta (int): RoPE theta parameter.
        canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
        glued_dims (Tuple[bool, bool, bool]): which dimensions are glued.
        tile_dims (Tuple[int, int, int]): dimensions of each tile.
        kernel_dims (Tuple[int, int, int]): dimensions of the attention kernel.
    Returns:
        torch.Tensor: output tensor of shape (B, L, H, E).
    """
    image_q, text_q = query_img_txt
    image_k, text_k = key_img_txt
    image_v, text_v = value_img_txt
    text_len = text_q.shape[1]

    image_q = rearrange(image_q, "B L H E -> B H L E")
    image_k = rearrange(image_k, "B L H E -> B H L E")
    image_v = rearrange(image_v, "B L H E -> B H L E")
    text_q = rearrange(text_q, "B L H E -> B H L E")
    text_k = rearrange(text_k, "B L H E -> B H L E")
    text_v = rearrange(text_v, "B L H E -> B H L E")
    
    # sanity check
    B, Hq, Lq, E = image_q.shape
    assert Lq == np.prod(canvas_dims), f"Lq ({Lq}) must equal to prod of canvas_dims ({canvas_dims})"
    assert len(canvas_dims) == 3
    assert len(glued_dims) == 3
    assert len(tile_dims) == 3
    assert len(kernel_dims) == 3
    block_size = np.prod(tile_dims)
    rank = len(canvas_dims)

    # process image tokens
    image_q = image_q.unflatten(2, canvas_dims)  # (B, Hq, D1, D2, ..., En)
    image_k = image_k.unflatten(2, canvas_dims)
    image_v = image_v.unflatten(2, canvas_dims)

    # glue boundaries of KV
    # note that glue padding is even to tile size
    glued_canvas_dims = []
    glue_paddings = []
    for i in range(rank):
        dim_size = image_k.shape[2 + i]
        if glued_dims[i]:
            # glue along dimension i
            glue_size = tile_dims[i] * (kernel_dims[i] // 2)
            left_glue = image_k.narrow(2 + i, dim_size - glue_size, glue_size)
            right_glue = image_k.narrow(2 + i, 0, glue_size)
            image_k = torch.cat([left_glue, image_k, right_glue], dim=2 + i)
            left_glue_v = image_v.narrow(2 + i, dim_size - glue_size, glue_size)
            right_glue_v = image_v.narrow(2 + i, 0, glue_size)
            image_v = torch.cat([left_glue_v, image_v, right_glue_v], dim=2 + i)
            glue_paddings.append((glue_size, glue_size))
            glued_canvas_dims.append(dim_size + 2 * glue_size)
        else:
            glue_paddings.append((0, 0))
            glued_canvas_dims.append(dim_size)
    
    # apply RoPE to image tokens
    freqs_cis_q = get_nd_rotary_pos_embed(
        tuple(rope_dim_list),
        tuple(canvas_dims),
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )
    freqs_cis_kv = get_nd_rotary_pos_embed(
        tuple(rope_dim_list),
        tuple([-g for g,_ in glue_paddings]),
        tuple([c + g for c, (_,g) in zip(canvas_dims, glue_paddings)]),
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )
    image_q = apply_rotary_emb(image_q, freqs_cis_q, head_first=True)
    image_k = apply_rotary_emb(image_k, freqs_cis_kv, head_first=True)
    
    # pad image tokens to even tile division
    pad = [0 for _ in range(2 * rank)]  # note that torch.nn.functional.pad uses reverse order
    for i in range(rank):
        if canvas_dims[i] % tile_dims[i] != 0:
            pad[-1 - 2 * i] = (-canvas_dims[i]) % tile_dims[i]
    image_q = torch.nn.functional.pad(image_q, [0, 0] + pad)  # (B, Hq, D1 + p1, D2 + p2, ..., En)
    image_k = torch.nn.functional.pad(image_k, [0, 0] + pad)
    image_v = torch.nn.functional.pad(image_v, [0, 0] + pad)
    full_q_dims = image_q.shape[2 : 2 + rank]
    full_kv_dims = image_k.shape[2 : 2 + rank]
    
    # flatten back
    image_q = image_q.flatten(2, 2 + rank - 1)  # (B, Hq, L_padded, E)
    image_k = image_k.flatten(2, 2 + rank - 1)  # (B, Hkv, S_padded, E)
    image_v = image_v.flatten(2, 2 + rank - 1)  # (B, Hkv, S_padded, E)

    # tiling image tokens
    image_q = tile_mat(image_q, full_q_dims, tile_dims)
    image_k = tile_mat(image_k, full_kv_dims, tile_dims)
    image_v = tile_mat(image_v, full_kv_dims, tile_dims)

    # pad text tokens
    if text_len % block_size != 0:
        pad_len = block_size - (text_len % block_size)
        text_q = torch.nn.functional.pad(text_q, (0, 0, 0, pad_len))
        text_k = torch.nn.functional.pad(text_k, (0, 0, 0, pad_len))
        text_v = torch.nn.functional.pad(text_v, (0, 0, 0, pad_len))
    
    # combine image and text tokens
    if text_len > 0:
        query = torch.cat([image_q, text_q], dim=2)
        key = torch.cat([image_k, text_k], dim=2)
        value = torch.cat([image_v, text_v], dim=2)
    else:
        query = image_q
        key = image_k
        value = image_v
    
    # create STA mask
    image_q_tokens = image_q.shape[2]
    image_kv_tokens = image_k.shape[2]

    full_block_mask = torch.zeros((query.shape[2] // block_size, key.shape[2] // block_size), dtype=bool)
    block_mask = create_sta_nd_mask(full_q_dims, glue_paddings, tile_dims, kernel_dims)
    full_block_mask[:block_mask.shape[0], :block_mask.shape[1]] = block_mask
    full_block_mask[:, block_mask.shape[1]:] = True  # any query can attend to all text tokens
    full_block_mask[block_mask.shape[0]:, :] = True  # text tokens can attend to all image tokens without glue

    def mask_mod(b, h, q_idx, kv_idx):
        qt, qy, qx = untile_indices(q_idx, full_q_dims, tile_dims)
        kt, ky, kx = untile_indices(kv_idx, full_kv_dims, tile_dims)
        return torch.where(
            q_idx < image_q_tokens,
            (qt < canvas_dims[0]) & (qy < canvas_dims[1]) & (qx < canvas_dims[2]),
            q_idx - image_q_tokens < text_len,
        ) & torch.where(
            kv_idx < image_kv_tokens,
            (kt < glued_canvas_dims[0]) & (ky < glued_canvas_dims[1]) & (kx < glued_canvas_dims[2]),
            kv_idx - image_kv_tokens < text_len,
        )

    kv_indices = torch.argsort(full_block_mask, dim=1, descending=True)
    kv_num_blocks = torch.sum(full_block_mask, dim=1)
    flex_block_mask = BlockMask(kv_num_blocks, kv_indices, BLOCK_SIZE=block_size, mask_mod=mask_mod)
    output = compiled_flex_attention(
        query,
        key,
        value,
        block_mask=flex_block_mask,
    )
    return rearrange(output, "B H L E -> B L H E")

from typing import Optional, Tuple
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15RotaryPosEmbed
from torch import nn

class GluedAttnProcessor(nn.Module):
    def __init__(
        self,
        rope_dim_list: Tuple[int, int, int],
        rope_theta: int,
        canvas_dims: Tuple[int, int, int],
        glued_dims: Tuple[bool, bool, bool] = (False, False, False),
        tile_dims: Tuple[int, int, int] = (4, 8, 8),
        kernel_dims: Tuple[int, int, int] = (1, 1, 1),
    ):
        """3-D Sliding Tile Attention with glued boundaries.

        Args:
            rope_dim_list (Tuple[int, int, int]): RoPE dimensions for image tokens.
            rope_theta (int): RoPE theta parameter.
            canvas_dims (Tuple[int, int, int]): dimensions of the canvas.
            glued_dims (Tuple[bool, bool, bool]): which dimensions are glued.
            tile_dims (Tuple[int, int, int]): dimensions of each tile.
            kernel_dims (Tuple[int, int, int]): dimensions of the attention kernel.
        """
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.canvas_dims = canvas_dims
        self.glued_dims = glued_dims
        self.tile_dims = tile_dims
        self.kernel_dims = kernel_dims

        axes_grids_k = []
        axes_grids_q = []
        for i in range(3):
            if glued_dims[i]:
                glue_size = tile_dims[i] * (kernel_dims[i] // 2)
                rope_start = -glue_size
                rope_end = canvas_dims[i] + glue_size
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
    
    def _glue_kv(self, image_k: torch.Tensor, image_v: torch.Tensor):
        """Glue boundaries of KV.
        Note that glue padding is even to tile size.
        # image_k, image_v: (B, L, heads, dim)
        """
        sequence_dim = 1
        # unflatten to THW
        image_k = image_k.unflatten(sequence_dim, self.canvas_dims)
        image_v = image_v.unflatten(sequence_dim, self.canvas_dims)

        glue_paddings = []
        for i in range(len(self.canvas_dims)):
            dim_size = image_k.shape[sequence_dim + i]
            if self.glued_dims[i]:
                # glue along dimension i
                glue_size = self.tile_dims[i] * (self.kernel_dims[i] // 2)
                left_glue = image_k.narrow(sequence_dim + i, dim_size - glue_size, glue_size)
                right_glue = image_k.narrow(sequence_dim + i, 0, glue_size)
                image_k = torch.cat([left_glue, image_k, right_glue], dim=sequence_dim + i)
                left_glue_v = image_v.narrow(sequence_dim + i, dim_size - glue_size, glue_size)
                right_glue_v = image_v.narrow(sequence_dim + i, 0, glue_size)
                image_v = torch.cat([left_glue_v, image_v, right_glue_v], dim=sequence_dim + i)
                glue_paddings.append((glue_size, glue_size))
            else:
                glue_paddings.append((0, 0))
        # flatten back
        image_k = image_k.flatten(sequence_dim, sequence_dim + len(self.canvas_dims) - 1)
        image_v = image_v.flatten(sequence_dim, sequence_dim + len(self.canvas_dims) - 1)
        return image_k, image_v, glue_paddings
    
    def _pad_to_tile(self, image_q, image_k, image_v):
        """Pad image tokens to even tile division."""
        sequence_dim = 1
        rank = len(self.canvas_dims)
        # unflatten to THW
        image_q = image_q.unflatten(sequence_dim, self.canvas_dims)
        image_k = image_k.unflatten(sequence_dim, self.canvas_dims)
        image_v = image_v.unflatten(sequence_dim, self.canvas_dims)

        pad = [0 for _ in range(2 * rank)]  # note that torch.nn.functional.pad uses reverse order
        for i in range(rank):
            if self.canvas_dims[i] % self.tile_dims[i] != 0:
                pad[-1 - 2 * i] = (-self.canvas_dims[i]) % self.tile_dims[i]
        image_q = torch.nn.functional.pad(image_q, [0, 0, 0, 0] + pad)  # (B, D1 + p1, D2 + p2, ..., H, E)
        image_k = torch.nn.functional.pad(image_k, [0, 0, 0, 0] + pad)
        image_v = torch.nn.functional.pad(image_v, [0, 0, 0, 0] + pad)
        new_canvas_dims_q = image_q.shape[sequence_dim : sequence_dim + rank]
        new_canvas_dims_kv = image_k.shape[sequence_dim : sequence_dim + rank]

        # flatten back
        image_q = image_q.flatten(sequence_dim, sequence_dim + rank - 1)
        image_k = image_k.flatten(sequence_dim, sequence_dim + rank - 1)
        image_v = image_v.flatten(sequence_dim, sequence_dim + rank - 1)
        return image_q, image_k, image_v, new_canvas_dims_q, new_canvas_dims_kv
    
    def _pad_to_tile_text(self, text_q, text_k, text_v):
        """Pad text tokens to even tile division."""
        sequence_dim = 2  # (B, H, L, E)
        block_size = np.prod(self.tile_dims)
        text_len = text_q.shape[sequence_dim]
        if text_len % block_size != 0:
            pad_len = block_size - (text_len % block_size)
            text_q = torch.nn.functional.pad(text_q, (0, 0, 0, pad_len))
            text_k = torch.nn.functional.pad(text_k, (0, 0, 0, pad_len))
            text_v = torch.nn.functional.pad(text_v, (0, 0, 0, pad_len))
        else:
            pad_len = 0
        return text_q, text_k, text_v, text_len + pad_len

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

        # 3. Glue KV boundaries for image tokens
        # token shape: (B, L, heads, dim)
        key, value, glue_paddings = self._glue_kv(key, value)

        # 4. Apply Glued RoPE to image tokens
        query = apply_rotary_emb(query, (self.freqs_cos_q, self.freqs_sin_q), sequence_dim=1)
        key = apply_rotary_emb(key, (self.freqs_cos_k, self.freqs_sin_k), sequence_dim=1)

        query, key, value, full_q_dims, full_kv_dims = self._pad_to_tile(query, key, value)

        # 5. Reordering to (B, heads, L, dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # 6. Tiling
        query = tile_mat(query, full_q_dims, self.tile_dims)
        key = tile_mat(key, full_kv_dims, self.tile_dims)
        value = tile_mat(value, full_kv_dims, self.tile_dims)
        assert query.shape[2] == np.prod(full_q_dims)
        assert key.shape[2] == np.prod(full_kv_dims)
        assert value.shape[2] == np.prod(full_kv_dims)

        # Encoder condition QKV projection and normalization
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
            
            # 7. Reordering to (B, heads, L, dim)
            encoder_query = encoder_query.permute(0, 2, 1, 3)
            encoder_key = encoder_key.permute(0, 2, 1, 3)
            encoder_value = encoder_value.permute(0, 2, 1, 3)
            text_len = encoder_query.shape[2]

            # 8. Pad
            encoder_query, encoder_key, encoder_value, _ = self._pad_to_tile_text(encoder_query, encoder_key, encoder_value)

            # 9. Combine image and text tokens
            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)
        else:
            text_len = 0
            
        block_size = np.prod(self.tile_dims)
        assert query.shape[2] % block_size == 0 and key.shape[2] % block_size == 0
        full_block_mask = torch.zeros((query.shape[2] // block_size, key.shape[2] // block_size), dtype=bool)
        block_mask = create_sta_nd_mask(full_q_dims, glue_paddings, self.tile_dims, self.kernel_dims)
        full_block_mask[:block_mask.shape[0], :block_mask.shape[1]] = block_mask
        full_block_mask[:, block_mask.shape[1]:] = True  # any query can attend to all text tokens
        full_block_mask[block_mask.shape[0]:, :] = True  # text tokens can attend to all image tokens without glue

        tile_dims = self.tile_dims
        full_q_len = np.prod(full_q_dims)
        full_kv_len = np.prod(full_kv_dims)
        canvas_dims = self.canvas_dims
        glued_canvas_dims = tuple(
            [c + (pad[0] + pad[1]) for c, pad in zip(canvas_dims, glue_paddings)]
        )
        # TODO: mark dense blocks to optimize mask_mod
        def mask_mod(b, h, q_idx, kv_idx):
            qt, qy, qx = untile_indices(q_idx, full_q_dims, tile_dims)
            kt, ky, kx = untile_indices(kv_idx, full_kv_dims, tile_dims)
            return torch.where(
                q_idx < full_q_len,
                (qt < canvas_dims[0]) & (qy < canvas_dims[1]) & (qx < canvas_dims[2]),
                q_idx - full_q_len < text_len,
            ) & torch.where(
                kv_idx < full_kv_len,
                (kt < glued_canvas_dims[0]) & (ky < glued_canvas_dims[1]) & (kx < glued_canvas_dims[2]),
                kv_idx - full_kv_len < text_len,
            )

        kv_indices = torch.argsort(full_block_mask, dim=1, descending=True)
        kv_num_blocks = torch.sum(full_block_mask, dim=1)
        flex_block_mask = BlockMask(kv_num_blocks[None, None], kv_indices[None, None], BLOCK_SIZE=block_size, mask_mod=mask_mod)
        hidden_states = compiled_flex_attention(
            query,
            key,
            value,
            block_mask=flex_block_mask,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3)  # (B, L, heads * dim)

        # Need Untile. Optimize?
        # 10. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
