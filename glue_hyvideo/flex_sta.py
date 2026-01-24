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

def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    canvas_dims: Sequence[int],
    glued_dims: Sequence[bool],
    tile_dims: Sequence[int] = (4, 8, 8),
    kernel_dims: Sequence[int] = (1, 1, 1),
    text_len: int = 0,
) -> torch.Tensor:
    """N-D Sliding Tile Attention with glued boundaries.

    Args:
        query (torch.Tensor): query tensor of shape (B, Hq, L, E).
        key (torch.Tensor): key tensor of shape (B, Hkv, S, E).
        value (torch.Tensor): value tensor of shape (B, Hkv, S, E).
        canvas_dims (Sequence[int]): dimensions of the canvas.
        glued_dims (Sequence[bool]): which dimensions are glued.
        tile_dims (Sequence[int]): dimensions of each tile.
        kernel_dims (Sequence[int]): dimensions of the attention kernel.
        text_len (int): length of the text. It is assumed that text tokens are at the end.
    Returns:
        torch.Tensor: output tensor of shape (B, Hq, L, E).
    """
    if text_len > 0:
        image_q = query[:, :, :-text_len, :]
        image_k =   key[:, :, :-text_len, :]
        image_v = value[:, :, :-text_len, :]

        text_q = query[:, :, -text_len:, :]
        text_k =   key[:, :, -text_len:, :]
        text_v = value[:, :, -text_len:, :]
    else:
        image_q = query
        image_k = key
        image_v = value
    B, Hq, Lq, E = image_q.shape
    assert Lq == np.prod(canvas_dims), f"Lq ({Lq}) must equal to prod of canvas_dims ({canvas_dims})"
    block_size = np.prod(tile_dims)
    rank = len(canvas_dims)

    # padding for even tile division
    # dense_qblock_nums = [canvas_dims[i] // tile_dims[i] for i in range(rank)]
    even_canvas_dims = [canvas_dims[i] + ((-canvas_dims[i]) % tile_dims[i]) for i in range(rank)]
    image_q = image_q.unflatten(2, canvas_dims)  # (B, Hq, D1, D2, ..., En)
    image_k = image_k.unflatten(2, canvas_dims)
    image_v = image_v.unflatten(2, canvas_dims)

    pad = [0 for _ in range(2 * rank)]  # note that torch.nn.functional.pad uses reverse order
    for i in range(rank):
        if canvas_dims[i] % tile_dims[i] != 0:
            pad[-1 - 2 * i] = (-canvas_dims[i]) % tile_dims[i]
    image_q = torch.nn.functional.pad(image_q, [0, 0] + pad)  # (B, Hq, D1 + p1, D2 + p2, ..., En)
    image_k = torch.nn.functional.pad(image_k, [0, 0] + pad)
    image_v = torch.nn.functional.pad(image_v, [0, 0] + pad)

    # glue boundaries of KV
    glue_paddings = []
    for i in range(rank):
        if glued_dims[i]:
            # glue along dimension i
            dim_size = image_k.shape[2 + i]
            assert dim_size % tile_dims[i] == 0
            glue_size = tile_dims[i] * (kernel_dims[i] // 2)
            left_glue = image_k.narrow(2 + i, dim_size - glue_size, glue_size)
            right_glue = image_k.narrow(2 + i, 0, glue_size)
            image_k = torch.cat([left_glue, image_k, right_glue], dim=2 + i)
            left_glue_v = image_v.narrow(2 + i, dim_size - glue_size, glue_size)
            right_glue_v = image_v.narrow(2 + i, 0, glue_size)
            image_v = torch.cat([left_glue_v, image_v, right_glue_v], dim=2 + i)
            glue_paddings.append((glue_size, glue_size))
        else:
            glue_paddings.append((0, 0))
    
    # flatten back
    image_q = image_q.flatten(2, 2 + rank - 1)  # (B, Hq, L_padded, E)
    image_k = image_k.flatten(2, 2 + rank - 1)  # (B, Hkv, S_padded, E)
    image_v = image_v.flatten(2, 2 + rank - 1)  # (B, Hkv, S_padded, E)

    # tiling image tokens
    src_names = []
    qdict = {}
    kvdict = {}
    for i in range(rank):
        src_names.append(f"D{i}")
        src_names.append(f"T{i}")
        qdict[f"D{i}"] = image_q.shape[2 + i] // tile_dims[i]
        qdict[f"T{i}"] = tile_dims[i]
        kvdict[f"D{i}"] = image_k.shape[2 + i] // tile_dims[i]
        kvdict[f"T{i}"] = tile_dims[i]
    target_names = src_names[::2] + src_names[1::2]  # D0, D1, ..., E0, E1, ...
    image_q = rearrange(image_q, f"B H ({' '.join(src_names)}) E -> B H ({' '.join(target_names)}) E", **qdict)
    image_k = rearrange(image_k, f"B H ({' '.join(src_names)}) E -> B H ({' '.join(target_names)}) E", **kvdict)
    image_v = rearrange(image_v, f"B H ({' '.join(src_names)}) E -> B H ({' '.join(target_names)}) E", **kvdict)

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
    full_block_mask = torch.zeros((query.shape[2] // block_size, key.shape[2] // block_size), dtype=bool)
    block_mask = create_sta_nd_mask(even_canvas_dims, glue_paddings, tile_dims, kernel_dims)
    full_block_mask[:block_mask.shape[0], :block_mask.shape[1]] = block_mask
    full_block_mask[:, block_mask.shape[1]:] = True  # any query can attend to all text tokens
    full_block_mask[block_mask.shape[0]:, :] = True  # text tokens can attend to all image tokens without glue

    kv_indices = torch.argsort(full_block_mask, dim=1, descending=True)
    kv_num_blocks = torch.sum(full_block_mask, dim=1)
    flex_block_mask = BlockMask(kv_num_blocks, kv_indices, BLOCK_SIZE=block_size)
    output = flex_attention(
        query,
        key,
        value,
        block_mask=flex_block_mask,
    )
    return output
