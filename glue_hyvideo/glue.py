from typing import Optional, Tuple, MethodType

import torch
import torch.nn as nn
from einops import rearrange

from .modules.modulate_layers import modulate, apply_gate
from .flex_sta import glued_attention_3d


def patched_forward_doublestream(
    self,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec: torch.Tensor,
    freqs_cis: tuple = None,
    text_mask=None,
    attn_param=None,
    is_flash=False,
    block_idx=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patched version of MMDoubleStreamBlock."""
    (
        img_mod1_shift,
        img_mod1_scale,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ) = self.img_mod(vec).chunk(6, dim=-1)

    (
        txt_mod1_shift,
        txt_mod1_scale,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ) = self.txt_mod(vec).chunk(6, dim=-1)

    img_modulated = self.img_norm1(img)
    img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

    img_q = self.img_attn_q(img_modulated)
    img_k = self.img_attn_k(img_modulated)
    img_v = self.img_attn_v(img_modulated)
    img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
    img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
    img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
    img_q = self.img_attn_q_norm(img_q).to(img_v)
    img_k = self.img_attn_k_norm(img_k).to(img_v)

    txt_modulated = self.txt_norm1(txt)
    txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
    txt_q = self.txt_attn_q(txt_modulated)
    txt_k = self.txt_attn_k(txt_modulated)
    txt_v = self.txt_attn_v(txt_modulated)
    txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
    txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
    txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
    txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
    txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

    attn = glued_attention_3d(
        (img_q, txt_q),
        (img_k, txt_k),
        (img_v, txt_v),
        attn_param["rope_dim_list"],
        attn_param["rope_theta"],
        attn_param["thw"],
        attn_param["glued_thw"],
        attn_param["tile_thw"],
        attn_param["kernel_thw"],
    )

    img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

    img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
    img = img + apply_gate(
        self.img_mlp(
            modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
        ),
        gate=img_mod2_gate,
    )

    txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
    txt = txt + apply_gate(
        self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
        gate=txt_mod2_gate,
    )

    return img, txt


def patched_forward_singlestream(
    self,
    x: torch.Tensor,
    vec: torch.Tensor,
    txt_len: int,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    text_mask=None,
    attn_param=None,
    is_flash=False,
) -> torch.Tensor:
    """Patched version of MMSingleStreamBlock."""
    mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
    x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

    q = self.linear1_q(x_mod)
    k = self.linear1_k(x_mod)
    v = self.linear1_v(x_mod)

    q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
    k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
    v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
    
    mlp = self.linear1_mlp(x_mod)

    # Apply QK-Norm if needed.
    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)

    img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
    img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
    img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]

    attn = glued_attention_3d(
        (img_q, txt_q),
        (img_k, txt_k),
        (img_v, txt_v),
        attn_param["rope_dim_list"],
        attn_param["rope_theta"],
        attn_param["thw"],
        attn_param["glued_thw"],
        attn_param["tile_thw"],
        attn_param["kernel_thw"],
    )
    output = self.linear2(attn, self.mlp_act(mlp))
    
    return x + apply_gate(output, gate=mod_gate)

def patch_model(
    model: nn.Module,
    glue_t = False,
    glue_h = False,
    glue_w = False,
    tile_size: Tuple[int, int, int] = (4, 8, 8),
    kernel_dims: Tuple[int, int, int] = (1, 1, 1),
) -> nn.Module:
    if getattr(model, "__already_patched", False):
        return model
    model.__already_patched = True
    assert model.__class__.__name__ == "HunyuanVideo_1_5_DiffusionTransformer"

    # overwrite attn_param
    model.attn_param = {
        "rope_dim_list": model.rope_dim_list,
        "rope_theta": model.rope_theta,
        "thw": None,  # to be set during forward
        "glued_thw": (glue_t, glue_h, glue_w),
        "tile_thw": tile_size,
        "kernel_thw": kernel_dims,
    }

    for block in model.double_blocks:
        block.forward = MethodType(patched_forward_doublestream, block)
    for block in model.single_blocks:
        block.forward = MethodType(patched_forward_singlestream, block)
    
    return model