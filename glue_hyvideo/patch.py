from typing import Union, List, Optional, Dict, Any, Tuple
from types import MethodType
import torch
from diffusers import HunyuanVideo15Pipeline
from .gsta import GluedSlidingTiledFlexAttnProcessor

class PatchedHunyuanVideo15Pipeline:
    def __init__(
        self,
        pipe: HunyuanVideo15Pipeline,
        glued_dims: Tuple[bool, bool, bool] = (True, False, False),
        tile_dims: Tuple[int, int, int] = (4, 8, 8),
        kernel_dims: Optional[Tuple[int, int, int]] = None,
        rope_dim_list: Tuple[int, int, int] = (16, 56, 56),
        rope_theta: int = 256,
        temporal_rotation: int = 0,
    ):
        self.pipe = pipe
        self.glued_dims = glued_dims
        self.tile_dims = tile_dims
        self.kernel_dims = kernel_dims
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.temporal_rotation = temporal_rotation

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if height is None and width is None:
            height, width = self.pipe.video_processor.calculate_default_height_width(
                self.pipe.default_aspect_ratio[1], self.pipe.default_aspect_ratio[0], self.pipe.target_size
            )
        canvas_dims =(
            (num_frames - 1) // int(self.pipe.vae_scale_factor_temporal) + 1,
            int(height) // int(self.pipe.vae_scale_factor_spatial),
            int(width) // int(self.pipe.vae_scale_factor_spatial),
        )
        if self.kernel_dims is None:
            self.kernel_dims = (
                canvas_dims[0] // self.tile_dims[0],
                canvas_dims[1] // self.tile_dims[1],
                canvas_dims[2] // self.tile_dims[2],
            )
        processor = GluedSlidingTiledFlexAttnProcessor(canvas_dims, self.tile_dims, self.kernel_dims, self.glued_dims, self.rope_dim_list, self.rope_theta, self.temporal_rotation)
        for block in self.pipe.transformer.transformer_blocks:
            block.attn.set_processor(processor)
        print("called patched pipeline")
        return self.pipe(
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            num_inference_steps,
            sigmas,
            num_videos_per_prompt,
            generator,
            latents,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
            output_type,
            return_dict,
            attention_kwargs,
        )

def patch_pipeline(
    pipe: HunyuanVideo15Pipeline,
    glued_dims: Tuple[bool, bool, bool] = (True, False, False),
    tile_dims: Tuple[int, int, int] = (4, 8, 8),
    kernel_dims: Optional[Tuple[int, int, int]] = None,
    rope_dim_list: Tuple[int, int, int] = (16, 56, 56),
    rope_theta: int = 256,
    temporal_rotation: int = 0,
) -> PatchedHunyuanVideo15Pipeline:
    """Patches HunyuanVideo15Pipeline in-place to use Glued Sliding Tiled Attention (GSTA).
    Args:
        pipe (HunyuanVideo15Pipeline): The pipeline to be patched.
        glued_dims (Tuple[bool, bool, bool]): A tuple indicating which dimensions to apply gluing (temporal, height, width).
        tile_dims (Tuple[int, int, int]): The dimensions of the tiles (temporal, height, width).
            As it gets larger, the computation gets more efficient but the quality may degrade.
        kernel_dims (Optional[Tuple[int, int, int]]): The dimensions of the attention kernel (temporal, height, width).
            If None, it gets as large as possible.
            As it gets smaller, the computation gets more efficient but the quality may degrade.
        rope_dim_list (Tuple[int, int, int]): The dimensions for RoPE (temporal, height, width).
        rope_theta (int): The theta parameter for RoPE.
        temporal_rotation (int): frames to rotate along temporal dimension on gluing right boundary.
    """

    assert pipe.__class__.__name__ == "HunyuanVideo15Pipeline", "This patch function only works for HunyuanVideo15Pipeline."
    if getattr(pipe, "__patched_gsta__", False):
        print("already patched")
        return
    setattr(pipe, "__patched_gsta__", True)
    return PatchedHunyuanVideo15Pipeline(pipe, glued_dims, tile_dims, kernel_dims, rope_dim_list, rope_theta, temporal_rotation)
