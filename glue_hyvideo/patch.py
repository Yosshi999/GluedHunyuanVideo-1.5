from typing import MethodType, Union, List, Optional, Dict, Any, Tuple
import torch
from diffusers import HunyuanVideo15Pipeline
from .gsta import GluedSlidingTiledFlexAttnProcessor


def patch_pipeline(
    pipe: HunyuanVideo15Pipeline,
    glued_dims: Tuple[bool, bool, bool] = (True, False, False),
    tile_dims: Tuple[int, int, int] = (4, 8, 8),
    kernel_dims: Optional[Tuple[int, int, int]] = None,
    rope_dim_list: Tuple[int, int, int] = (16, 56, 56),
    rope_theta: int = 256,
) -> None:
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
    """

    assert pipe.__class__.__name__ == "HunyuanVideo15Pipeline", "This patch function only works for HunyuanVideo15Pipeline."
    if getattr(pipe, "__patched_gsta__", False):
        # already patched
        return
    setattr(pipe, "__patched_gsta__", True)
    original_call = pipe.__call__

    def patched_call(
        self: HunyuanVideo15Pipeline,
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
            height, width = self.video_processor.calculate_default_height_width(
                self.default_aspect_ratio[1], self.default_aspect_ratio[0], self.target_size
            )
        canvas_dims =(
            (num_frames - 1) // int(self.vae_scale_factor_temporal) + 1,
            int(height) // int(self.vae_scale_factor),
            int(width) // int(self.vae_scale_factor),
        )
        if kernel_dims is None:
            _kernel_dims = (
                canvas_dims[0] // tile_dims[0],
                canvas_dims[1] // tile_dims[1],
                canvas_dims[2] // tile_dims[2],
            )
        else:
            _kernel_dims = kernel_dims
        processor = GluedSlidingTiledFlexAttnProcessor(canvas_dims, tile_dims, _kernel_dims, glued_dims, rope_dim_list, rope_theta)
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(processor)
        
        return original_call(
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

    pipe.__call__ = MethodType(patched_call, pipe)