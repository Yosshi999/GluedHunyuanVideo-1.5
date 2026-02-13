from typing import Union, Tuple
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
import torch


class PatchedAutoencoderKLHunyuanVideo15(AutoencoderKLHunyuanVideo15):
    @classmethod
    def from_original_vae(
        cls,
        vae: AutoencoderKLHunyuanVideo15,
        glued_dims: Tuple[bool, bool, bool] = (True, False, False),
    ):
        vae = cls.from_config(vae.config, dtype=vae.dtype)
        vae.load_state_dict(vae.state_dict())
        vae.glued_dims = glued_dims
        vae.kernel_dims = (
            vae.temporal_compression_ratio,
            vae.spatial_compression_ratio,
            vae.spatial_compression_ratio,
        )
        return vae

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        # z: (B, C, T, H, W)
        for i in range(3):
            if self.glued_dims[i]:
                dim_size = z.shape[2+i]
                glue_size = 1
                left_glue = z.narrow(2+i, dim_size - glue_size, glue_size)
                right_glue = z.narrow(2+i, 0, glue_size)
                z = torch.cat([left_glue, z, right_glue], dim=2+i)
        decoded = super().decode(z, return_dict=False)[0]
        for i in range(3):
            if self.glued_dims[i]:
                decoded = decoded.narrow(2+i, self.kernel_dims[i], decoded.shape[2+i] - 2 * self.kernel_dims[i])
        if return_dict:
            return DecoderOutput(sample=decoded)
        else:
            return (decoded,)