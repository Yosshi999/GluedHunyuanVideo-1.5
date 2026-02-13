from typing import Union, Tuple
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.models.autoencoders.vae import DecoderOutput
import torch


class PatchedAutoencoderKLHunyuanVideo15(AutoencoderKLHunyuanVideo15):
    def __init__(
        self,
        config: dict,
        glued_dims: Tuple[bool, bool, bool] = (True, False, False),
    ):
        super().__init__(**config)
        self.glued_dims = glued_dims
        self.kernel_dims = (
            self.temporal_compression_ratio,
            self.spatial_compression_ratio,
            self.spatial_compression_ratio,
        )
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