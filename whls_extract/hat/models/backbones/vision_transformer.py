import logging
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch.nn.init import trunc_normal_
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vit_module import PatchEmbed, ViTBlock
from hat.models.base_modules.mlp_module import MlpModule2d as Mlp
from hat.registry import OBJECT_REGISTRY

__all__ = ["VisionTransformer"]

_logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class VisionTransformer(nn.Module):
    """Vision Transformer.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Mumber of classes for classification head.
        global_pool: Type of global pooling for final sequence.
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values
             (layer-scale enabled if not None).
        class_token: Use class token.
        fc_norm: Pre head norm after pool (instead of before),
            if None, enabled when global_pool == 'avg'.
        drop_rate: Head dropout rate.
        pos_drop_rate: Position embedding dropout rate.
        proj_drop_rate: Projection dropout in Attention module.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        weight_init: Weight initialization scheme.
        embed_layer: Patch embedding layer.
        norm_layer: Normalization layer.
        act_layer: MLP activation layer.
        block_fn: Transformer block layer.
        mlp_layer: MLP layer.
        set_int16_qconfig: Flag of whether set int16 qconfig.
        include_top: If include top classifier head.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = ViTBlock,
        mlp_layer: Callable = Mlp,
        set_int16_qconfig: bool = True,
        include_top: bool = True,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.cls_quant = QuantStub()
        embed_len = (
            num_patches
            if no_embed_class
            else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * 0.02
        )
        self.pos_quant = QuantStub()
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.quant = QuantStub(scale=1 / 128.0)
        self.dequant = DeQuantStub()

        self.pos_add = FloatFunctional()
        self.pos_cat = FloatFunctional()

        self.include_top = include_top

        if weight_init != "skip":
            self._init_weights(weight_init)
        self.set_int16_qconfig = set_int16_qconfig

    def _init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(init_weights_vit_timm)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token,
            # add then concat
            x = self.pos_add.add(x, self.pos_quant(self.pos_embed))
            if self.cls_token is not None:
                cls_token = self.cls_quant(self.cls_token)
                x = self.pos_cat.cat(
                    (cls_token.expand(x.shape[0], -1, -1), x), dim=1
                )
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                cls_token = self.cls_quant(self.cls_token)
                x = self.pos_cat.cat(
                    (cls_token.expand(x.shape[0], -1, -1), x), dim=1
                )
            x = self.pos_add.add(x, self.pos_quant(self.pos_embed))
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.quant(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        if not self.include_top:
            return x

        x = self.head(x)
        out = self.dequant(x)
        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.include_top:
            self.head.qconfig = qconfig_manager.get_default_qat_out_qconfig()
        if self.set_int16_qconfig:
            int16_module = [
                self.blocks[0].add1,
                self.pos_cat,
                self.pos_add,
                self.blocks[11].add1,
                self.blocks[11].add2,
                self.blocks[10].add2,
                self.blocks[10].add1,
                self.blocks[9].add1,
                self.blocks[9].add2,
                self.blocks[8].add2,
                self.blocks[7].add2,
                self.blocks[8].add1,
                self.blocks[7].add1,
                self.blocks[6].add2,
                self.blocks[0].add2,
                self.blocks[1].add1,
                self.blocks[5].add2,
                self.blocks[6].add1,
                self.blocks[4].add2,
                self.blocks[5].add1,
                self.blocks[1].add2,
            ]
            for module in int16_module:
                module.qconfig = qconfig_manager.get_qconfig(
                    activation_qat_qkwargs={"dtype": qint16},
                    activation_calibration_qkwargs={"dtype": qint16},
                )


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """Vit weight initialization, original timm impl."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


VIT_PARAMS = {
    "vit_tiny_patch16_224": dict(  # noqa [C408]
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3
    ),
    "vit_tiny_patch16_384": dict(  # noqa [C408]
        img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3
    ),
    "vit_small_patch32_224": dict(  # noqa [C408]
        img_size=224, patch_size=32, embed_dim=384, depth=12, num_heads=6
    ),
    "vit_small_patch32_384": dict(  # noqa [C408]
        img_size=384, patch_size=32, embed_dim=384, depth=12, num_heads=6
    ),
    "vit_small_patch16_224": dict(  # noqa [C408]
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6
    ),
    "vit_small_patch16_384": dict(  # noqa [C408]
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6
    ),
    "vit_small_patch8_224": dict(  # noqa [C408]
        img_size=224, patch_size=8, embed_dim=384, depth=12, num_heads=6
    ),
    "vit_base_patch32_224": dict(  # noqa [C408]
        img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12
    ),
    "vit_base_patch32_384": dict(  # noqa [C408]
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12
    ),
    "vit_base_patch16_224": dict(  # noqa [C408]
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    ),
    "vit_base_patch16_384": dict(  # noqa [C408]
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12
    ),
    "vit_base_patch8_224": dict(  # noqa [C408]
        img_size=224, patch_size=8, embed_dim=768, depth=12, num_heads=12
    ),
    "vit_large_patch32_224": dict(  # noqa [C408]
        img_size=224, patch_size=32, embed_dim=1024, depth=24, num_heads=16
    ),
    "vit_large_patch32_384": dict(  # noqa [C408]
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16
    ),
    "vit_large_patch16_224": dict(  # noqa [C408]
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16
    ),
    "vit_large_patch16_384": dict(  # noqa [C408]
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16
    ),
    "vit_large_patch14_224": dict(  # noqa [C408]
        img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16
    ),
}


@OBJECT_REGISTRY.register
class ViT(VisionTransformer):
    def __init__(self, model_type: str, **kwargs):
        variant_params = VIT_PARAMS[model_type]
        super().__init__(**variant_params, **kwargs)
