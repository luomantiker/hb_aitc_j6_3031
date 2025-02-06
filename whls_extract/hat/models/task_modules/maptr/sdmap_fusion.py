import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.cuda.amp import autocast

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class MapFusion(nn.Module):
    """Fusion module of map feature and bev feature.

    Args:
        input_dim: The inputs dim.
        embed_dims: Dimension of the embeddings.
        bev_h: Height of the bird's-eye view.
        bev_w: Width of the bird's-eye view.
        bev_down: The downsample module applied to bev features.
        fusion_up: The upsample module applied to the fusion features.
        positional_encoding: Positional Encoding.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dims: int,
        bev_h: int,
        bev_w: int,
        bev_down: nn.Module,
        fusion_up: nn.Module,
        positional_encoding: nn.Module = None,
    ):
        super(MapFusion, self).__init__()
        self.input_proj = nn.Conv2d(input_dim, embed_dims, kernel_size=1)

        mask = torch.zeros([1, bev_h // 2, bev_w // 2], dtype=torch.bool)
        if positional_encoding is not None:
            self.pos = positional_encoding(mask)

        self.quant_pos = QuantStub()
        self.bev_down = bev_down
        self.fusion_up = fusion_up
        self.bev_h = bev_h
        self.bev_w = bev_w

        # conv fusion
        self.conv_fusion = BasicResBlock(
            in_channels=embed_dims * 2,
            out_channels=embed_dims * 2,
            bn_kwargs={},
            stride=1,
        )

    @autocast(enabled=False)
    def forward(self, bev_feat: Tensor, osm_feat: Tensor) -> Tensor:
        bs = bev_feat.shape[0]
        bev_feat = (
            bev_feat.view(bs, self.bev_h, self.bev_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        bev_small = self.bev_down(bev_feat)
        bev_small = self.input_proj(bev_small)

        bev_out = torch.cat([bev_small, osm_feat], dim=1)
        bev_out = self.conv_fusion(bev_out)

        bev_final = self.fusion_up(bev_out)
        bev_final = (
            bev_final.view(bs, -1, self.bev_h * self.bev_w)
            .permute(0, 2, 1)
            .contiguous()
        )
        return bev_final


@OBJECT_REGISTRY.register
class ConvDown(nn.Module):
    """The basic structure of ConvDown for downsampling the input tensor.

    Args:
        in_dim: The inputs dim of module.
        mid_dim: The middle dim of module.
        out_dim: The outputs dim of module.
        quant_input: Apply quantstub for inputs. Default is True.
    """

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        quant_input: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dim, out_channels=mid_dim, kernel_size=4, stride=2, padding=1
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            mid_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1
        )
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.act3 = nn.ReLU(inplace=True)
        self.quant = QuantStub(scale=1.0 / 128.0) if quant_input else None

    @autocast(enabled=False)
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor = None
    ) -> torch.Tensor:
        if self.quant is not None:
            x = self.quant(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.act3(x)
        return x


@OBJECT_REGISTRY.register
class ConvUp(nn.Module):
    """The basic structure of ConvUp for upsampling the input tensor.

    Args:
        in_dim: The inputs dim of module.
        mid_dim: The middle dim of module.
        out_dim: The outputs dim of module.
    """

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_dim, out_channels=mid_dim, kernel_size=4, stride=2, padding=1
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            mid_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1
        )
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.act3 = nn.ReLU(inplace=True)

    @autocast(enabled=False)
    def forward(
        self, x: torch.Tensor, skip: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.act3(x)
        return x
