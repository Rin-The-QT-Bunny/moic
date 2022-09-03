import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from aluneth.rinlearn.nn.functional_net import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        """UNet map in_channel to out_channel with num_block and base

        Args:
            num_blocks (_type_): _description_
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            channel_base (int, optional): channelBase. Defaults to 64.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class AttentionNet(nn.Module):
    def __init__(self, blocks,base):
        super().__init__()
        self.unet = UNet(num_blocks= blocks,
                         in_channels=4 + 1,
                         out_channels=2,
                         channel_base= base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height,inchannle = 4,latent_dim = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inchannle, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        self.mlp = FCBlock(200,3,64 * width * height,latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)

        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, width, height,in_channels,out_channels = 4):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)
        #self.T = FCBlock(128,3,in_channels,in_channels)

    def forward(self, z):
        #z = self.T(z)
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = 10*self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)

        result = self.convs(inp)
        return result

    
class ResidualBlock(nn.Module):
    def __init__(self,inchanel,outchanel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchanel,outchanel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchanel,outchanel,3,1,1,bias=False),
            nn.BatchNorm2d(outchanel)
        )
        self.right=shortcut
    def forward(self, x):
        out=self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
    
class ResNet34(nn.Module):

    def __init__(self,num_classes=1000):
        super(ResNet34,self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self.__make_layer__(64,128,3)
        self.layer2 = self.__make_layer__(128,256,4,stride=2)
        self.layer3 = self.__make_layer__(256,512,6,stride=2)
        self.layer4 = self.__make_layer__(512,512,3,stride=2)
        self.fc = nn.Linear(512,num_classes)

    def __make_layer__(self,inchannel,outchannel,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(inchannel,outchannel))
        return  nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        return self.fc(x)

class DETR(nn.Module):
    def __init__(self,num_classes = 10,hidden_dim = 256,
    nheads = 8,num_encoder_layers = 6,num_decoder_layers = 6):
        super().__init__()
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048,hidden_dim,1)
        self.transformer = nn.Transformer(hidden_dim,nheads,
        num_encoder_layers,num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim,num_classes+1)
        self.linear_bbox = nn.Linear(hidden_dim,4)

        # create positional encodings
        self.query_pos = nn.Parameter(torch.randn(100,hidden_dim))
        self.row_embed = nn.Parameter(torch.randn(50,hidden_dim//2))
        self.col_embed = nn.Parameter(torch.randn(50,hidden_dim//2))
        
    def forward(self,inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        h = self.conv(x)
        H,W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1,W,1),
        ],dim=-1).flatten(0,1).unsqueeze(1)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2,0,1),
            self.query_pos.unsqueeze(1)).transpose(0,1)
        return {"pred_logits":self.linear_class(h),"pred_boxes":self.linear_bbox(h).sigmoid}

from aluneth.rinlearn.nn.utils import *

class SpatialBroadcastMaskDecoder(nn.Module):
        def __init__(self,resolution,backbone,pos_emb):
            super().__init__()
            self.resolution = resolution
            self.backbone = backbone
            self.pos_emb = pos_emb

            #sub modules
            self.mask_pred = nn.Linear(self.backbone.features[-1],1)

        def forward(self,slots):
            batch_size,n_slots,n_features = slots.shape

            # Fold slot dim into batch dim
            x = slots.reshape(shape = (batch_size * n_slots, n_features))
            x = spatial_broadcast(x,self.resolution)
            x = self.pos_emb(x)

            # bb_features.shape = (batch_size * n_slots,h,w,c)
            bb_features = self.backbone(x,channel_last=True)
            spatial_dims = bb_features.shape[-3:-1]

            alpha_logits = self.mask_pred(
                bb_features.reshape(shape = (-1,bb_features.shape[-1]))
            )
            alpha_logits = alpha_logits.reshape(
                shape=(batch_size,n_slots,*spatial_dims,-1)
            )
            return bb_features,alpha_logits

class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)

        return inputs + emb_proj * 5


class MaskDecoder(nn.Module):
    def __init__(self,feature_dim,out_dim,resolution):
        super().__init__()
        self.height,self.width = resolution
        self.pos_embed = SoftPositionEmbed(4,feature_dim,resolution)
        self.pred = FCBlock(132,3,feature_dim,out_dim)
        self.feature_dim = feature_dim

    def forward(self,z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height, self.width)
        z_tiled = self.pos_embed(z_tiled)
        batch_size,feature_dim = z.shape
        z_tiled = z_tiled.permute([0,2,3,1])
        logits = self.pred(z_tiled)
        mask = torch.sigmoid(3*logits)
        mask = mask.permute([0,3,1,2])
        return mask

class CNN(nn.Module):
    """ CNN model with conv and normalization layers """

    def __init__(
        self,
        features,
        kernel_size,
        strides,
        layer_transpose,
        activation_fn= F.relu,
        norm_type = None,
        dim_name = None,  # Name of dim over which to aggregate batch stats
        output_size = None,
        train = True,
    ):
        super().__init__()
        self.num_layers = len(features)
        self.features = features
        self.strides = strides
        self.kernel_size = kernel_size
        self.layer_transpose = layer_transpose
        self.activation_fn = activation_fn
        self.norm_type = norm_type
        self.dim_name = dim_name
        self.output_size = output_size
        self.training = train

        assert self.num_layers >= 1, "Need to have at least one layer."
        assert (
            len(self.kernel_size) == self.num_layers
        ), "len(kernel_size) and len(features) must match."
        assert (
            len(self.strides) == self.num_layers
        ), "len(strides) and len(features) must match."
        assert (
            len(self.layer_transpose) == self.num_layers
        ), "len(layer_transpose) and len(features) must match."

        if self.norm_type:
            assert self.norm_type in {
                "batch",
                "group",
                "instance",
                "layer",
            }, f"{self.norm_type} is not a valid normalization module."

        # Build conv net
        self.conv_layers = []
        for i, lt in enumerate(layer_transpose):
            if not lt:
                conv = nn.Conv2d
            else:
                conv = nn.ConvTranspose2d

            in_channels = 3 if i == 0 else self.features[i - 1]
            layer = conv(
                in_channels,
                self.features[i],
                kernel_size=kernel_size[i],
                stride=strides[i],
                padding="same",
                bias=False if self.norm_type else True,
            )
            self.conv_layers.append(layer)

        self.conv = nn.Sequential(*self.conv_layers)

        # Select norm type
        if self.norm_type == "batch":
            self.norm_module = nn.BatchNorm2d(
                momentum=0.1, track_running_stats=not self.training
            )
        elif self.norm_type == "group" or self.norm_type == "instance":
            self.norm_module = nn.GroupNorm(
                num_channels=self.features[-1], num_groups=32
            )
        elif self.norm_type == "layer":
            self.norm_module = nn.LayerNorm(self.features[-1])

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type:
            x = self.norm_module(x)
        x = self.activation_fn(x)

        return x