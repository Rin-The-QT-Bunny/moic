import torch
from torch import nn
from torch.nn import init

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )
            #print(updates.shape,slots_prev.shape)
            slots= updates + slots_prev
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

class Transformer(nn.Module):
    """Transformer with multiple blocks."""

    def __init__(
        self,
        num_heads: int,
        qkv_size: int,
        mlp_size: int,
        num_layers: int,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        self.qkv_size = qkv_size

        self.transformer_blocks = nn.Sequential(
            *[
                nn.TransformerDecoderLayer(
                    d_model=qkv_size,
                    nhead=num_heads,
                    dim_feedforward=mlp_size,
                    norm_first=pre_norm,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        queries,
        inputs,
        padding_mask
    ):
        x = self.transformer_blocks(queries, inputs, padding_mask)
        return x

import torch
import torchvision
from torch import nn

class double_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(double_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class unet_downsample_block(nn.Module):
    def __init__(self, inchannel, outchannel, downsample):
        super(unet_downsample_block, self).__init__()
        self.conv = double_conv(inchannel, outchannel)
        if downsample:
            self.output_layer = nn.MaxPool2d(2)
        self.downsample = downsample

    def forward(self, x):
        skip = self.conv(x)
        if self.downsample:
            y = self.output_layer(skip)
        else:
            y = skip
        return skip, y

class unet_upsample_block(nn.Module):
    def __init__(self, inchannel, outchannel, upsample):
        super(unet_upsample_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.upsample = upsample

    def forward(self, x, skip):
        x = torch.cat((skip, x), 1)
        x = self.double_conv(x)
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2)
        y = x
        return y


class attention_net(nn.Module):
    def __init__(self, slots):
        super(attention_net, self).__init__()
        # define contracting path
        self.down_block1 = unet_downsample_block(4, 64, True)
        self.down_block2 = unet_downsample_block(64, 128, True)
        self.down_block3 = unet_downsample_block(128, 256, True)
        self.down_block4 = unet_downsample_block(256, 512, True)
        # self.down_block5 = unet_downsample_block(512, 512, False)
        self.down_block5 = unet_downsample_block(512, 1024, True)
        self.down_block6 = unet_downsample_block(1024, 1024, False)

        self.linear1 = nn.Linear(4 * 4 * 1024, 128)
        # self.linear1 = nn.Linear(8 * 8 * 512, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 4 * 4 * 1024)

        # define expansive path
        self.up_block1 = unet_upsample_block(1024, 1024, True)
        self.up_block2 = unet_upsample_block(1024, 512, True)
        self.up_block3 = unet_upsample_block(512, 256, True)
        self.up_block4 = unet_upsample_block(256, 128, True)
        self.up_block5 = unet_upsample_block(128, 64, True)
        self.up_block6 = unet_upsample_block(64, 64, False)

        self.layer = nn.Conv2d(64, 2, 1)
        self.relu = nn.ReLU()
        self.slots = slots

    def forward(self, x):
        bs, channel0, w0, h0 = x.shape
        logsk = torch.zeros([bs, 1, w0, h0]).to(x.device).requires_grad_(False)
        history_logsk = logsk
        # Calculate log(mask)
        for i in range(self.slots):
            x_in = torch.cat((x, logsk), 1)
            skip1, x1 = self.down_block1(x_in)
            skip2, x2 = self.down_block2(x1)
            skip3, x3 = self.down_block3(x2)
            skip4, x4 = self.down_block4(x3)
            skip5, x5 = self.down_block5(x4)
            skip6, x6 = self.down_block6(x5)
            bs, channel1, w, h = x6.shape

            h0 = x6.view([bs, channel1 * w * h])
            h1 = self.linear1(h0)
            h2 = self.linear2(h1)
            h3 = self.linear3(h2)
            h3 = self.relu(h3)

            y0 = h3.view([bs, channel1, w, h])

            y1 = self.up_block1(y0, skip6)
            y2 = self.up_block2(y1, skip5)
            y3 = self.up_block3(y2, skip4)
            y4 = self.up_block4(y3, skip3)
            y5 = self.up_block5(y4, skip2)
            y6 = self.up_block6(y5, skip1)

            y = self.layer(y6)
            # y has 2 channel for alpha and 1-alpha respectively, use softmax to make use they sum up to one
            tmp = nn.functional.log_softmax(y, dim=1)
            logalpha = tmp[:, 1, :, :].unsqueeze(1)
            log1_alpha = tmp[:, 0, :, :].unsqueeze(1)
            if i == self.slots - 1:
                logmk = logsk
            else:
                logmk = logsk + logalpha
                logsk = logsk + log1_alpha
            if i == 0:
                ans = logmk
            else:
                ans = torch.cat((ans, logmk), 1)
            history_logsk = torch.cat((history_logsk, logsk), 1)
        return ans, history_logsk