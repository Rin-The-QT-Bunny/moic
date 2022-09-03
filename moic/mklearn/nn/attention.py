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