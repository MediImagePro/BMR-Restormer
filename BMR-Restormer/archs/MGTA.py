import torch
import torch.nn as nn
from einops import rearrange


class MGTA(nn.Module):
    """Multi-Group Transposed Attention."""

    def __init__(self, dim, num_heads, bias, num_groups=2):
        super(MGTA, self).__init__()
        self.num_heads = num_heads
        self.num_groups = max(1, min(num_groups, dim))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _split_sizes(self, total):
        base = total // self.num_groups
        rem = total % self.num_groups
        return [base + (1 if i < rem else 0) for i in range(self.num_groups)]

    def forward(self, x):
        _, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        split_sizes = self._split_sizes(c)
        q_groups = torch.split(q, split_sizes, dim=1)
        k_groups = torch.split(k, split_sizes, dim=1)
        v_groups = torch.split(v, split_sizes, dim=1)

        out_groups = []
        for qg, kg, vg in zip(q_groups, k_groups, v_groups):
            cg = qg.shape[1]
            use_heads = self.num_heads if (cg % self.num_heads == 0) else 1

            qg = rearrange(qg, "b (head c) h w -> b head c (h w)", head=use_heads)
            kg = rearrange(kg, "b (head c) h w -> b head c (h w)", head=use_heads)
            vg = rearrange(vg, "b (head c) h w -> b head c (h w)", head=use_heads)

            qg = torch.nn.functional.normalize(qg, dim=-1)
            kg = torch.nn.functional.normalize(kg, dim=-1)

            temp = self.temperature if use_heads == self.num_heads else self.temperature[:1]
            attn = (qg @ kg.transpose(-2, -1)) * temp
            attn = attn.softmax(dim=-1)
            outg = attn @ vg

            outg = rearrange(outg, "b head c (h w) -> b (head c) h w", head=use_heads, h=h, w=w)
            out_groups.append(outg)

        out = torch.cat(out_groups, dim=1)
        return self.project_out(out)
