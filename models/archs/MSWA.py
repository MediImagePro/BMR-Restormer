import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MWSA(nn.Module):
    """Masked Window Self-Attention.

    File name follows user's request: MSWA.py
    """

    def __init__(self, dim, num_heads, bias, window_size=8):
        super(MWSA, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self._mask_cache = {}

    def _get_even_coord_mask(self, ws, device):
        key = (ws, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        yy, xx = torch.meshgrid(
            torch.arange(ws, device=device),
            torch.arange(ws, device=device),
            indexing="ij",
        )
        valid = ((yy % 2 == 0) & (xx % 2 == 0)).reshape(-1)
        mask = torch.full((ws * ws, ws * ws), float("-inf"), device=device)
        mask[:, valid] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._mask_cache[key] = mask
        return mask

    def _window_partition(self, x, ws):
        b, c, h, w = x.shape
        x = rearrange(x, "b c (nh ws1) (nw ws2) -> (b nh nw) c ws1 ws2", ws1=ws, ws2=ws)
        return x, b, h, w

    def _window_reverse(self, x, b, h, w, ws):
        nh = h // ws
        nw = w // ws
        return rearrange(
            x,
            "(b nh nw) c ws1 ws2 -> b c (nh ws1) (nw ws2)",
            b=b,
            nh=nh,
            nw=nw,
            ws1=ws,
            ws2=ws,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        ws = self.window_size

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q, b0, h0, w0 = self._window_partition(q, ws)
        k, _, _, _ = self._window_partition(k, ws)
        v, _, _, _ = self._window_partition(v, ws)

        q = rearrange(q, "bn (head c) ws1 ws2 -> bn head (ws1 ws2) c", head=self.num_heads)
        k = rearrange(k, "bn (head c) ws1 ws2 -> bn head (ws1 ws2) c", head=self.num_heads)
        v = rearrange(v, "bn (head c) ws1 ws2 -> bn head (ws1 ws2) c", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn + self._get_even_coord_mask(ws, x.device)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(
            out,
            "bn head (ws1 ws2) c -> bn (head c) ws1 ws2",
            head=self.num_heads,
            ws1=ws,
            ws2=ws,
        )
        out = self._window_reverse(out, b0, h0, w0, ws)

        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        return self.project_out(out)
