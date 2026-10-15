"""Token-preserving conditional flow matching with a four-block small Flow-DiT."""
import math
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Flow time in [0,1], independent of the 30 Hz robot control clock."""
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('freq', torch.exp(torch.linspace(0, math.log(1000), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        phase = t.float()[:, None] * self.freq[None] * (2 * math.pi)
        return self.mlp(torch.cat([phase.sin(), phase.cos()], -1))


class VLAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(cfg.vl_dim), nn.Linear(cfg.vl_dim, cfg.dim))
        self.xy = nn.Linear(2, cfg.dim, bias=False)
        self.kind = nn.Embedding(2, cfg.dim)
        self.camera = nn.Embedding(len(cfg.cameras) + 1, cfg.dim)  # 0=text

    def forward(self, tokens, xy, camera):
        return self.project(tokens.float()) + self.xy(xy.float()) + self.kind((camera > 0).long()) + self.camera(camera)


class AdaLNZero(nn.Module):
    """shift, scale and residual gate for each of the three branches."""
    def __init__(self, dim):
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, time):
        return self.modulation(time).unsqueeze(1).chunk(9, dim=-1)


class FlowDiTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.norms = nn.ModuleList([nn.LayerNorm(d, elementwise_affine=False) for _ in range(3)])
        self.adaln = AdaLNZero(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ffn), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, context, context_mask, sequence_mask, time):
        ss, sc, sg, cs, cc, cg, fs, fc, fg = self.adaln(time)
        z = self.norms[0](x) * (1 + sc) + ss
        x = x + sg * self.drop(self.self_attn(z, z, z, key_padding_mask=~sequence_mask, need_weights=False)[0])
        z = self.norms[1](x) * (1 + cc) + cs
        x = x + cg * self.drop(self.cross(z, context, context, key_padding_mask=~context_mask, need_weights=False)[0])
        z = self.norms[2](x) * (1 + fc) + fs
        return x + fg * self.drop(self.ff(z))


class ActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapter = VLAdapter(cfg)
        self.state = nn.Sequential(nn.Linear(6, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.action = nn.Linear(6, cfg.dim)
        self.position = nn.Parameter(torch.randn(1, cfg.horizon, cfg.dim) * .02)
        self.time = TimeEmbedding(cfg.dim)
        self.blocks = nn.ModuleList([FlowDiTBlock(cfg) for _ in range(cfg.layers)])
        self.out = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 6))
        weights = torch.ones(6); weights[-1] = cfg.gripper_weight
        self.register_buffer('joint_weights', weights)

    def context(self, features):
        tokens, mask, xy, camera = features
        if tokens.ndim != 3 or tokens.shape[-1] != self.cfg.vl_dim:
            raise ValueError('Bad VL token shape')
        if mask.shape != tokens.shape[:2] or xy.shape != (*mask.shape, 2) or camera.shape != mask.shape:
            raise ValueError('VL token/metadata shape mismatch')
        if not mask.any(-1).all(): raise ValueError('Each sample needs at least one valid VL token')
        tokens = tokens.masked_fill(~mask.bool()[..., None], 0)
        return self.adapter(tokens, xy, camera), mask.bool()

    def forward(self, context, context_mask, state, noisy, time, action_mask=None):
        b = state.shape[0]
        if state.shape != (b, 6): raise ValueError('Expected state [B,6]')
        if noisy.shape != (b, self.cfg.horizon, 6) or time.shape != (b,):
            raise ValueError('Expected noisy actions [B,H,6] and flow time [B]')
        if action_mask is None:
            action_mask = torch.ones(b, self.cfg.horizon, device=state.device, dtype=torch.bool)
        if action_mask.shape != (b, self.cfg.horizon) or not action_mask.any(-1).all():
            raise ValueError('Empty or malformed action supervision')
        actions = self.action(noisy.float().masked_fill(~action_mask[..., None], 0)) + self.position
        x = torch.cat([self.state(state.float())[:, None], actions], dim=1)
        sequence_mask = torch.cat([torch.ones(b, 1, device=state.device, dtype=torch.bool), action_mask], dim=1)
        time = self.time(time)
        for block in self.blocks:
            x = block(x, context, context_mask, sequence_mask, time)
        return self.out(x[:, 1:])

    def weighted_loss(self, error, mask):
        w = mask[..., None].float() * self.joint_weights
        if w.sum() == 0: raise ValueError('No valid actions')
        return (error.float().masked_fill(~mask[..., None], 0) * w).sum() / w.sum()

    def loss(self, features, state, actions, mask):
        ctx, cmask = self.context(features)
        if actions.shape != (state.shape[0], self.cfg.horizon, 6): raise ValueError('Bad action shape')
        actions = actions.masked_fill(~mask[..., None], 0)
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        pred = self(ctx, cmask, state, x, t, mask)
        loss = self.weighted_loss((pred - (actions - noise)).square(), mask)
        return loss, {'action_loss': float(loss.detach())}

    @torch.no_grad()
    def sample(self, features, state, noise=None):
        ctx, cmask = self.context(features)  # reused for all eight Euler steps
        x = torch.randn(state.shape[0], self.cfg.horizon, 6, device=state.device) if noise is None else noise.to(state.device).clone()
        if x.shape != (state.shape[0], self.cfg.horizon, 6): raise ValueError('Bad inference noise shape')
        for k in range(self.cfg.flow_steps):
            t = torch.full((state.shape[0],), k / self.cfg.flow_steps, device=state.device)
            x = x + self(ctx, cmask, state, x, t) / self.cfg.flow_steps
        return x
