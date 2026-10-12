import math
import torch
from torch import nn
from torch.nn import functional as F

class TimeEmbedding(nn.Module):
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
        self.camera = nn.Embedding(5, cfg.dim)  # 0=text, 1..4=cameras
    def forward(self, tokens, xy, camera):
        return self.project(tokens.float()) + self.xy(xy.float()) + self.kind((camera > 0).long()) + self.camera(camera)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ffn), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn, d))
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x, context, context_mask, action_mask):
        z = self.norms[0](x)
        x = x + self.drop(self.self_attn(z, z, z, key_padding_mask=~action_mask, need_weights=False)[0])
        x = x + self.drop(self.cross(self.norms[1](x), context, context, key_padding_mask=~context_mask, need_weights=False)[0])
        return x + self.drop(self.ff(self.norms[2](x)))

class ActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapter = VLAdapter(cfg)
        self.state = nn.Sequential(nn.Linear(6, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.position = nn.Parameter(torch.randn(1, cfg.horizon, cfg.dim) * .02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.out = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 6))
        if cfg.head_type == 'flow':
            self.action = nn.Linear(6, cfg.dim)
            self.time = TimeEmbedding(cfg.dim)
        self.ground_queries = nn.Parameter(torch.randn(1, 2, cfg.dim) * .02)
        self.ground_attention = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
        self.ground_out = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 2), nn.Sigmoid())
        weights = torch.ones(6); weights[-1] = cfg.gripper_weight
        self.register_buffer('joint_weights', weights)

    def context(self, features):
        tokens, mask, xy, camera = features
        if tokens.ndim != 3 or tokens.shape[-1] != self.cfg.vl_dim:
            raise ValueError('Bad VL token shape')
        if not mask.any(-1).all(): raise ValueError('Each sample needs at least one valid VL token')
        return self.adapter(tokens, xy, camera), mask.bool()

    def forward(self, context, context_mask, state, noisy=None, time=None, action_mask=None):
        b = state.shape[0]
        if state.shape != (b, 6): raise ValueError('Expected state [B,6]')
        if action_mask is None:
            action_mask = torch.ones(b, self.cfg.horizon, device=state.device, dtype=torch.bool)
        if not action_mask.any(-1).all(): raise ValueError('Empty action supervision')
        x = self.position.expand(b, -1, -1) + self.state(state.float())[:, None]
        if self.cfg.head_type == 'flow':
            if noisy is None or time is None: raise ValueError('Flow requires noise and time')
            x = x + self.action(noisy.float().masked_fill(~action_mask[..., None], 0)) + self.time(time)[:, None]
        for block in self.blocks: x = block(x, context, context_mask, action_mask)
        return self.out(x)

    def grounding(self, context, mask):
        q = self.ground_queries.expand(context.shape[0], -1, -1)
        z = self.ground_attention(q, context, context, key_padding_mask=~mask, need_weights=False)[0]
        return self.ground_out(z + q)

    def weighted_loss(self, error, mask):
        w = mask[..., None].float() * self.joint_weights
        if w.sum() == 0: raise ValueError('No valid actions')
        return (error.float() * w).sum() / w.sum()

    def loss(self, features, state, actions, mask, grounding=None, grounding_mask=None):
        ctx, cmask = self.context(features)
        if actions.shape != (state.shape[0], self.cfg.horizon, 6): raise ValueError('Bad action shape')
        actions = actions.masked_fill(~mask[..., None], 0)
        if self.cfg.head_type == 'flow':
            noise = torch.randn_like(actions)
            t = torch.rand(actions.shape[0], device=actions.device)
            x = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
            pred = self(ctx, cmask, state, x, t, mask)
            loss = self.weighted_loss((pred - (actions-noise)).square(), mask)
        else:
            pred = self(ctx, cmask, state, action_mask=mask)
            loss = self.weighted_loss(F.smooth_l1_loss(pred, actions, reduction='none'), mask)
        metrics = {'action_loss': float(loss.detach())}
        if grounding is not None and grounding_mask is not None and grounding_mask.any():
            g = F.smooth_l1_loss(self.grounding(ctx, cmask), grounding, reduction='none').mean(-1)
            aux = (g * grounding_mask).sum() / grounding_mask.sum()
            loss = loss + self.cfg.grounding_weight * aux
            metrics['grounding_loss'] = float(aux.detach())
        return loss, metrics

    @torch.no_grad()
    def sample(self, features, state, noise=None):
        ctx, cmask = self.context(features)
        if self.cfg.head_type == 'deterministic': return self(ctx, cmask, state)
        x = torch.randn(state.shape[0], self.cfg.horizon, 6, device=state.device) if noise is None else noise.clone()
        if x.shape != (state.shape[0], self.cfg.horizon, 6): raise ValueError('Bad inference noise shape')
        for k in range(self.cfg.flow_steps):
            t = torch.full((state.shape[0],), k/self.cfg.flow_steps, device=state.device)
            x = x + self(ctx, cmask, state, x, t) / self.cfg.flow_steps
        return x
