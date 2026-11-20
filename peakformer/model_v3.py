# ============================================================
# PeakFormer v3: gap-aware peak tokens + relative m/z attention
#   bias + learnable strided-conv pyramid (index-space) + FPN
#   top-down + attention-pooling readout.
#
# vs v2:
#   * peak tokens carry neighbor m/z gaps (log-scaled)
#   * attention gets a learned per-head bias on |mz_i - mz_j|
#     (log-bucketed) -> explicit neutral-loss/complementary prior
#   * fixed 25 Da masked-mean patches REPLACED by learnable
#     within-window conv downsample in peak-index space:
#     256 -> 32 -> 4, attention at every level (pyramid)
#   * top-down fusion by nearest upsample + add (was scatter-mean)
#   * readout: attention pooling per level (was masked mean)
#   * max_peaks truncation kept as in v2 (only 1.31% of spectra
#     are affected; intentionally unchanged)
# ============================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PEAKFORMER3_CONFIG = {
    'max_peaks': 256,
    'd_model': 128,
    'n_heads': 4,
    'stride': 8,               # downsample factor per level
    'n_blocks': [1, 2, 1],     # attention blocks at levels 256 / 32 / 4
    'dropout': 0.1,
    'gap_log_base': 0.01,      # Da; gaps compressed via log1p(gap/base)
    'max_mz': 1500.0,
    'n_rel_buckets': 32,       # |mz_i - mz_j| log-bucket count
    'count_classes': 128,
    'n_atoms': 13,
}


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class RelAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN with a shared learned relative-m/z bias."""

    def __init__(self, d, n_heads, dropout, rel_bias=None):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d // n_heads
        self.rel_bias = rel_bias  # nn.Embedding(n_buckets, n_heads) shared across levels
        self.norm1 = RMSNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.norm2 = RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, rel=None):
        # x: (B,N,D), mask: (B,N) True=valid, rel: (B,N,N) long bucket ids
        B, N, D = x.shape
        r = x
        x = self.norm1(x)
        q = self.wq(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, N, self.n_heads, self.dh).transpose(1, 2)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if self.rel_bias is not None and rel is not None:
            s = s + self.rel_bias(rel).permute(0, 3, 1, 2)
        pad = ~mask.unsqueeze(1).unsqueeze(2)
        s = s.masked_fill(pad, float('-inf'))
        a = torch.softmax(s, dim=-1)
        a = a.masked_fill(pad, 0.0)
        o = (a @ v).transpose(1, 2).reshape(B, N, D)
        x = r + self.drop(self.wo(o))
        r = x
        x = self.norm2(x)
        x = r + self.drop(self.ffn(x))
        return x


class WindowDown(nn.Module):
    """Learnable downsampling in peak-index space: local depthwise conv
    within each S-peak window, masked mean, then pointwise mix."""

    def __init__(self, d, stride):
        super().__init__()
        self.S = stride
        self.local = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d)
        self.proj = nn.Conv1d(d, d, kernel_size=1)
        self.norm = RMSNorm(d)

    def forward(self, x, mask):
        # x: (B,N,D) -> (B,N/S,D); window j covers peaks [j*S,(j+1)*S)
        B, N, D = x.shape
        S = self.S
        xm = (x * mask.unsqueeze(-1))
        w = xm.view(B, N // S, S, D).permute(0, 1, 3, 2).reshape(B * (N // S), D, S)
        wm = mask.view(B * (N // S), 1, S)
        y = self.local(w) * wm
        y = y.sum(-1) / wm.sum(-1).clamp(min=1.0)          # (B*N/S, D)
        y = self.proj(y.view(B, N // S, D).transpose(1, 2)).transpose(1, 2)
        cnt = mask.view(B, N // S, S).sum(-1)
        return self.norm(y), cnt > 0


class AttnPool(nn.Module):
    """Single-query masked attention pooling."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(d, d)
        self.norm = RMSNorm(d)

    def forward(self, x, mask):
        # x: (B,N,D) -> (B,D)
        x = self.norm(x)
        s = (self.k(x) @ self.q) / math.sqrt(x.size(-1))
        s = s.masked_fill(~mask, float('-inf'))
        a = torch.softmax(s, dim=-1).unsqueeze(-1)
        return (a * x).sum(1)


class FiLM(nn.Module):
    def __init__(self, d_env, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d_env, 32), nn.ReLU(), nn.Linear(32, 2 * d))

    def forward(self, x, env):
        gb = self.mlp(env)
        g, b = gb[:, :self.d].unsqueeze(1), gb[:, self.d:].unsqueeze(1)
        return x * (1 + g) + b


class PeakFormerV3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = PEAKFORMER3_CONFIG if config is None else config
        c = self.cfg
        D, S = c['d_model'], c['stride']
        assert c['max_peaks'] % (S * S) == 0, 'max_peaks must divide by stride^2'
        self.n1 = c['max_peaks'] // S      # 32
        self.n2 = self.n1 // S             # 4

        self.token_mlp = nn.Sequential(nn.Linear(5, D), nn.GELU(), nn.Linear(D, D))

        edges = torch.logspace(math.log10(c['gap_log_base']),
                               math.log10(c['max_mz']), c['n_rel_buckets'] - 1)
        self.register_buffer('rel_edges', edges)
        self.rel_bias = nn.Embedding(c['n_rel_buckets'], c['n_heads'])
        nn.init.zeros_(self.rel_bias.weight)

        self.peak_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][0])])

        self.down1 = WindowDown(D, S)
        self.film1 = FiLM(d_env=3, d=D)
        self.patch_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][1])])

        self.down2 = WindowDown(D, S)
        self.film2 = FiLM(d_env=3, d=D)
        self.global_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][2])])

        self.pool0, self.pool1, self.pool2 = AttnPool(D), AttnPool(D), AttnPool(D)
        self.fuse = nn.Sequential(nn.Linear(3 * D, D), nn.GELU())
        self.fuse_pmz = nn.Sequential(nn.Linear(D + 1, D), nn.GELU())
        self.head_norm = RMSNorm(D)

        self.atom_head = nn.Linear(D, c['n_atoms'] * c['count_classes'])
        self.mass_head = nn.Linear(D, 1)
        self.atomnum_head = nn.Linear(D, 1)
        self.hcnum_head = nn.Linear(D, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------- helpers ----------------
    def _gap_feats(self, mz, mask):
        base = self.cfg['gap_log_base']
        denom = math.log1p(self.cfg['max_mz'] / base)
        prev = F.pad(mz, (1, 0))[..., :-1]
        nxt = F.pad(mz, (0, 1))[..., 1:]
        gp = (mz - prev).clamp(min=0)
        gn = (nxt - mz).clamp(min=0)
        gp = torch.where(mask, gp, torch.zeros_like(gp))
        gn = torch.where(mask & (nxt > 0), gn, torch.zeros_like(gn))
        return torch.log1p(gp / base) / denom, torch.log1p(gn / base) / denom

    def _rel(self, centers):
        d = (centers.unsqueeze(-1) - centers.unsqueeze(1)).abs()
        return torch.bucketize(d, self.rel_edges).clamp(max=self.cfg['n_rel_buckets'] - 1)

    @staticmethod
    def _win_mean(values, mask, n_win, win):
        # mean of values within each index window (padding zeros ignored)
        v = (values * mask).view(-1, n_win, win).sum(-1)
        c = mask.view(-1, n_win, win).sum(-1)
        return v / c.clamp(min=1.0)

    # ---------------- forward ----------------
    def forward(self, mz, intensity, env, return_features=False):
        # mz (B,N), intensity (B,N), env (B,3); peaks m/z-sorted, zero-padded
        B, N = mz.shape
        S = self.cfg['stride']
        mask = intensity > 0

        pmz = env[:, 0:1]
        nl = (pmz - mz) / self.cfg['max_mz']
        int_norm = intensity / intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        gp, gn = self._gap_feats(mz, mask)
        feat = torch.stack([
            mz / self.cfg['max_mz'],
            torch.log1p(int_norm * 50.0) / math.log1p(50.0),
            nl, gp, gn,
        ], dim=-1)

        env_norm = torch.stack([
            env[:, 0] / self.cfg['max_mz'],
            env[:, 1] / 100.0,
            env[:, 2] / 6.0,
        ], dim=-1)

        mzf = mz.float()
        maskf = mask.float()

        x0 = self.token_mlp(feat)
        rel0 = self._rel(mz)
        for blk in self.peak_blocks:
            x0 = blk(x0, mask, rel=rel0)

        x1, m1 = self.down1(x0, mask)
        x1 = self.film1(x1, env_norm)
        mz1 = self._win_mean(mzf, maskf, self.n1, S)
        for blk in self.patch_blocks:
            x1 = blk(x1, m1, rel=self._rel(mz1))

        x2, m2 = self.down2(x1, m1)
        x2 = self.film2(x2, env_norm)
        mz2 = self._win_mean(mz1, m1.float(), self.n2, S)
        for blk in self.global_blocks:
            x2 = blk(x2, m2, rel=self._rel(mz2))

        # top-down: nearest upsample + add
        x1 = x1 + F.interpolate(x2.transpose(1, 2), size=self.n1, mode='nearest').transpose(1, 2)
        x0 = x0 + F.interpolate(x1.transpose(1, 2), size=N, mode='nearest').transpose(1, 2)

        if return_features:
            return (x0, mask), (x1, m1), (x2, m2)

        h = self.fuse(torch.cat([
            self.pool0(x0, mask),
            self.pool1(x1, m1),
            self.pool2(x2, m2),
        ], dim=-1))
        h = self.fuse_pmz(torch.cat([h, pmz / self.cfg['max_mz']], dim=-1))
        h = self.head_norm(h)

        logits = self.atom_head(h).view(B, self.cfg['n_atoms'], self.cfg['count_classes'])
        mass = self.mass_head(h).squeeze(-1)
        atomnum = self.atomnum_head(h).squeeze(-1)
        hcnum = self.hcnum_head(h).squeeze(-1)
        return h, logits, mass, atomnum, hcnum

    def decode_counts(self, logits):
        return logits.argmax(dim=-1)
