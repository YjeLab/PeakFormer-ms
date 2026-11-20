# ============================================================
# PeakFormer v7 (PeakFormer-R): v3 pyramid with a high-resolution
# input representation + candidate-conditioned scoring head.
#
# Encoder changes vs v3 (all on the "make exact m/z usable" axis):
#   * sinusoidal m/z and neutral-loss encodings (mDa-visible)
#   * per-peak subformula annotation channel (id embedding + ppm
#     residual) from a data-mined vocab (MIST-CF inspired)
#   * piecewise relative-m/z buckets: 2 mDa linear bins below
#     0.2 Da, log bins above (isobar chemistry gets resolution)
#   * proper adduct embedding in the env vector (was a scalar/6)
#
# CandidateScorer: MIST-CF style energy head -- query = candidate
# formula embedding, cross-attention over the frozen encoder's
# multi-scale peak features, output = scalar score. Trained with
# listwise CE over same-mass candidate pools (P1 candidates).
# ============================================================
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v3 import RMSNorm, RelAttnBlock, WindowDown, AttnPool, FiLM
from .features import make_rel_edges

PEAKFORMER7_CONFIG = {
    'max_peaks': 256,
    'd_model': 128,
    'n_heads': 4,
    'stride': 8,
    'n_blocks': [1, 2, 1],
    'dropout': 0.1,
    'gap_log_base': 0.01,
    'max_mz': 1500.0,
    'n_sin_mz': 6,
    'n_sin_nl': 4,
    'sf_vocab': 513,       # 512 subformulae + id 0 = no match
    'sf_emb': 16,
    'n_rel_lin': 100,      # 2 mDa bins up to 0.2 Da
    'n_rel_log': 48,
    'count_classes': 128,
    'n_atoms': 13,
}


def _sinusoidal(x, n_freq, lam_min, lam_max):
    lams = torch.logspace(math.log10(lam_min), math.log10(lam_max), n_freq,
                          dtype=torch.float64, device=x.device)
    ang = 2 * math.pi * x.double().unsqueeze(-1) / lams
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).float()


class PeakFormerV7(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = PEAKFORMER7_CONFIG if config is None else config
        c = self.cfg
        D, S = c['d_model'], c['stride']
        assert c['max_peaks'] % (S * S) == 0
        self.n1 = c['max_peaks'] // S
        self.n2 = self.n1 // S

        # ablation switches (all default off -> identical to baseline v7)
        self.abl_single = c.get('abl_single_scale', False)   # A1
        self.abl_no_fpn = c.get('abl_no_fpn', False)         # A2
        self.abl_mean_pool = c.get('abl_mean_pool', False)   # A3
        self.abl_concat = c.get('abl_concat_env', False)     # A4
        self.abl_lin_mz = c.get('abl_linear_mz', False)      # B1
        self.abl_no_sf = c.get('abl_no_subformula', False)   # B2

        # numeric token dims: sin(mz) 2F1 + sin(nl) 2F2 + inten + mz + gaps2 + sf_res
        n_sin_mz_dim = 0 if self.abl_lin_mz else 2 * c['n_sin_mz']
        n_sf_res_dim = 0 if self.abl_no_sf else 1
        self.n_num = n_sin_mz_dim + 2 * c['n_sin_nl'] + 4 + n_sf_res_dim
        self.sf_emb = None if self.abl_no_sf else nn.Embedding(c['sf_vocab'], c['sf_emb'])
        sf_in = 0 if self.abl_no_sf else c['sf_emb']
        self.token_mlp = nn.Sequential(nn.Linear(self.n_num + sf_in, D),
                                       nn.GELU(), nn.Linear(D, D))

        edges = make_rel_edges(0.20, c['n_rel_lin'], 0.002, c['max_mz'], c['n_rel_log'])
        self.register_buffer('rel_edges', torch.tensor(edges, dtype=torch.float32))
        self.n_rel = len(edges) + 1
        self.rel_bias = nn.Embedding(self.n_rel, c['n_heads'])
        nn.init.zeros_(self.rel_bias.weight)

        self.peak_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][0])])
        self.down1 = None if self.abl_single else WindowDown(D, S)
        if self.abl_concat:
            self.concat1 = nn.Linear(D + 10, D)
        else:
            self.film1 = FiLM(d_env=10, d=D)
        self.patch_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][1])])
        self.down2 = None if self.abl_single else WindowDown(D, S)
        if self.abl_concat:
            self.concat2 = nn.Linear(D + 10, D)
        else:
            self.film2 = FiLM(d_env=10, d=D)
        self.global_blocks = nn.ModuleList([
            RelAttnBlock(D, c['n_heads'], c['dropout'], self.rel_bias)
            for _ in range(c['n_blocks'][2])])

        self.adduct_emb = nn.Embedding(7, 8)

        if self.abl_mean_pool:
            self.pool0 = self.pool1 = self.pool2 = None
        else:
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
            elif isinstance(m, nn.Embedding) and m is not self.rel_bias:
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _gap_feats(mz, mask, base, denom):
        prev = F.pad(mz, (1, 0))[..., :-1]
        nxt = F.pad(mz, (0, 1))[..., 1:]
        gp = (mz - prev).clamp(min=0)
        gn = (nxt - mz).clamp(min=0)
        gp = torch.where(mask, gp, torch.zeros_like(gp))
        gn = torch.where(mask & (nxt > 0), gn, torch.zeros_like(gn))
        return torch.log1p(gp / base) / denom, torch.log1p(gn / base) / denom

    def _rel(self, centers):
        d = (centers.unsqueeze(-1) - centers.unsqueeze(1)).abs()
        return torch.bucketize(d, self.rel_edges).clamp(max=self.n_rel - 1)

    @staticmethod
    def _win_mean(values, mask, n_win, win):
        v = (values * mask).view(-1, n_win, win).sum(-1)
        c = mask.view(-1, n_win, win).sum(-1)
        return v / c.clamp(min=1.0)

    def env_vec(self, env):
        return torch.cat([self.adduct_emb(env[:, 2].long().clamp(0, 6)),
                          env[:, 1:2] / 100.0, env[:, 0:1] / self.cfg['max_mz']], dim=-1)

    @staticmethod
    def _mean_pool(x, mask):
        return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def _env_mod(self, x, ev, which):
        # A4: FiLM -> concat injection
        if self.abl_concat:
            lin = self.concat1 if which == 1 else self.concat2
            e = ev.unsqueeze(1).expand(-1, x.size(1), -1)
            return lin(torch.cat([x, e], dim=-1))
        film = self.film1 if which == 1 else self.film2
        return film(x, ev)

    def forward(self, mz, intensity, env, sf_id=None, sf_res=None, return_features=False):
        B, N = mz.shape
        c = self.cfg
        S = c['stride']
        mask = intensity > 0
        if sf_id is None:
            sf_id = torch.zeros(B, N, dtype=torch.long, device=mz.device)
        if sf_res is None:
            sf_res = torch.zeros(B, N, dtype=torch.float32, device=mz.device)

        pmz = env[:, 0:1]
        nl = (pmz - mz).clamp(min=0)
        int_norm = intensity / intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        base = c['gap_log_base']
        denom = math.log1p(c['max_mz'] / base)
        gp, gn = self._gap_feats(mz, mask, base, denom)

        feats = []
        if not self.abl_lin_mz:
            feats.append(_sinusoidal(mz / c['max_mz'], c['n_sin_mz'],
                                     0.008 / c['max_mz'], 1.0))
        feats.append(_sinusoidal(nl / c['max_mz'], c['n_sin_nl'],
                                 0.008 / c['max_mz'], 0.4))
        feats.append((torch.log1p(int_norm * 50.0) / math.log1p(50.0)).unsqueeze(-1))
        feats.append((mz / c['max_mz']).unsqueeze(-1))
        feats.append(gp.unsqueeze(-1))
        feats.append(gn.unsqueeze(-1))
        if not self.abl_no_sf:
            feats.append((sf_res.clamp(-100, 100) / 100.0).unsqueeze(-1))
        feats = torch.cat(feats, dim=-1)
        if self.abl_no_sf:
            x0 = self.token_mlp(feats)
        else:
            x0 = self.token_mlp(torch.cat([feats, self.sf_emb(sf_id)], dim=-1))

        ev = self.env_vec(env)
        mzf = mz.float()
        maskf = mask.float()

        rel0 = self._rel(mz)
        for blk in self.peak_blocks:
            x0 = blk(x0, mask, rel=rel0)

        if self.abl_single:
            # A1: no pyramid -- all blocks operate at peak resolution
            x = self._env_mod(x0, ev, 1)
            for blk in self.patch_blocks:
                x = blk(x, mask, rel=rel0)
            x = self._env_mod(x, ev, 2)
            for blk in self.global_blocks:
                x = blk(x, mask, rel=rel0)
            x0 = x1 = x2 = x
            m1 = m2 = mask
        else:
            x1, m1 = self.down1(x0, mask)
            x1 = self._env_mod(x1, ev, 1)
            mz1 = self._win_mean(mzf, maskf, self.n1, S)
            for blk in self.patch_blocks:
                x1 = blk(x1, m1, rel=self._rel(mz1))

            x2, m2 = self.down2(x1, m1)
            x2 = self._env_mod(x2, ev, 2)
            mz2 = self._win_mean(mz1, m1.float(), self.n2, S)
            for blk in self.global_blocks:
                x2 = blk(x2, m2, rel=self._rel(mz2))

            if not self.abl_no_fpn:
                x1 = x1 + F.interpolate(x2.transpose(1, 2), size=self.n1,
                                        mode='nearest').transpose(1, 2)
                x0 = x0 + F.interpolate(x1.transpose(1, 2), size=N,
                                        mode='nearest').transpose(1, 2)

        if return_features:
            return (x0, mask), (x1, m1), (x2, m2)

        if self.abl_mean_pool:
            p0, p1, p2 = (self._mean_pool(x0, mask), self._mean_pool(x1, m1),
                          self._mean_pool(x2, m2))
        else:
            p0, p1, p2 = self.pool0(x0, mask), self.pool1(x1, m1), self.pool2(x2, m2)
        h = self.fuse(torch.cat([p0, p1, p2], dim=-1))
        h = self.fuse_pmz(torch.cat([h, pmz / c['max_mz']], dim=-1))
        h = self.head_norm(h)

        logits = self.atom_head(h).view(B, c['n_atoms'], c['count_classes'])
        mass = self.mass_head(h).squeeze(-1)
        atomnum = self.atomnum_head(h).squeeze(-1)
        hcnum = self.hcnum_head(h).squeeze(-1)
        return h, logits, mass, atomnum, hcnum

    def decode_counts(self, logits):
        return logits.argmax(dim=-1)


class CandCrossAttn(nn.Module):
    """Multi-query cross-attention against concatenated multi-scale peak
    features WITHOUT replicating KV (einsum), so K candidates are cheap."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.nq = RMSNorm(d)
        self.nkv = RMSNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.n2 = RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv, kmask):
        # q: (B,K,D); kv: (B,N,D); kmask: (B,N) True=valid
        B, K, D = q.shape
        N = kv.size(1)
        r = q
        qs = self.wq(self.nq(q)).view(B, K, self.h, self.dh).transpose(1, 2)   # B,h,K,dh
        xk = self.wk(self.nkv(kv)).view(B, N, self.h, self.dh).transpose(1, 2)
        vv = self.wv(kv).view(B, N, self.h, self.dh).transpose(1, 2)
        s = torch.einsum('bhkd,bhnd->bhkn', qs, xk) / math.sqrt(self.dh)
        s = s.masked_fill(~kmask.unsqueeze(1).unsqueeze(2), float('-inf'))
        a = torch.softmax(s, dim=-1)
        a = torch.nan_to_num(a, 0.0)
        o = torch.einsum('bhkn,bhnd->bhkd', a, vv)
        o = o.transpose(1, 2).reshape(B, K, D)
        q = r + self.drop(self.wo(o))
        q = q + self.drop(self.ffn(self.n2(q)))
        return q


class CandidateScorer(nn.Module):
    """score(spectrum-features, candidate formula). Counts enter as a dense
    13-vector: all mass arithmetic (exact mass, RDBE) is linear in counts,
    so a single MLP resolves mDa-level differences exactly."""

    def __init__(self, d, n_heads, atom_mass, n_layers=2, dropout=0.1, n_cand=48):
        super().__init__()
        self.register_buffer('atom_mass', torch.tensor(atom_mass, dtype=torch.float64))
        self.cand_in = nn.Sequential(nn.Linear(24, d), nn.GELU(), nn.Linear(d, d))
        self.lvl0 = nn.Parameter(torch.zeros(1, 1, 3, d))
        self.blocks = nn.ModuleList([CandCrossAttn(d, n_heads, dropout)
                                     for _ in range(n_layers)])
        self.out = nn.Sequential(nn.Linear(d + 8, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 1))
        self.mass_stats = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 8))

    def cand_feats(self, counts, mass, env):
        # counts: (B,K,13) float; mass: (B,); env: (B,3)
        cm = counts.double() @ self.atom_mass                              # (B,K)
        ppm = ((cm - mass.double().unsqueeze(1)) / mass.double().unsqueeze(1)
               * 1e6).clamp(-50, 50) / 50.0
        tot = counts.sum(-1) / 50.0
        rdbe = (counts[..., 0] - counts[..., 1] / 2 + counts[..., 3] / 2
                + counts[..., 7] / 2 + 1.0) / 15.0
        hc = counts[..., 1] / counts[..., 0].clamp(min=1) / 4.0
        nheavy = (counts[..., [0, 2, 3, 5, 6, 7]].sum(-1)) / 30.0
        nidx = (counts > 0).sum(-1) / 8.0
        extra = torch.stack([ppm.float(), tot, rdbe.float(), hc, nheavy, nidx,
                             (counts[..., 1] / 20.0), counts[..., 4:13].sum(-1) / 10.0], dim=-1)
        env_b = torch.stack([env[:, 0] / 1500.0, env[:, 1] / 100.0,
                             env[:, 2] / 6.0], dim=-1)[:, None, :].expand(-1, counts.size(1), -1)
        return torch.cat([counts / 20.0, extra, env_b], dim=-1)

    def forward(self, feats, counts, mass, env):
        # feats: ((x0,m0),(x1,m1),(x2,m2)); counts: (B,K,13) float
        (x0, m0), (x1, m1), (x2, m2) = feats
        B, K, _ = counts.shape
        kv = torch.cat([x0 + self.lvl0[:, :, 0], x1 + self.lvl0[:, :, 1],
                        x2 + self.lvl0[:, :, 2]], dim=1)
        kmask = torch.cat([m0, m1, m2], dim=1)
        cf = self.cand_feats(counts, mass, env)
        q = self.cand_in(cf)
        for blk in self.blocks:
            q = blk(q, kv, kmask)
        ppm_dev = cf[..., 13:14]
        ms = self.mass_stats(torch.cat([ppm_dev, ppm_dev.abs()], dim=-1))
        return self.out(torch.cat([q, ms], dim=-1)).squeeze(-1)
