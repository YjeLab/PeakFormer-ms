# ============================================================
# v7 feature engineering:
#   * sinusoidal m/z encoding (mDa-resolution visible to MLPs)
#   * fixed subformula vocabulary mined from train peak/gap
#     statistics; each peak annotated with nearest subformula id
#     + signed ppm residual (MIST-CF style, precursor-relative)
#   * piecewise rel-m/z buckets (linear mDa bins, then log)
# ============================================================
import json

import numpy as np
import torch

# monoisotopic masses (molmass isotope masses)
SUB_MASSES = {'C': 12.0, 'H': 1.00782503207, 'N': 14.00307400522, 'O': 15.9949146221,
              'S': 31.97207100, 'P': 30.97376151, 'F': 18.99840322, 'Cl': 34.96885271,
              'Br': 78.91833760}
PROTON = 1.00727646

# enumeration caps for the subformula vocab (organic fragments)
SUB_CAPS = {'C': 10, 'N': 4, 'O': 5, 'S': 2, 'P': 2, 'F': 4, 'Cl': 2, 'Br': 1}
SUB_MAX_MASS = 600.0


def sinusoidal_features(mz, n_freq=6, lam_min=0.008, lam_max=1500.0):
    # mz (Tensor) -> (..., 2*n_freq) sin/cos at log-spaced wavelengths
    lams = torch.logspace(torch.log10(torch.tensor(lam_min)).item(),
                          torch.log10(torch.tensor(lam_max)).item(), n_freq,
                          dtype=torch.float64, device=mz.device)
    ang = 2 * torch.pi * mz.double().unsqueeze(-1) / lams
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).float()


def enumerate_subformulae():
    """All (mass, counts) with heavy atoms under caps, H closed by RDBE-valid range,
    SENIOR parity + valence filters, mass <= SUB_MAX_MASS. Returns sorted arrays."""
    import itertools
    heavy = list(SUB_CAPS.keys())
    caps = [SUB_CAPS[a] for a in heavy]
    out_m, out_c = [], []
    for combo in itertools.product(*[range(c + 1) for c in caps]):
        mh = sum(n * SUB_MASSES[a] for a, n in zip(heavy, combo))
        if mh > SUB_MAX_MASS:
            continue
        cC, cN = combo[0], combo[2]
        maxH = max(0, 2 * cC + cN + 3) if cC else (0 if sum(combo) <= 1 else 12)
        maxH = min(maxH, 24)
        for h in range(maxH + 1):
            counts = dict(zip(heavy, combo))
            counts['H'] = h
            m = mh + h * SUB_MASSES['H']
            if m < 15.0 or m > SUB_MAX_MASS:
                continue
            val = {'C': 4, 'H': 1, 'N': 3, 'O': 2, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1}
            tot = sum(val[a] * n for a, n in counts.items() if n)
            odd = sum(val[a] * n for a, n in counts.items() if val[a] % 2 and n)
            mx = max((val[a] for a, n in counts.items() if n), default=0)
            na = sum(counts.values())
            if odd % 2 or tot < 2 * mx or tot < 2 * (na - 1):
                continue
            out_m.append(m)
            out_c.append([n for n in (counts[a] for a in heavy)] + [h])
    masses = np.asarray(out_m, dtype=np.float64)
    counts = np.asarray(out_c, dtype=np.int16)
    order = np.argsort(masses)
    return masses[order], counts[order], heavy + ['H']


def build_vocab(train_path, top_k=512, tol_ppm=8.0, chunk=500):
    """Mine the most frequently observed (mass, counts) among all fragment peaks
    and all precursor-neutral gaps in the training set."""
    import pickle
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    masses, counts, atoms = enumerate_subformulae()
    hits = np.zeros(len(masses), dtype=np.int64)
    for s in range(0, len(data), chunk):
        mz = np.asarray(data[s]['mz'], dtype=np.float64)
        if len(mz) == 0:
            continue
        pmz = float(np.asarray(data[s]['env'], dtype=np.float64)[0])
        vals = np.concatenate([pmz - mz[mz < pmz - 0.001], mz[mz > 15.0]])
        lo = np.searchsorted(masses, vals * (1 - tol_ppm * 1e-6))
        hi = np.searchsorted(masses, vals * (1 + tol_ppm * 1e-6))
        for i in range(len(vals)):
            if hi[i] > lo[i]:
                seg = masses[lo[i]:hi[i]]
                j = lo[i] + int(np.argmin(np.abs(seg - vals[i])))
                hits[j] += 1
    sel = np.argsort(hits)[::-1][:top_k]
    sel = sel[hits[sel] > 0]
    sel = np.sort(sel)
    return {'masses': masses[sel].tolist(), 'counts': counts[sel].tolist(),
            'atoms': atoms, 'freq': hits[sel].tolist()}


def annotate_peaks(mz, pmz, vocab_masses, tol_ppm=8.0):
    """mz (n,) -> (sf_id int16 (n,), resid_ppm float32 (n,)).
    Matches neutral-loss values pmz-mz (and bare mz) against the vocab; id 0
    reserved for 'no match' (vocab arrays are 1-indexed)."""
    neutral = pmz - mz
    neutral[neutral < 15.0] = -1.0
    bare = mz.copy()
    def _match(vals):
        n = len(vals)
        ids = np.zeros(n, dtype=np.int16)
        res = np.zeros(n, dtype=np.float32)
        valid = vals > 0
        if not valid.any():
            return ids, res
        vv = vals[valid]
        pos = np.searchsorted(vocab_masses, vv)
        li = np.clip(pos - 1, 0, len(vocab_masses) - 1)
        ri = np.clip(pos, 0, len(vocab_masses) - 1)
        use_r = np.abs(vocab_masses[ri] - vv) <= np.abs(vocab_masses[li] - vv)
        j = np.where(use_r, ri, li)
        r = (vv - vocab_masses[j]) / vv * 1e6
        ok = np.abs(r) <= tol_ppm
        vidx = np.nonzero(valid)[0]
        ids[vidx[ok]] = (j[ok] + 1).astype(np.int16)
        res[vidx[ok]] = r[ok].astype(np.float32)
        return ids, res
    idn, resn = _match(neutral)
    idb, resb = _match(bare)
    # prefer neutral-loss match; fall back to bare m/z match
    out_id = np.where(idn > 0, idn, idb)
    out_res = np.where(idn > 0, resn, resb)
    return out_id, out_res


def make_rel_edges(bound_lin=0.20, n_lin=100, lo=0.002, hi=1500.0, n_log=48):
    """Piecewise bucket edges: n_lin equal bins on [0, bound_lin) (1-2 mDa
    resolution for isobar chemistry), then log bins to hi."""
    lin = np.linspace(0.0, bound_lin, n_lin, endpoint=False)[1:]
    logb = np.logspace(np.log10(bound_lin), np.log10(hi), n_log)
    return np.unique(np.concatenate([lin, logb]))
