from typing import Dict, Optional, Tuple

import torch

from .device import _eye
from .stats import _check_neg, _check_st


def _eigh(A: torch.Tensor, B: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized symmetric eigendecomposition. If B is given, solves A v = λ B v via Cholesky factorisation."""
    if B is not None:
        L = torch.linalg.cholesky(B)
        Li = torch.linalg.inv(L)
        M = Li @ A @ Li.T
        ev, evec = torch.linalg.eigh(M)
        evec = Li.T @ evec
    else:
        ev, evec = torch.linalg.eigh(A)
    return ev, evec


def _importance_sort(W: torch.Tensor, st: Dict) -> torch.Tensor:
    """Sort columns of W by decreasing cross-covariance-to-total-variance ratio."""
    Cf = W.T @ st['Sigma_cross'] @ W
    Sf = W.T @ st['Sigma_total'] @ W
    eps = 1e-4 if Sf.dtype == torch.float16 else 1e-8
    imp = torch.diag(Cf) / (torch.diag(Sf) + eps)
    return W[:, torch.argsort(imp, descending=True)]


def _mse_refine(W: torch.Tensor, st: Dict, reg_mse: float = 0.1) -> torch.Tensor:
    """MSE refinement step: rotate W to align with cross-covariance in the projected subspace."""
    k = W.shape[1]
    Sp = W.T @ st['Sigma_total'] @ W
    Cp = W.T @ st['Sigma_cross'] @ W
    M = torch.linalg.solve(Sp + reg_mse * _eye(k), Cp)
    W2 = W @ M
    return _importance_sort(W2, st)


def _asym_refine(
    W: torch.Tensor,
    st: Dict,
    reg_refine: float = 0.1,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric Rayleigh refinement via iterative gamma balancing between X and Y variances."""
    k = W.shape[1]
    Axx = W.T @ st['Sigma_XX'] @ W
    Ayy = W.T @ st['Sigma_YY'] @ W
    Axy = W.T @ st['Sigma_XY'] @ W
    Across = Axy + Axy.T
    gamma = 1.0
    I_k = _eye(k)
    for _ in range(max_iter):
        Adenom = gamma * Axx + Ayy + reg_refine * I_k
        ev, evec = _eigh(Across, Adenom)
        M = evec[:, torch.argsort(ev, descending=True)]
        txx = torch.trace(M.T @ Axx @ M)
        tyy = torch.trace(M.T @ Ayy @ M)
        gamma_new = (tyy / (txx + 1e-12)).item()
        if abs(gamma_new - gamma) / (abs(gamma) + 1e-12) < tol:
            break
        gamma = gamma_new
    return _importance_sort(W @ M, st)


def m_rayleigh(st: Dict, neg: Optional[Dict] = None, reg: float = 0.1, **kw) -> torch.Tensor:
    _check_st(st)
    d = st['Sigma_total'].shape[0]
    ev, evec = _eigh(st['Sigma_cross'], st['Sigma_total'] + reg * _eye(d))
    return evec[:, torch.argsort(ev, descending=True)]


def m_mse(st: Dict, neg: Optional[Dict] = None, reg: float = 0.01, **kw) -> torch.Tensor:
    _check_st(st)
    d = st['Sigma_total'].shape[0]
    return torch.linalg.solve(st['Sigma_total'] + reg * _eye(d), st['Sigma_cross'])


def m_ray_mse(st: Dict, neg: Optional[Dict] = None, reg: float = 0.01, reg_mse: float = 0.1, **kw) -> torch.Tensor:
    _check_st(st)
    W = m_rayleigh(st, reg=reg)
    return _mse_refine(W, st, reg_mse=reg_mse)


def m_asym_rayleigh(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.1,
    max_iter: int = 20,
    tol: float = 1e-8,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    SXX, SYY = st['Sigma_XX'], st['Sigma_YY']
    S_cross = st['Sigma_cross']
    d = SXX.shape[0]
    gamma = 1.0
    W = None
    for _ in range(max_iter):
        S_denom = gamma * SXX + SYY + reg * _eye(d)
        ev, evec = _eigh(S_cross, S_denom)
        W = evec[:, torch.argsort(ev, descending=True)]
        tXX = torch.trace(W.T @ SXX @ W)
        tYY = torch.trace(W.T @ SYY @ W)
        gamma_new = (tYY / (tXX + 1e-12)).item()
        if abs(gamma_new - gamma) / (abs(gamma) + 1e-12) < tol:
            break
        gamma = gamma_new
    return W


def m_asym_ray_mse(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.1,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_asym_rayleigh(st, reg=reg)
    return _mse_refine(W, st, reg_mse=reg_mse)


def m_ray_asym_refine(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.1,
    reg_refine: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_rayleigh(st, reg=reg)
    return _asym_refine(W, st, reg_refine=reg_refine)


def m_ray_mse_asym_refine(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.01,
    reg_mse: float = 0.1,
    reg_refine: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_rayleigh(st, reg=reg)
    W = _mse_refine(W, st, reg_mse=reg_mse)
    return _asym_refine(W, st, reg_refine=reg_refine)


def m_ray_asym_refine_mse(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.1,
    reg_refine: float = 0.1,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_rayleigh(st, reg=reg)
    W = _asym_refine(W, st, reg_refine=reg_refine)
    return _mse_refine(W, st, reg_mse=reg_mse)


def m_ray_iterate(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.1,
    reg_mse: float = 0.1,
    reg_refine: float = 0.1,
    n_rounds: int = 3,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_rayleigh(st, reg=reg)
    for _ in range(n_rounds):
        W = _mse_refine(W, st, reg_mse=reg_mse)
        W = _asym_refine(W, st, reg_refine=reg_refine)
    return W


def m_split_rank_ray(
    st: Dict,
    neg: Optional[Dict] = None,
    frac_cross: float = 0.5,
    frac_total: float = 0.5,
    reg: float = 0.01,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    S_cross = st['Sigma_cross']
    S_total = st['Sigma_total']
    d = S_total.shape[0]
    d_cross = max(1, round(d * frac_cross))
    d_total = max(1, round(d * frac_total))

    ev_c, U_c = torch.linalg.eigh(S_cross)
    ev_c, U_c = ev_c.flip(0), U_c.flip(1)
    idx_c = torch.argsort(ev_c.abs(), descending=True)[:d_cross]
    S_cross_lr = U_c[:, idx_c] @ torch.diag(ev_c[idx_c]) @ U_c[:, idx_c].T

    ev_t, U_t = torch.linalg.eigh(S_total)
    ev_t, U_t = ev_t.flip(0), U_t.flip(1)
    S_total_lr = U_t[:, :d_total] @ torch.diag(ev_t[:d_total]) @ U_t[:, :d_total].T + reg * _eye(d)

    ev, evec = _eigh(S_cross_lr, S_total_lr)
    return evec[:, torch.argsort(ev, descending=True)]


def m_split_rank_ray_mse(
    st: Dict,
    neg: Optional[Dict] = None,
    frac_cross: float = 0.5,
    frac_total: float = 0.5,
    reg: float = 0.01,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_split_rank_ray(st, frac_cross=frac_cross, frac_total=frac_total, reg=reg)
    return _mse_refine(W, st, reg_mse=reg_mse)


def m_split_rank_ray_iterate(
    st: Dict,
    neg: Optional[Dict] = None,
    frac_cross: float = 0.5,
    frac_total: float = 0.5,
    reg: float = 0.01,
    reg_mse: float = 0.1,
    reg_refine: float = 0.1,
    n_rounds: int = 3,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    W = m_split_rank_ray(st, frac_cross=frac_cross, frac_total=frac_total, reg=reg)
    for _ in range(n_rounds):
        W = _mse_refine(W, st, reg_mse=reg_mse)
        W = _asym_refine(W, st, reg_refine=reg_refine)
    return W


def _uber_core(
    XTX_p: torch.Tensor,
    XTY_p: torch.Tensor,
    xm: torch.Tensor,
    alpha: float,
    reg: float,
    frac_low: float,
    Sigma_solve: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = XTX_p.shape[0]
    d_low = max(1, round(d * frac_low))
    s1 = XTX_p.abs().mean()
    s2 = XTY_p.abs().mean()
    XTX_n = XTX_p / s1
    XTY_n = XTY_p / s2
    diff = XTX_n - XTY_n + 1e-10 * _eye(d)
    Xcov = 0.5 * XTX_n + reg * _eye(d)
    L = torch.linalg.cholesky(Xcov)
    Li = torch.linalg.inv(L)
    ev, evec = torch.linalg.eigh(Li @ diff @ Li.T)
    idx = torch.argsort(ev)
    Wl = (Li.T @ evec)[:, idx[:d_low]]
    if Sigma_solve is None:
        Sigma_solve = XTX_p
    Wr = torch.linalg.solve(Sigma_solve + reg * _eye(d), XTY_p)
    Wrn = alpha * Wr / (Wr.norm() + 1e-10)
    Wln = (1 - alpha) * Wl / (Wl.norm() + 1e-10)
    W = torch.cat([Wrn, Wln], dim=1)
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    final = U @ torch.diag(S)
    pc = final.T @ Xcov @ final
    ev2, evec2 = torch.linalg.eigh(pc)
    P = evec2[:, torch.argsort(ev2, descending=True)]
    Wf = final @ P
    bias = -xm @ Wf
    return Wf, bias


def m_uber(
    st: Dict,
    neg: Optional[Dict] = None,
    alpha: float = 0.0,
    reg: float = 1e-4,
    frac_low: float = 0.33,
    **kw,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_st(st)
    return _uber_core(st['Sigma_total'], st['Sigma_cross'], st['X_mean'], alpha, reg, frac_low)


def _ray_contr_evd(
    st: Dict,
    neg: Dict,
    reg: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Contrastive Rayleigh eigendecomposition using negative pair cross-covariance."""
    d = st['Sigma_total'].shape[0]
    Sn = st['Sigma_cross'] - alpha * neg['Sigma_cross_neg']
    Sd = st['Sigma_total'] + beta * neg['Sigma_cross_neg'] + reg * _eye(d)
    ev, evec = _eigh(Sn, Sd)
    return evec[:, torch.argsort(ev, descending=True)]


def m_ray_contr_mse(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.01,
    alpha: float = 0.1,
    beta: float = 0.1,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    _check_neg(neg)
    W = _ray_contr_evd(st, neg, reg, alpha, beta)
    return _mse_refine(W, st, reg_mse=reg_mse)


def m_ray_contr_mse_neg(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.01,
    alpha: float = 0.1,
    beta: float = 0.1,
    alpha_mse: float = 0.1,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    _check_neg(neg)
    W = _ray_contr_evd(st, neg, reg, alpha, beta)
    k = W.shape[1]
    Sp = W.T @ st['Sigma_total'] @ W
    Ce = st['Sigma_cross'] - alpha_mse * neg['Sigma_cross_neg']
    Cp = W.T @ Ce @ W
    M = torch.linalg.solve(Sp + reg_mse * _eye(k), Cp)
    W2 = W @ M
    return _importance_sort(W2, st)


def m_resid_guided(
    st: Dict,
    neg: Optional[Dict] = None,
    reg: float = 0.01,
    gamma: float = 0.3,
    reg_mse: float = 0.1,
    **kw,
) -> torch.Tensor:
    _check_st(st)
    _check_neg(neg)
    d = st['Sigma_total'].shape[0]
    Sr = st['Sigma_total'] - st['Sigma_cross'] + reg * _eye(d)
    L = torch.linalg.cholesky(Sr)
    Li = torch.linalg.inv(L)
    Stw = Li @ st['Sigma_total'] @ Li.T
    Scw = Li @ st['Sigma_cross'] @ Li.T
    Snw = Li @ neg['Sigma_cross_neg'] @ Li.T
    Tw = (1 + gamma) * Scw - gamma * Snw
    Mw = torch.linalg.solve(Stw + reg_mse * _eye(d), Tw)
    U, S, _ = torch.linalg.svd(Mw, full_matrices=False)
    W = Li.T @ (U @ torch.diag(S))
    return _importance_sort(W, st)


def m_uber_neg(
    st: Dict,
    neg: Optional[Dict] = None,
    alpha: float = 0.0,
    reg: float = 1e-4,
    frac_low: float = 0.33,
    beta_neg: float = 0.1,
    **kw,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_st(st)
    _check_neg(neg)
    XTX_p = st['Sigma_total'] + beta_neg * neg['Sigma_cross_neg']
    return _uber_core(XTX_p, st['Sigma_cross'], st['X_mean'], alpha, reg, frac_low, Sigma_solve=st['Sigma_total'])


def m_uber_contr(
    st: Dict,
    neg: Optional[Dict] = None,
    alpha: float = 0.0,
    reg: float = 1e-4,
    frac_low: float = 0.33,
    alpha_neg: float = 0.1,
    beta_neg: float = 0.1,
    **kw,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_st(st)
    _check_neg(neg)
    XTX_p = st['Sigma_total'] + beta_neg * neg['Sigma_cross_neg']
    XTY_p = st['Sigma_cross'] - alpha_neg * neg['Sigma_cross_neg']
    return _uber_core(XTX_p, XTY_p, st['X_mean'], alpha, reg, frac_low, Sigma_solve=st['Sigma_total'])
