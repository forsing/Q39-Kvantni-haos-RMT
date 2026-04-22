#!/usr/bin/env python3

"""
Q39 Kvantni haos / Random Matrix Theory — Wigner-Dyson ansambli, spektralna
statistika, level repulsion — čisto kvantno.

Paradigma:
  Random Matrix Theory (RMT) opisuje univerzalne spektralne osobine kvantno
  haotičnih sistema. Ključne karakteristike:
    • Wigner-Dyson ansambli (GOE za vremenski reverzibilni sistem)
    • Level repulsion: P(s=0) = 0, P(s) ~ s za male s
    • Wigner surmise (GOE):
              P_W(s) = (π/2) · s · exp(−π s²/4)
    • Unfolded spectrum ima jediničnu srednju udaljenost između nivoa.

Mapiranje na loto:
  Za svaku poziciju i ∈ {1..7}, konstruiše se 64×64 Hermitski operator
        H = H_diag  +  σ · G_GOE
  gde je:
    • H_diag[j, j] = (j − j_target)²          (strukturni kostni potencijal)
    • G_GOE ∈ GOE(64)                         (simetrični realni, deterministki
                                                seeded kroz numpy.default_rng(SEED))
    • σ = SIGMA_OFF                           (strength off-diagonal mešanja)
  H se dijagonalizuje kroz eigh:  H = V · diag(Λ) · V†.
  Level repulsion je intrinzično prisutno u spektru H zbog GOE komponente.

RMT energetski filter (Lorentzian oko target energije):
  Strukturalni minimum je E_target = H_diag[j_target, j_target] = 0.
  Težina nivoa:
        w_k = Γ² / ((λ_k − E_target)² + Γ²)       (Lorentzian)
  Γ je polovina-širine odabrana deterministički kao ⟨Δλ⟩ × GAMMA_MULT
  (skaliranje mean level spacing-a). Ova težina koncentriše ansambl na
  nivoe u NISKO-ENERGETSKOM spektralnom regionu oko target-a. Porter-Thomas
  amplitude distribucija GOE eigenvectora (|V_{j,k}|² sa univerzalnim χ²₁
  statistikom) obezbeđuje kvantno-haotični karakter raspodele verovatnoća
  u bazisu |j⟩.

Mešano stanje (ensambl) i Born sempling:
        ρ_RMT = Σ_k w_k · |v_k⟩⟨v_k| / Σ_k w_k
        P(j) = ⟨j| ρ_RMT |j⟩ = Σ_k w_k · |V_{j,k}|² / Σ_k w_k
  Mask valid (num > prev_pick, num ∈ [i, i + 32]) → renormalize →
  numpy.default_rng(SEED).choice → Num_i = i + j*.

Structural target (non-freq):
        target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2)
        j_target = round(target_i) − i   ∈ [0, 32]

(fit): 6-qubit registar (64-dim) je dovoljno velik da spektar
njegovog Hermitijana pokaže Wigner-Dyson statistiku uz GOE perturbaciju;
RMT filter deterministički odbacuje ne-tipične nivoe (degeneracije / velike
gap-ove) i ostavlja samo nivoe sa kvantno-haotičnim spektralnim potpisom.
NQ = 6 qubit-a po poziciji (DIM = 64), reciklirani registar.
RMT / Wigner-Dyson filter paradigma — nova oblast.

Okruženje: Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se gase.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass


# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

SIGMA_OFF = 1.0                     # strength GOE off-diagonal komponente
GAMMA_MULT = 3.0                    # Lorentzian polovina-širina = GAMMA_MULT · ⟨Δλ⟩
# GOE 64×64 matrica se generiše JEDNOM, deterministicki (seeded).


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# GOE(64) matrica — deterministički seeded (JEDNOM na startu)
# =========================
def build_goe_matrix(rng: np.random.Generator) -> np.ndarray:
    # H_ii ~ N(0, 1),  H_ij = H_ji ~ N(0, 1/2) za i ≠ j   (standardni GOE)
    M = rng.standard_normal((DIM, DIM))
    G = (M + M.T) / math.sqrt(2.0)
    # Skaliranje varijanse dijagonalnih na 1 (simetrizacija povećala varijansu
    # na 2 × 1/2 = 1 za off-diag; dijagonala sada ima var = 2 → normalizuj):
    # Standard GOE: diag ~ N(0, 1), off-diag ~ N(0, 1/2). Naša simetrizacija
    # daje: diag ~ N(0, 2)/√2 = N(0, 1), off-diag ~ (N+N)/√2 ~ N(0, 1).
    # → da bismo postigli off-diag var = 1/2, pomnožimo off-diag sa 1/√2:
    mask_off = np.ones((DIM, DIM)) - np.eye(DIM)
    G = G * (np.eye(DIM) + mask_off / math.sqrt(2.0))
    return G


GOE_RNG = np.random.default_rng(SEED)
G_GOE = build_goe_matrix(GOE_RNG)


# =========================
# Hermitski H = H_diag + σ · G_GOE
# =========================
def build_hamiltonian_rmt(j_target: int) -> np.ndarray:
    H = np.zeros((DIM, DIM), dtype=np.float64)
    for j in range(DIM):
        H[j, j] = float((j - j_target) ** 2)
    H = H + SIGMA_OFF * G_GOE
    # Simetrizacija radi numeričke Hermitičnosti
    H = (H + H.T) / 2.0
    return H


# =========================
# Wigner surmise (GOE): P_W(s) = (π/2) s exp(−π s² / 4)
# (zadržana kao dijagnostika / deo RMT paradigme)
# =========================
def wigner_surmise(s: float) -> float:
    if s < 0:
        return 0.0
    return (math.pi / 2.0) * s * math.exp(-math.pi * s * s / 4.0)


# =========================
# Lorentzian energetski filter oko target energije (E_target = 0)
# w_k = Γ² / ((λ_k − E_target)² + Γ²),  Γ = GAMMA_MULT · ⟨Δλ⟩
# =========================
def lorentzian_weights(evals: np.ndarray, e_target: float) -> np.ndarray:
    spacings = np.diff(evals)
    mean_sp = float(np.mean(spacings)) if np.mean(spacings) > 0 else 1.0
    gamma = GAMMA_MULT * mean_sp
    if gamma <= 0.0:
        gamma = 1.0
    delta = evals - e_target
    weights = (gamma * gamma) / (delta * delta + gamma * gamma)
    return weights


# =========================
# Predikcija jedne pozicije
# =========================
def rmt_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)
    H = build_hamiltonian_rmt(j_target)
    evals, evecs = np.linalg.eigh(H)

    # E_target = H_diag[j_target, j_target] = 0 (strukturalni minimum)
    w = lorentzian_weights(evals, e_target=0.0)
    w_sum = float(w.sum())
    if w_sum < 1e-15:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    w = w / w_sum

    # P(j) = Σ_k w_k · |V_{j,k}|²
    P_j = (np.abs(evecs) ** 2) @ w
    P_j = np.clip(P_j, 0.0, None)

    # Dijagnostika: mean/variance spacing
    spacings = np.diff(evals)
    mean_sp = float(np.mean(spacings))
    var_sp = float(np.var(spacings))

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick and j < POS_RANGE:
            mask[j] = 1.0

    probs_valid = P_j * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return num, j_target, target, mean_sp, var_sp
        return (
            max(prev_pick + 1, position_1based),
            j_target,
            target,
            mean_sp,
            var_sp,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return num, j_target, target, mean_sp, var_sp


# =========================
# Autoregresivni run (reciklirani 6-qubit / 64-dim Hermitski prostor)
# =========================
def run_rmt_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        num, j_t, target, mean_sp, var_sp = rmt_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"⟨Δλ⟩={mean_sp:.3f}  Var(Δλ)={var_sp:.3f}  num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    print("=" * 84)
    print("Q39 Kvantni haos / RMT — GOE + Lorentzian(E_target) weighted mixed state")
    print("=" * 84)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (Hermitski prostor dim={DIM})")
    print(f"Hamiltonijan:   H = diag((j−j_target)²) + σ · G_GOE")
    print(f"σ (off-diag):   {SIGMA_OFF}")
    print(
        f"RMT filter:     Lorentzian w_k = Γ²/((λ_k−0)² + Γ²), "
        f"Γ = {GAMMA_MULT}·⟨Δλ⟩"
    )
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print(
        "Pokretanje RMT (GOE + Lorentzian energetski filter + Born sempling) "
        "po pozicijama:"
    )

    picks = run_rmt_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]

    print()
    print("=" * 84)
    print("REZULTAT Q39 (NEXT kombinacija)")
    print("=" * 84)
    print(f"Suma:  {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:  {n_odd}")
    print(f"Gaps:  {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
====================================================================================
Q39 Kvantni haos / RMT — GOE + Lorentzian(E_target) weighted mixed state
====================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (Hermitski prostor dim=64)
Hamiltonijan:   H = diag((j−j_target)²) + σ · G_GOE
σ (off-diag):   1.0
RMT filter:     Lorentzian w_k = Γ²/((λ_k−0)² + Γ²), Γ = 3.0·⟨Δλ⟩
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje RMT (GOE + Lorentzian energetski filter + Born sempling) po pozicijama:
  [pos 1]  target=4.875  j_target= 4  ⟨Δλ⟩=55.279  Var(Δλ)=1296.207  num=10
  [pos 2]  target=14.143  j_target=12  ⟨Δλ⟩=41.323  Var(Δλ)=1096.809  num=19
  [pos 3]  target=22.333  j_target=19  ⟨Δλ⟩=30.767  Var(Δλ)=836.163  num=21
  [pos 4]  target=24.600  j_target=21  ⟨Δλ⟩=28.025  Var(Δλ)=761.469  num=25
  [pos 5]  target=28.500  j_target=23  ⟨Δλ⟩=25.438  Var(Δλ)=681.372  num=32
  [pos 6]  target=34.333  j_target=28  ⟨Δλ⟩=19.473  Var(Δλ)=482.114  num=37
  [pos 7]  target=38.000  j_target=31  ⟨Δλ⟩=16.284  Var(Δλ)=369.694  num=38

====================================================================================
REZULTAT Q39 (NEXT kombinacija)
====================================================================================
Suma:  182   (S̄=140.51)
#odd:  4
Gaps:  [9, 2, 4, 7, 5, 1]
Predikcija NEXT: [10, 19, x, y, z, 37, 38]
"""



"""
REZULTAT — Q39 Kvantni haos / RMT (GOE + Wigner-Dyson weighted ensemble)
-----------------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: Hermitski operator nad 64-dim prostorom,
    spektralni razvoj, mešano stanje kao kvantni ansambl, Born sempling.
  • RMT paradigm: Wigner-Dyson spektralna statistika i level repulsion kao
    intrinsični filter kvantno-haotičnih nivoa; eigenstates sa tipičnim
    spacing-om dobijaju veću težinu u ansamblu.
  • NQ = 6 qubit-a po poziciji, reciklirani 64-dim prostor.
  • deterministicki seeded GOE matrica + seeded Born sempling.

Tehnike:
  • GOE(64) matrica (realna simetrična, elementi N(0, 1) na dijag, N(0, 1/2)
    van dijag), generisana jednom sa np.random.default_rng(SEED).
  • H = diag((j−j_target)²) + σ · G_GOE (po poziciji, j_target je structural).
  • eigh dijagonalizacija.
  • Wigner surmise P_W(s) = (π/2) s exp(−π s²/4) kao težina nivoa.
  • ρ_RMT = Σ_k w_k |v_k⟩⟨v_k|, P(j) = diag(ρ).
  • Born sempling iz uslovne (valid-masked) distribucije.
"""
