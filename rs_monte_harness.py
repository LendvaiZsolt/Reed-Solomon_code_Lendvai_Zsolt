"""Random RS(7,4) and RS(8,4) GF(9) harness logic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import galois
import numpy as np

import rs74_core as rc
import rs84_core_gf9 as g9

Branch = Literal["RS74", "RS84_GF9"]

# Branch weighting: RS(7,4) vs RS(8,4) GF(9) = 1:2.
HARNESS_WEIGHT_RS74 = 1.0 / 3.0

# TSV output separator.
HARNESS_FIELD_SEP = "\t"

# First-column branch labels.
BRANCH_FIELD_RS74 = "RS(7,4)"
BRANCH_FIELD_RS84_GF9 = "RS(8,4) GF(9)"


def harness_header_line() -> str:
    """Return TSV header row."""
    return HARNESS_FIELD_SEP.join(
        (
            "ág",
            "eredeti_m_m0_m1_m2_m3",
            "hiba_db",
            "hiba_pozíciók_j",
            "fogadott_r",
            "javított_c",
            "eredmény_TEST",
        )
    )


def _branch_field(b: Branch) -> str:
    return BRANCH_FIELD_RS74 if b == "RS74" else BRANCH_FIELD_RS84_GF9


F9 = g9.F
ALPHA9 = F9.primitive_element
N84, K84 = g9.N, g9.K


def _build_g_vandermonde_gf9() -> galois.FieldArray:
    g_mat = F9.Zeros((K84, N84))
    for i in range(K84):
        for j in range(N84):
            g_mat[i, j] = ALPHA9 ** (i * j)
    return g_mat


def _build_h_eval_transpose_gf9() -> galois.FieldArray:
    h_t = F9.Zeros((N84, N84 - K84))
    for v in range(N84):
        for j in range(N84 - K84):
            h_t[v, j] = ALPHA9 ** (v * (j + 1))
    return h_t


_G84 = _build_g_vandermonde_gf9()
_H_T84 = _build_h_eval_transpose_gf9()


def _format_int_row(vec: list[int]) -> str:
    return "[" + ", ".join(str(int(x)) for x in vec) + "]"


def _rs74_one_case(rng: np.random.Generator) -> tuple[str, str, str, str, str, str]:
    chosen = [str(rng.choice(rc.LETTER_ORDER)) for _ in range(3)]
    pad_int = 0
    m_vals = [rc.letter_to_gf_int(ch) for ch in chosen] + [pad_int]
    parity_right = bool(rng.integers(0, 2))
    G, H = rc.permute_columns_parity_order(rc.G_BASE, rc.H_BASE, parity_right)
    m = rc.GF(m_vals).reshape(1, rc.K)
    c = m @ G
    corrupt = bool(rng.integers(0, 2))
    err_positions: list[int] = []
    r = c.copy()
    inj_direct = bool(rng.integers(0, 2))
    num_err = 0
    if corrupt:
        num_err = 1
        pos = int(rng.integers(0, rc.N))
        err_positions = [pos]
        if inj_direct:
            cj = int(c[0, pos])
            recv = int(rng.integers(0, 8))
            if recv == cj:
                recv = (recv + 1) % 8
            r[0, pos] = rc.GF(recv)
        else:
            em = int(rng.integers(1, 8))
            r[0, pos] = r[0, pos] + rc.GF(em)
    r_dec_ints = rc.gf_row_to_ints(r)
    c_ints = rc.gf_row_to_ints(c)
    orig_m_str = ",".join(str(int(x)) for x in m_vals)
    s_dec = rc.syndrome_row(r, H)
    j_hat, a_hat = rc.single_error_from_syndrome(s_dec.flatten(), H)
    if j_hat is not None:
        e_manual = rc.GF([0] * rc.N)
        e_manual[j_hat] = a_hat
        c_manual = r.flatten() - e_manual
        c_hat_ints = rc.gf_row_to_ints(c_manual.reshape(1, -1))
    elif np.all(s_dec == 0):
        c_hat_ints = list(r_dec_ints)
    else:
        c_hat_ints = list(r_dec_ints)
    ok = c_hat_ints == c_ints
    pos_str = ",".join(str(p) for p in err_positions) if err_positions else ""
    tag = "TEST_TRUE" if ok else "TEST_FALSE"
    return (
        orig_m_str,
        str(num_err),
        pos_str,
        _format_int_row(r_dec_ints),
        _format_int_row(c_hat_ints),
        tag,
    )


def _rs84_one_case(rng: np.random.Generator) -> tuple[str, str, str, str, str, str]:
    m_vals = [int(rng.integers(0, 9)) for _ in range(K84)]
    m = F9(m_vals).reshape(1, K84)
    c = m @ _G84
    corrupt = bool(rng.integers(0, 2))
    err_positions: list[int] = []
    r = c.copy()
    inj_direct = bool(rng.integers(0, 2))
    num_err = 0
    if corrupt:
        num_err = int(rng.integers(1, 3))
        positions = sorted(rng.choice(N84, size=num_err, replace=False).tolist())
        err_positions = positions
        for pos in positions:
            cj = int(c[0, pos])
            if inj_direct:
                recv = int(rng.integers(0, 9))
                if recv == cj:
                    recv = (recv + 1) % 9
                r[0, pos] = F9(recv)
            else:
                em = int(rng.integers(1, 9))
                r[0, pos] = r[0, pos] + F9(em)
    r_ints = [int(x) for x in np.asarray(r).flatten()][:N84]
    c_ints = [int(x) for x in np.asarray(c).flatten()][:N84]
    orig_m_str = ",".join(str(int(x)) for x in m_vals)
    dec = g9.decode_rs84(r_ints, G_mat=_G84, H_T_mat=_H_T84)
    if dec is not None:
        c_hat_ints, _e = dec
    else:
        c_hat_ints = list(r_ints)
    ok = c_hat_ints == c_ints
    pos_str = ",".join(str(p) for p in err_positions) if err_positions else ""
    tag = "TEST_TRUE" if ok else "TEST_FALSE"
    return (
        orig_m_str,
        str(num_err),
        pos_str,
        _format_int_row(r_ints),
        _format_int_row(c_hat_ints),
        tag,
    )


def run_random_batch(
    *,
    n_tests: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, str, str, str, str, str, str]]:
    """Generate random harness rows with weighted branch selection."""
    if rng is None:
        rng = np.random.default_rng()
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for _ in range(n_tests):
        branch: Branch = (
            "RS74" if rng.random() < HARNESS_WEIGHT_RS74 else "RS84_GF9"
        )
        base = _rs74_one_case(rng) if branch == "RS74" else _rs84_one_case(rng)
        rows.append((_branch_field(branch),) + base)
    return rows


def default_output_path(*, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent
    return root / "exports" / "rs_streamlit_random_harness.tsv"


def write_harness_file(path: Path, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [HARNESS_FIELD_SEP.join(t) for t in rows]
    lines = [harness_header_line(), *body]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
