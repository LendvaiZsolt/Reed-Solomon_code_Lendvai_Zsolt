from __future__ import annotations

from typing import Optional, Tuple

import galois
import numpy as np

LETTER_ORDER = tuple('ABCDEFGH')
LETTER_BITS: dict[str, str] = {'A': '000', 'B': '001', 'C': '010', 'D': '011', 'E': '100', 'F': '101', 'G': '110', 'H': '111'}
INT_TO_ALPHA_STR: dict[int, str] = {0: '0', 1: '1', 2: 'α', 3: 'α+1', 4: 'α²', 5: 'α²+1', 6: 'α²+α', 7: 'α²+α+1'}
INT_TO_ALPHA_POWER_STR: dict[int, str] = {0: '0', 1: 'α⁰', 2: 'α¹', 3: 'α³', 4: 'α²', 5: 'α⁶', 6: 'α⁴', 7: 'α⁵'}


def letter_to_gf_int(ch: str) -> int:
    return int(LETTER_BITS[ch], 2)


def int_to_bits3(v: int) -> str:
    return f'{v & 7:03b}'


def bits21_c0_to_c6(c_ints: list[int]) -> str:
    return ''.join((int_to_bits3(v) for v in c_ints))


def bits21_spaced_c0_to_c6(c_ints: list[int]) -> str:
    return ' '.join((int_to_bits3(v) for v in c_ints))


def bits21_c6_to_c0(c_ints: list[int]) -> str:
    return ''.join((int_to_bits3(v) for v in reversed(c_ints)))


def bracket_groups_bits(c_ints: list[int], *, descending: bool) -> str:
    seq = list(reversed(c_ints)) if descending else c_ints
    return '[' + ','.join((int_to_bits3(v) for v in seq)) + ']'


def gf8_element_table_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for ch in LETTER_ORDER:
        v = letter_to_gf_int(ch)
        rows.append({'Karakter': ch, '3 bites kód': LETTER_BITS[ch], 'Polinom alak': INT_TO_ALPHA_STR[v], 'α hatvány alak': INT_TO_ALPHA_POWER_STR[v], 'GF(8) int (0–7)': v})
    return rows


def gf_int_to_labels(v: int) -> dict[str, str]:
    return {'int (0–7)': str(int(v)), 'Bits (b₂b₁b₀)': int_to_bits3(v), 'α alak': INT_TO_ALPHA_STR[int(v) & 7]}


def gf_symbol_select_label(v: int) -> str:
    x = int(v) & 7
    return f'{x} — {int_to_bits3(x)} — {INT_TO_ALPHA_STR[x]}'


def build_h_systematic_from_g(G: galois.FieldArray, *, parity_right: bool) -> galois.FieldArray:
    F = type(G)
    k, n = (int(G.shape[0]), int(G.shape[1]))
    nmk = n - k
    H = F.Zeros((nmk, n))
    if parity_right:
        P = G[:, k:n]
        H[:, :k] = P.T
        H[:, k:n] = F.Identity(nmk)
    else:
        P = G[:, :nmk]
        H[:, :nmk] = F.Identity(nmk)
        H[:, nmk:n] = P.T
    return H


def build_gh_from_generator_polynomial() -> tuple[galois.FieldArray, galois.FieldArray]:
    n, k = (7, 4)
    F = galois.GF(2 ** 3)
    alpha = F.primitive_element
    ONE = F(1)
    Z = F(0)
    g0 = alpha ** 2 + ONE
    g1 = alpha
    g2 = alpha ** 2 + ONE
    g_poly = galois.Poly([g0, g1, g2, ONE], field=F, order='asc')
    x3 = galois.Poly([Z, Z, Z, ONE], field=F, order='asc')
    rows: list[list[int]] = []
    for i in range(k):
        coeffs = [Z, Z, Z, Z]
        coeffs[i] = ONE
        m_poly = galois.Poly(coeffs, field=F, order='asc')
        t = m_poly * x3
        r = t % g_poly
        cp = t + r
        full = [int(x) for x in cp.coefficients(order='asc', size=n)]
        c_sys = full[3:7] + full[0:3]
        rows.append(c_sys)
    G = F(rows)
    H = build_h_systematic_from_g(G, parity_right=True)
    return (G, H)


def permute_columns_parity_order(G_base: galois.FieldArray, H_base: galois.FieldArray, parity_right: bool) -> tuple[galois.FieldArray, galois.FieldArray]:
    if parity_right:
        return (G_base, H_base)
    n, k = (G_base.shape[1], G_base.shape[0])
    old_order = list(range(k, n)) + list(range(k))
    F = type(G_base)
    Pi = F.Zeros((n, n))
    for j in range(n):
        Pi[old_order[j], j] = F(1)
    return (G_base @ Pi, H_base @ Pi)


GF = galois.GF(2 ** 3)
N, K = (7, 4)
SHOW_KODOLAS_C6_DESCENDING = False
SHOW_KODOLAS_CI_INT_LIST = False
G_BASE, H_BASE = build_gh_from_generator_polynomial()


def gf_row_to_ints(row: np.ndarray) -> list[int]:
    return [int(x) for x in np.asarray(row).flatten()]


def syndrome_row(r: galois.FieldArray, H_mat: galois.FieldArray) -> galois.FieldArray:
    r = GF(r).reshape(1, -1)
    return r @ H_mat.T


def single_error_from_syndrome(s: galois.FieldArray, H_mat: galois.FieldArray) -> Tuple[Optional[int], Optional[galois.FieldArray]]:
    s = GF(s).flatten()
    if np.all(s == 0):
        return (None, None)
    for j in range(N):
        col = H_mat[:, j]
        nz = np.nonzero(col)[0]
        if len(nz) == 0:
            continue
        i0 = int(nz[0])
        if int(col[i0]) == 0:
            continue
        a = GF(s[i0]) / GF(col[i0])
        if np.array_equal(s, a * col):
            return (j, a)
    return (None, None)


def format_int_row(vec: list[int]) -> str:
    return '[' + ', '.join((str(int(x)) for x in vec)) + ']'


def format_gf8_symbols_as_bits(vec: list[int]) -> str:
    return '[' + ', '.join((int_to_bits3(int(x) & 7) for x in vec)) + ']'


def format_gf8_int_tuple(vec: list[int]) -> str:
    return '(' + ', '.join((str(int(x) & 7) for x in vec)) + ')'


def format_gf8_bits_tuple(vec: list[int]) -> str:
    return '(' + ', '.join((int_to_bits3(int(x) & 7) for x in vec)) + ')'


def format_r_epsilon_hat_c_aligned_block(r_vals: list[int], epsilon_hat_vals: list[int], c_hat_vals: list[int]) -> str:
    t_r = format_gf8_int_tuple(r_vals)
    t_e = format_gf8_int_tuple(epsilon_hat_vals)
    t_c = format_gf8_int_tuple(c_hat_vals)
    p1 = 'r = '
    p2 = 'ε̂ = '
    p3 = 'r-ε̂=c = '
    w = max(len(p1), len(p2), len(p3))
    return '\n'.join([p1.rjust(w - 1) + t_r, p2.rjust(w) + t_e, p3.rjust(w) + t_c])
