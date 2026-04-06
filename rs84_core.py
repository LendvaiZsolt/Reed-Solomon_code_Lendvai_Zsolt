from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

import galois
import numpy as np

IRREDUCIBLE_DESC = 'x⁴ + x + 1'
F = galois.GF(2**4, irreducible_poly='x^4+x+1')
N, K = 8, 4


def _superscript_digit(d: int) -> str:
    return '⁰¹²³⁴⁵⁶⁷⁸⁹'[d % 10]


def _superscript_int(n: int) -> str:
    if n == 0:
        return '⁰'
    s = ''
    while n:
        s = _superscript_digit(n % 10) + s
        n //= 10
    return s


def _build_alpha_power_labels() -> dict[int, str]:
    alpha = F.primitive_element
    d: dict[int, str] = {0: '0'}
    for k in range(15):
        d[int(alpha**k)] = 'α' + _superscript_int(k)
    return d


INT_TO_ALPHA_POWER_STR: dict[int, str] = _build_alpha_power_labels()


def _poly_label_for_int(v: int) -> str:
    if v == 0:
        return '0'
    p = galois.Poly(F(v).vector(), field=galois.GF2)
    return str(p).replace('x', 'α')


def _build_poly_labels() -> dict[int, str]:
    return {i: _poly_label_for_int(i) for i in range(16)}


INT_TO_ALPHA_STR: dict[int, str] = _build_poly_labels()

LETTER_ORDER = tuple('ABCDEFGHIJKLMNOP')
LETTER_BITS: dict[str, str] = {ch: f'{i:04b}' for i, ch in enumerate(LETTER_ORDER)}


def letter_to_gf_int(ch: str) -> int:
    return int(LETTER_BITS[ch], 2)


def int_to_bits4(v: int) -> str:
    return f'{int(v) & 15:04b}'


def bits32_c0_to_c7(c_ints: list[int]) -> str:
    return ''.join((int_to_bits4(v) for v in c_ints))


def bits32_spaced_c0_to_c7(c_ints: list[int]) -> str:
    return ' '.join((int_to_bits4(v) for v in c_ints))


def bits32_c7_to_c0(c_ints: list[int]) -> str:
    return ''.join((int_to_bits4(v) for v in reversed(c_ints)))


def bracket_groups_bits(c_ints: list[int], *, descending: bool) -> str:
    seq = list(reversed(c_ints)) if descending else c_ints
    return '[' + ','.join((int_to_bits4(v) for v in seq)) + ']'


def gf16_element_table_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for ch in LETTER_ORDER:
        v = letter_to_gf_int(ch)
        rows.append({'Karakter': ch, '4 bites kód': LETTER_BITS[ch], 'Polinom alak': INT_TO_ALPHA_STR[v], 'α hatvány alak': INT_TO_ALPHA_POWER_STR[v], 'GF(16) int (0–15)': v})
    return rows


def gf_int_to_labels(v: int) -> dict[str, str]:
    v = int(v) & 15
    return {'int (0–15)': str(v), 'Bits (b₃b₂b₁b₀)': int_to_bits4(v), 'α alak': INT_TO_ALPHA_STR[v]}


def gf_symbol_select_label(v: int) -> str:
    x = int(v) & 15
    return f'{x} — {int_to_bits4(x)} — {INT_TO_ALPHA_STR[x]}'


def build_narrow_sense_generator_polynomial() -> galois.Poly:
    n, k = N, K
    alpha = F.primitive_element
    ONE = F(1)
    g_poly = galois.Poly([ONE], field=F)
    for j in range(1, n - k + 1):
        fac = galois.Poly([alpha**j, ONE], field=F, order='asc')
        g_poly = g_poly * fac
    return g_poly


def build_h_systematic_from_g(G: galois.FieldArray, *, parity_right: bool) -> galois.FieldArray:
    F_ = type(G)
    k_, n_ = (int(G.shape[0]), int(G.shape[1]))
    nmk = n_ - k_
    H = F_.Zeros((nmk, n_))
    if parity_right:
        P = G[:, k_:n_]
        H[:, :k_] = P.T
        H[:, k_:n_] = F_.Identity(nmk)
    else:
        P = G[:, :nmk]
        H[:, :nmk] = F_.Identity(nmk)
        H[:, nmk:n_] = P.T
    return H


def permute_columns_parity_order(G_base: galois.FieldArray, H_base: galois.FieldArray, parity_right: bool) -> tuple[galois.FieldArray, galois.FieldArray]:
    return (G_base, H_base)


@functools.lru_cache(maxsize=1)
def rs84_reedsolo_codec():
    from reedsolo import RSCodec

    return RSCodec(nsym=N - K, nsize=N, c_exp=4, fcr=1, generator=2, prim=19, single_gen=True)


def reedsolo_encode_systematic_m(m4: list[int]) -> list[int]:
    rsc = rs84_reedsolo_codec()
    msg = bytearray((int(x) & 15 for x in m4[:K]))
    return [int(b) & 15 for b in rsc.encode(msg)]


def build_g_rs_and_h_from_reedsolo() -> tuple[galois.FieldArray, galois.FieldArray]:
    rows: list[list[int]] = []
    for i in range(K):
        m = [0] * K
        m[i] = 1
        rows.append(reedsolo_encode_systematic_m(m))
    G = F(rows)
    H = build_h_systematic_from_g(G, parity_right=True)
    return (G, H)


G_POLY = build_narrow_sense_generator_polynomial()
G_RS, H_RS = build_g_rs_and_h_from_reedsolo()
G_BASE, H_BASE = G_RS, H_RS
SHOW_KODOLAS_C7_DESCENDING = False
SHOW_KODOLAS_CI_INT_LIST = False
_C_SUBSCRIPT = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇')


def gf_row_to_ints(row: np.ndarray) -> list[int]:
    return [int(x) for x in np.asarray(row).flatten()]


def _t_r_cp_polys_from_message(m_vals: list[int]) -> tuple[galois.Poly, galois.Poly, galois.Poly]:
    Z = F(0)
    m_poly = galois.Poly([F(int(m_vals[i]) & 15) for i in range(4)], field=F, order='asc')
    xnk = galois.Poly([Z] * (N - K) + [F(1)], field=F, order='asc')
    t = m_poly * xnk
    r = t % G_POLY
    cp = t + r
    return (t, r, cp)


def format_int_row(vec: list[int]) -> str:
    return '[' + ', '.join((str(int(x)) for x in vec)) + ']'


def format_gf16_symbols_as_bits(vec: list[int]) -> str:
    return '[' + ', '.join((int_to_bits4(int(x) & 15) for x in vec)) + ']'


def format_gf16_int_tuple(vec: list[int]) -> str:
    return '(' + ', '.join((str(int(x) & 15) for x in vec)) + ')'


def format_gf16_bits_tuple(vec: list[int]) -> str:
    return '(' + ', '.join((int_to_bits4(int(x) & 15) for x in vec)) + ')'


def format_r_epsilon_hat_c_aligned_block(r_vals: list[int], epsilon_hat_vals: list[int], c_hat_vals: list[int]) -> str:
    t_r = format_gf16_int_tuple(r_vals)
    t_e = format_gf16_int_tuple(epsilon_hat_vals)
    t_c = format_gf16_int_tuple(c_hat_vals)
    p1 = 'r = '
    p2 = 'ε̂ = '
    p3 = 'r-ε̂=c = '
    w = max(len(p1), len(p2), len(p3))
    return '\n'.join([p1.rjust(w - 1) + t_r, p2.rjust(w) + t_e, p3.rjust(w) + t_c])


def syndrome_row(r: galois.FieldArray, H_mat: galois.FieldArray) -> galois.FieldArray:
    r = F(r).reshape(1, -1)
    return r @ H_mat.T


def vector_index_to_poly_degree(v: int) -> int:
    return (N - 1 - int(v)) % N


def poly_degree_to_vector_index(d: int) -> int:
    return (N - 1 - int(d)) % N


def compute_evaluation_syndromes_fields(r_ints: list[int]) -> list[galois.FieldArray]:
    alpha = F.primitive_element
    out: list[galois.FieldArray] = []
    for j in range(1, N - K + 1):
        s = F(0)
        aj = alpha**j
        for v in range(N):
            d = vector_index_to_poly_degree(v)
            s += F(int(r_ints[v]) & 15) * (aj**d)
        out.append(s)
    return out


def compute_evaluation_syndromes_ints(r_ints: list[int]) -> list[int]:
    return [int(x) for x in compute_evaluation_syndromes_fields(r_ints)]


def _gf_det2(a11: galois.FieldArray, a12: galois.FieldArray, a21: galois.FieldArray, a22: galois.FieldArray) -> galois.FieldArray:
    return a11 * a22 + a12 * a21


def _gf_solve_1x1(a: galois.FieldArray, b: galois.FieldArray) -> galois.FieldArray:
    if a == 0:
        raise ValueError('szingularis 1x1')
    return b / a


def _gf_solve_2x2(A: galois.FieldArray, b: galois.FieldArray) -> tuple[galois.FieldArray, galois.FieldArray]:
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    d = _gf_det2(a11, a12, a21, a22)
    if d == 0:
        raise ValueError('szingularis 2x2')
    x1 = (a22 * b[0] + a12 * b[1]) / d
    x2 = (a21 * b[0] + a11 * b[1]) / d
    return (x1, x2)


def determine_num_errors_hankel(S: list[galois.FieldArray]) -> int:
    S1, S2, S3, S4 = S[0], S[1], S[2], S[3]
    delta2 = _gf_det2(S1, S2, S2, S3)
    return 2 if delta2 != 0 else 1


def compute_locator_coefficients(S: list[galois.FieldArray], m: int) -> list[galois.FieldArray]:
    S1, S2, S3, S4 = S[0], S[1], S[2], S[3]
    if m == 1:
        L1 = _gf_solve_1x1(S1, S2)
        return [F(1), L1]
    A = F([[S2, S1], [S3, S2]])
    b = F([S3, S4])
    L1, L2 = _gf_solve_2x2(A, b)
    return [F(1), L1, L2]


def compute_pgz_hankel_state(r_ints: list[int]) -> dict[str, Any]:
    r_clean = [int(x) & 15 for x in r_ints[:N]]
    S = compute_evaluation_syndromes_fields(r_clean)
    S_ints = [int(x) for x in S]
    t = 2
    if all(x == 0 for x in S_ints):
        return {'all_S_zero': True, 'r_ints': r_clean, 'S_ints': S_ints, 't': t, 'm': 0}
    S1, S2, S3, S4 = S[0], S[1], S[2], S[3]
    det_h2 = int(_gf_det2(S1, S2, S2, S3))
    det_h2_zero = det_h2 == 0
    m = 1 if det_h2_zero else 2
    det_h1 = int(S1)
    step3: dict[str, Any] | None = None
    if m >= 1:
        try:
            loc = compute_locator_coefficients(S, m)
            if m == 1:
                L1 = int(loc[1])
                step3 = {'A': [[int(S1)]], 'b': [int(S2)], 'x': [L1], 'L_asc': [1, L1]}
            else:
                L1, L2 = int(loc[1]), int(loc[2])
                step3 = {
                    'A': [[int(S1), int(S2)], [int(S2), int(S3)]],
                    'b': [int(S3), int(S4)],
                    'x': [L2, L1],
                    'L_asc': [1, L1, L2],
                }
        except (ValueError, ZeroDivisionError):
            step3 = {'error': True}
    return {
        'all_S_zero': False,
        'r_ints': r_clean,
        'S_ints': S_ints,
        't': t,
        'det_h2': det_h2,
        'det_h2_zero': det_h2_zero,
        'det_h1': det_h1,
        'm': m,
        'step3': step3,
    }


def find_locator_ix_pairs(locator_asc: list[int]) -> list[tuple[int, int, int]]:
    coeffs_f = [F(int(c) & 15) for c in locator_asc]
    Lp = galois.Poly(coeffs_f, field=F, order='asc')
    alpha = F.primitive_element
    triples: list[tuple[int, int, int]] = []
    for val in range(16):
        zf = F(val)
        if Lp(zf) != F(0) or val == 0:
            continue
        xi = int(F(1) / zf)
        d_found: int | None = None
        for d in range(15):
            if int(alpha**d) == xi:
                d_found = d
                break
        if d_found is None:
            continue
        v = poly_degree_to_vector_index(d_found)
        triples.append((xi, d_found, v))
    triples.sort(key=lambda t: t[2])
    return triples


def compute_error_values_vandermonde(S: list[galois.FieldArray], X: list[galois.FieldArray]) -> list[galois.FieldArray]:
    m = len(X)
    if m == 0:
        return []
    if m == 1:
        return [S[0] / X[0]]
    A = F([[X[0], X[1]], [X[0] ** 2, X[1] ** 2]])
    b = F([S[0], S[1]])
    y1, y2 = _gf_solve_2x2(A, b)
    return [y1, y2]


def _locator_roots_positions_and_X(locator_asc: list[galois.FieldArray]) -> tuple[list[int], list[galois.FieldArray]] | None:
    Lp = galois.Poly(locator_asc, field=F, order='asc')
    alpha = F.primitive_element
    pairs: list[tuple[int, galois.FieldArray]] = []
    for x in F.elements:
        if Lp(F(x)) != 0:
            continue
        r = F(x)
        if r == 0:
            return None
        xv = F(1) / r
        d_found: int | None = None
        for d in range(15):
            if alpha**d == xv:
                d_found = d
                break
        if d_found is None:
            return None
        v = poly_degree_to_vector_index(d_found)
        pairs.append((v, xv))
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        return None
    positions = [t[0] for t in pairs]
    xs = [t[1] for t in pairs]
    return (positions, xs)


def decode_rs84(r_ints: list[int]) -> tuple[list[int], list[int]] | None:
    r_field = [F(int(x) & 15) for x in r_ints]
    r_ints_f = [int(x) for x in r_field]
    S = compute_evaluation_syndromes_fields(r_ints_f)
    if all(s == F(0) for s in S):
        return (r_ints_f, [0] * N)
    m = determine_num_errors_hankel(S)
    try:
        locator = compute_locator_coefficients(S, m)
        px = _locator_roots_positions_and_X(locator)
    except (ValueError, ZeroDivisionError):
        return None
    if px is None:
        return None
    positions, x_list = px
    if len(positions) != m or len(set(positions)) != m:
        return None
    try:
        y_vals = compute_error_values_vandermonde(S, x_list)
    except (ValueError, ZeroDivisionError):
        return None
    ev = [0] * N
    for pos, yv in zip(positions, y_vals, strict=True):
        ev[int(pos)] = int(yv) & 15
    c = [int(r_field[p] + F(ev[p])) for p in range(N)]
    for p in range(N):
        c[p] &= 15
    S_check = compute_evaluation_syndromes_fields(c)
    if any(s != F(0) for s in S_check):
        return None
    return (c, ev)


def reedsolo_apply_error_vector(cw8: list[int], e8: list[int]) -> list[int]:
    return [int(F(int(cw8[i]) & 15) + F(int(e8[i]) & 15)) & 15 for i in range(N)]


def reedsolo_decode_try(r8: list[int]) -> tuple[list[int], list[int], list[int]] | None:
    from reedsolo import ReedSolomonError

    rsc = rs84_reedsolo_codec()
    try:
        dmsg, dfull, erra = rsc.decode(bytearray((int(x) & 15 for x in r8[:N])))
    except ReedSolomonError:
        return None
    return (
        [int(x) & 15 for x in dmsg],
        [int(x) & 15 for x in dfull],
        [int(x) for x in erra],
    )


def reedsolo_package_version() -> str:
    try:
        from importlib.metadata import version

        return version('reedsolo')
    except Exception:
        return '?'


def single_error_from_syndrome(s: galois.FieldArray, H_mat: galois.FieldArray) -> Tuple[Optional[int], Optional[galois.FieldArray]]:
    s = F(s).flatten()
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
        a = F(s[i0]) / F(col[i0])
        if np.array_equal(s, a * col):
            return (j, a)
    return (None, None)

