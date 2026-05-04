"""RS(8,4) segédfüggvények GF(9) felett — PGZ / szindróma, dekódolás.

A szindrómasor megegyezik a ``pages/2_rs84.py`` **r @ Hᵀ** definíciójával (Hᵀ[v,j] = α^(v·(j+1))).
A ``vector_index_to_poly_degree`` a lokátor–pozíció leképezéshez marad (gyökök).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import galois
import numpy as np

F = galois.GF(3**2, irreducible_poly="x^2 + 1")
N, K = 8, 4


def format_int_row(vec: list[int]) -> str:
    return "[" + ", ".join(str(int(x)) for x in vec) + "]"


def vector_index_to_poly_degree(v: int) -> int:
    return (N - 1 - int(v)) % N


def poly_degree_to_vector_index(d: int) -> int:
    return (N - 1 - int(d)) % N


def compute_evaluation_syndromes_fields(r_ints: list[int]) -> list[galois.FieldArray]:
    """s[j] = Σ_v r_v·α^(v·(j+1)), j=0…N−K−1 — ugyanaz, mint a 2_rs84 Hᵀ melletti r @ Hᵀ."""
    alpha = F.primitive_element
    out: list[galois.FieldArray] = []
    for j in range(N - K):
        s = F(0)
        k = j + 1
        for v in range(N):
            s += F(int(r_ints[v]) % 9) * (alpha ** (v * k))
        out.append(s)
    return out


def compute_evaluation_syndromes_ints(r_ints: list[int]) -> list[int]:
    return [int(x) for x in compute_evaluation_syndromes_fields(r_ints)]


def syndrome_row(r: galois.FieldArray, H_mat: galois.FieldArray) -> galois.FieldArray:
    r = F(r).reshape(1, -1)
    return r @ H_mat.T


def _gf_det2(a11: galois.FieldArray, a12: galois.FieldArray, a21: galois.FieldArray, a22: galois.FieldArray) -> galois.FieldArray:
    return a11 * a22 - a12 * a21


def _gf_solve_1x1(a: galois.FieldArray, b: galois.FieldArray) -> galois.FieldArray:
    if a == 0:
        raise ValueError("szingularis 1x1")
    return b / a


def _gf_solve_2x2(A: galois.FieldArray, b: galois.FieldArray) -> tuple[galois.FieldArray, galois.FieldArray]:
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    d = _gf_det2(a11, a12, a21, a22)
    if d == 0:
        raise ValueError("szingularis 2x2")
    x1 = (a22 * b[0] - a12 * b[1]) / d
    x2 = (a11 * b[1] - a21 * b[0]) / d
    return (x1, x2)


def determine_num_errors_hankel(S: list[galois.FieldArray]) -> int:
    s1, s2, s3, s4 = S[0], S[1], S[2], S[3]
    delta2 = _gf_det2(s1, s2, s2, s3)
    return 2 if delta2 != 0 else 1


def compute_locator_coefficients(S: list[galois.FieldArray], m: int) -> list[galois.FieldArray]:
    """Peterson–Hankel: m=1 esetén S1·L1 = −S2; m=2 esetén [[S1,S2],[S2,S3]]·[L2,L1]^T = [−S3,−S4]^T."""
    s1, s2, s3, s4 = S[0], S[1], S[2], S[3]
    if m == 1:
        # S1 * L1 = -S2  (nem char 2: $-S_2 \\neq S_2$ általában)
        l1 = _gf_solve_1x1(s1, -s2)
        return [F(1), l1]
    # [S1 S2; S2 S3] [L2; L1] = [-S3; -S4]
    a_mat = F([[s1, s2], [s2, s3]])
    b_vec = F([-s3, -s4])
    l2, l1 = _gf_solve_2x2(a_mat, b_vec)
    return [F(1), l1, l2]


def compute_pgz_hankel_state(r_ints: list[int]) -> dict[str, Any]:
    r_clean = [int(x) % 9 for x in r_ints[:N]]
    s_fields = compute_evaluation_syndromes_fields(r_clean)
    s_ints = [int(x) for x in s_fields]
    t = 2
    if all(x == 0 for x in s_ints):
        return {"all_S_zero": True, "r_ints": r_clean, "S_ints": s_ints, "t": t, "m": 0}
    s1, s2, s3, s4 = s_fields[0], s_fields[1], s_fields[2], s_fields[3]
    det_h2 = int(_gf_det2(s1, s2, s2, s3))
    det_h2_zero = det_h2 == 0
    m = 1 if det_h2_zero else 2
    det_h1 = int(s1)
    step3: dict[str, Any] | None = None
    if m >= 1:
        try:
            loc = compute_locator_coefficients(s_fields, m)
            if m == 1:
                l1 = int(loc[1])
                step3 = {
                    "A": [[int(s1)]],
                    "b": [int(-s2)],
                    "x": [l1],
                    "L_asc": [1, l1],
                }
            else:
                l1, l2 = int(loc[1]), int(loc[2])
                step3 = {
                    "A": [[int(s1), int(s2)], [int(s2), int(s3)]],
                    "b": [int(-s3), int(-s4)],
                    "x": [l2, l1],
                    "L_asc": [1, l1, l2],
                }
        except (ValueError, ZeroDivisionError):
            step3 = {"error": True}
    return {
        "all_S_zero": False,
        "r_ints": r_clean,
        "S_ints": s_ints,
        "t": t,
        "det_h2": det_h2,
        "det_h2_zero": det_h2_zero,
        "det_h1": det_h1,
        "m": m,
        "step3": step3,
    }


def gf9_sqrt_element(d: galois.FieldArray) -> galois.FieldArray | None:
    """Ha létezik t ∈ GF(9), hogy t² = d, visszaad egy ilyen t-t; különben None."""
    d = F(d)
    if d == 0:
        return F(0)
    for t in F.elements:
        if t * t == d:
            return F(t)
    return None


def locator_quadratic_solution_gf9(l1: int, l2: int) -> dict[str, Any]:
    """$L(x)=1+L_1x+L_2x^2$ gyökei: $L_2x^2+L_1x+1=0$; másodfokú megoldóképlet GF(9)-ben.

    A visszaadott ``kind`` értékek: ``quadratic``, ``linear`` (L_2=0), ``no_sqrt``, ``no_degree``.
    """
    l1i, l2i = int(l1) % 9, int(l2) % 9
    a, b, c = F(l2i), F(l1i), F(1)
    base: dict[str, Any] = {"l1": l1i, "l2": l2i, "a": int(a), "b": int(b), "c": int(c)}
    if a == 0:
        if b == 0:
            return {**base, "kind": "no_degree"}
        z = -c / b
        zi = int(z)
        Xi = int(F(1) / z) if z != 0 else None
        return {**base, "kind": "linear", "z": zi, "X": Xi}
    two = F(2)
    two_a = two * a
    if two_a == 0:
        return {**base, "kind": "no_degree"}
    # (2a)^2 = (2·2)·a² a testben; char 3 esetén 2·2 = 1 ≠ a „4” decimális címke GF(9)-ben.
    four_eff = two * two
    D = b * b - four_eff * a * c
    s = gf9_sqrt_element(D)
    Di = int(D)
    if s is None:
        return {**base, "kind": "no_sqrt", "D": Di}
    si = int(s)
    numer_p = int(-b + s)
    numer_m = int(-b - s)
    two_ai = int(two_a)
    z1 = int((-b + s) / two_a)
    z2 = int((-b - s) / two_a)
    fz1, fz2 = F(z1), F(z2)
    X1 = int(F(1) / fz1) if fz1 != 0 else None
    X2 = int(F(1) / fz2) if fz2 != 0 else None
    return {
        **base,
        "kind": "quadratic",
        "D": Di,
        "sqrt_D": si,
        "neg_b_plus_sqrt": numer_p,
        "neg_b_minus_sqrt": numer_m,
        "two_a": two_ai,
        "z1": z1,
        "z2": z2,
        "X1": X1,
        "X2": X2,
    }


def find_locator_ix_pairs(locator_asc: list[int]) -> list[tuple[int, int, int]]:
    coeffs_f = [F(int(c) % 9) for c in locator_asc]
    lp = galois.Poly(coeffs_f, field=F, order="asc")
    alpha = F.primitive_element
    triples: list[tuple[int, int, int]] = []
    for val in range(9):
        zf = F(val)
        if lp(zf) != F(0) or val == 0:
            continue
        xi = int(F(1) / zf)
        d_found: int | None = None
        for d in range(8):
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
    a_mat = F([[X[0], X[1]], [X[0] ** 2, X[1] ** 2]])
    b_vec = F([S[0], S[1]])
    y1, y2 = _gf_solve_2x2(a_mat, b_vec)
    return [y1, y2]


def _locator_roots_positions_and_X(locator_asc: list[galois.FieldArray]) -> tuple[list[int], list[galois.FieldArray]] | None:
    lp = galois.Poly(locator_asc, field=F, order="asc")
    alpha = F.primitive_element
    pairs: list[tuple[int, galois.FieldArray]] = []
    for x in F.elements:
        if lp(F(x)) != 0:
            continue
        r = F(x)
        if r == 0:
            return None
        xv = F(1) / r
        d_found: int | None = None
        for d in range(8):
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


def decode_rs84(
    r_ints: list[int], *, G_mat: galois.FieldArray, H_T_mat: galois.FieldArray
) -> tuple[list[int], list[int]] | None:
    """Dekódolás legfeljebb 2 szimbólumhibára: teljes keresés a **G** sorok által feszített kódtérben (**c·Hᵀ=0**)."""
    r_row = F(r_ints[:N]).reshape(1, -1)
    if np.all(np.asarray(r_row @ H_T_mat) == 0):
        return ([int(x) for x in r_row.flatten()], [0] * N)
    for j in range(N):
        for em in range(1, 9):
            e_row = F.Zeros((1, N))
            e_row[0, j] = F(em)
            c_row = r_row - e_row
            if np.all(np.asarray(c_row @ H_T_mat) == 0):
                return ([int(x) for x in c_row.flatten()], [int(x) for x in e_row.flatten()])
    for j1 in range(N):
        for j2 in range(j1 + 1, N):
            for e1 in range(1, 9):
                for e2 in range(1, 9):
                    e_row = F.Zeros((1, N))
                    e_row[0, j1] = F(e1)
                    e_row[0, j2] = F(e2)
                    c_row = r_row - e_row
                    if np.all(np.asarray(c_row @ H_T_mat) == 0):
                        return ([int(x) for x in c_row.flatten()], [int(x) for x in e_row.flatten()])
    return None


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
