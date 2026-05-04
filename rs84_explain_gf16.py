from __future__ import annotations

import html
from typing import Any

import galois
import numpy as np

import rs84_core_gf16 as rc

_C_SUB = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇')


def format_gf_matrix(mat: galois.FieldArray) -> str:
    rows = []
    for i in range(mat.shape[0]):
        rows.append(' & '.join((str(int(mat[i, j])) for j in range(mat.shape[1]))))
    return ' \\\\ '.join(rows)


def encoding_cj_column_derivation_lines(m_vals: list[int], G: galois.FieldArray, col_j: int) -> tuple[list[str], int]:
    k_ = int(G.shape[0])
    acc = rc.F(0)
    lines: list[str] = []
    for i in range(k_):
        mi = int(m_vals[i]) & 15
        gij = int(G[i, col_j]) & 15
        term = rc.F(mi) * rc.F(gij)
        acc = acc + term
        lines.append(f'$m_{{{i}}}\\cdot G_{{{i},{col_j}}} = {mi}\\cdot {gij} = {int(term)}$')
    return (lines, int(acc))


def markdown_line_cj_from_m_dot_g(m_vals: list[int], G: galois.FieldArray, j: int) -> str:
    k_ = int(G.shape[0])
    terms_mul: list[str] = []
    terms_val: list[str] = []
    acc = rc.F(0)
    for i in range(k_):
        mi = int(m_vals[i]) & 15
        gij = int(G[i, j]) & 15
        prod = int(rc.F(mi) * rc.F(gij))
        acc = acc + rc.F(prod)
        terms_mul.append(f'{mi}·{gij}')
        terms_val.append(str(prod))
    total = int(acc) & 15
    sub = _C_SUB[j]
    alpha = rc.INT_TO_ALPHA_STR[total]
    return f'**c{sub}** = ' + '+'.join(terms_mul) + ' = ' + '+'.join(terms_val) + f' = **{total}** ({alpha})'


def syndrome_ci_derivation_lines(c_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int) -> tuple[list[str], int]:
    acc = rc.F(0)
    lines: list[str] = []
    c_flat = rc.F(c_vec).reshape(1, -1)
    for j in range(rc.N):
        cj = int(c_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = rc.F(cj) * rc.F(hij)
        acc = acc + term
        lines.append(f'$c_{{{j}}}\\cdot H_{{{row_i},{j}}} = {cj}\\cdot {hij} = {int(term)}$')
    return (lines, int(acc))


def syndrome_si_derivation_lines(r_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int) -> tuple[list[str], int]:
    acc = rc.F(0)
    lines: list[str] = []
    r_flat = rc.F(r_vec).reshape(1, -1)
    for j in range(rc.N):
        rj = int(r_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = rc.F(rj) * rc.F(hij)
        acc = acc + term
        lines.append(f'$r_{{{j}}}\\cdot H_{{{row_i},{j}}} = {rj}\\cdot {hij} = {int(term)}$')
    return (lines, int(acc))


def evaluation_syndrome_sj_derivation_lines(r_ints: list[int], syndrome_j: int) -> tuple[list[str], list[str], int]:
    if syndrome_j < 1 or syndrome_j > rc.N - rc.K:
        raise ValueError('syndrome_j')
    r_clean = [int(x) & 15 for x in r_ints[: rc.N]]
    alpha = rc.F.primitive_element
    aj = alpha**syndrome_j
    term_lines: list[str] = []
    term_ints: list[int] = []
    acc = rc.F(0)
    for v in range(rc.N):
        d = rc.vector_index_to_poly_degree(v)
        rv = rc.F(r_clean[v])
        pw = aj**d
        term = rv * pw
        acc = acc + term
        ti = int(term)
        term_ints.append(ti)
        term_lines.append(
            f'$r_{{{v}}}\\,(\\alpha^{{{syndrome_j}}})^{{{7-v}}} = {r_clean[v]} \\cdot {int(pw)} = {ti}$'
        )
    sj_sum = int(acc)
    xor_lines: list[str] = []
    xor_lines.append(
        f'$S_{{{syndrome_j}}} = {term_ints[0]}'
        + ''.join(f' \\oplus {t}' for t in term_ints[1:])
        + f' = {sj_sum}$'
    )
    xor_lines.append('*Lépésenkénti részösszegek:*')
    cur = term_ints[0]
    xor_lines.append(f'$S^{{(0)}} = {cur}$')
    for i in range(1, rc.N):
        nxt = int(rc.F(cur) + rc.F(term_ints[i]))
        xor_lines.append(f'$S^{{({i})}} = S^{{({i-1})}} \\oplus {term_ints[i]} = {cur} \\oplus {term_ints[i]} = {nxt}$')
        cur = nxt
    xor_lines.append(f'$\\Rightarrow S_{{{syndrome_j}}} = {sj_sum}$')
    return (term_lines, xor_lines, sj_sum)


def explain_locator_scan_and_roots(locator_asc: list[int]) -> tuple[list[str], list[str]]:
    coeffs_f = [rc.F(int(c) & 15) for c in locator_asc]
    mdeg = len(coeffs_f) - 1
    Lp = galois.Poly(coeffs_f, field=rc.F, order='asc')
    alpha = rc.F.primitive_element
    scan_lines: list[str] = []
    for val in range(16):
        zf = rc.F(val)
        ei = int(Lp(zf))
        if mdeg == 1:
            c1 = int(coeffs_f[1])
            p1 = int(coeffs_f[1] * zf)
            line = f'$L({val}) = 1 + {c1} \\cdot {val} = 1 + {p1} = {ei}$'
        else:
            c1, c2 = int(coeffs_f[1]), int(coeffs_f[2])
            zz = int(zf**2)
            p1 = int(coeffs_f[1] * zf)
            p2 = int(coeffs_f[2] * (zf**2))
            line = (
                f'$L({val}) = 1 + {c1} \\cdot {val} + {c2} \\cdot {zz} = '
                f'1 + {p1} + {p2} = {ei}$'
            )
        if ei == 0:
            line = f'**{line}**'
        scan_lines.append(line)
    root_rows: list[tuple[int, int, int, int | None, str]] = []
    for val in range(16):
        zf = rc.F(val)
        if Lp(zf) != rc.F(0) or val == 0:
            continue
        Xi = int(rc.F(1) / zf)
        d_found: int | None = None
        for d in range(15):
            if alpha**d == rc.F(Xi):
                d_found = d
                break
        v = rc.poly_degree_to_vector_index(d_found) if d_found is not None else -1
        alab = rc.INT_TO_ALPHA_POWER_STR.get(Xi, str(Xi))
        root_rows.append((v, val, Xi, d_found, alab))
    root_rows.sort(key=lambda row: row[0])
    root_lines: list[str] = []
    for _, val, Xi, d_found, alab in root_rows:
        root_lines.append(
            f'$L({val}) = 0$: gyök **$r = {val}$** $\\Rightarrow$ **$X = r^{{-1}} = {Xi}$** ({alab}), '
            f'$X = \\alpha^{{{d_found}}}$.'
        )
    if Lp(rc.F(0)) == rc.F(0):
        root_lines.insert(0, '$L(0) = 0$ — a szokásos Chien / teljes GF(16) keresés a **$r \\neq 0$** gyököket használja.')
    return (scan_lines, root_lines)


def explain_forney_B_system_lines(m: int, X_ints: list[int], S_ints: list[int], y_ints: list[int]) -> list[str]:
    lines: list[str] = []
    if m == 1:
        x1, s1, y1 = X_ints[0], S_ints[0], y_ints[0]
        lines.append(
            f'$Y_1 = S_1 / X_1 = {s1} / {x1} = {y1}$ (osztás **GF(16)**-ban).'
        )
        return lines
    x1, x2 = X_ints[0], X_ints[1]
    y1, y2 = y_ints[0], y_ints[1]
    s1, s2 = S_ints[0], S_ints[1]
    x1f, x2f = rc.F(x1), rc.F(x2)
    t11 = int(x1f * rc.F(y1))
    t12 = int(x2f * rc.F(y2))
    ch1 = int(rc.F(t11) + rc.F(t12))
    t21 = int((x1f**2) * rc.F(y1))
    t22 = int((x2f**2) * rc.F(y2))
    ch2 = int(rc.F(t21) + rc.F(t22))
    lines.append(
        f'$X_1 Y_1 + X_2 Y_2 = {x1}\\cdot{y1} + {x2}\\cdot{y2} = {t11} + {t12} = {ch1}$ '
        f'(=$S_1 = {s1}$).'
    )
    lines.append(
        f'$X_1^2 Y_1 + X_2^2 Y_2$: ${x1}^2\\cdot{y1} = {t21}$, ${x2}^2\\cdot{y2} = {t22}$; '
        f'$\\oplus$-összeg: $= {ch2}$ (=$S_2 = {s2}$).'
    )
    return lines


def forney_ui_bundle(r_ints: list[int], locator_asc: list[int]) -> dict[str, Any] | None:
    ix = rc.find_locator_ix_pairs(locator_asc)
    if not ix:
        return None
    r_clean = [int(x) & 15 for x in r_ints[: rc.N]]
    S = rc.compute_evaluation_syndromes_fields(r_clean)
    Xg = [rc.F(xv) for xv, _, _ in ix]
    m = len(Xg)
    try:
        Yg = rc.compute_error_values_vandermonde(S, Xg)
    except (ValueError, ZeroDivisionError):
        return {'error': True, 'm': m, 'ix_pairs': ix}
    B_int = [[int(Xg[j] ** (i + 1)) for j in range(m)] for i in range(m)]
    s_int = [int(S[i]) for i in range(m)]
    y_int = [int(Yg[i]) & 15 for i in range(m)]
    detail_lines = explain_forney_B_system_lines(m, [int(x) for x in Xg], s_int, y_int)
    return {
        'm': m,
        'ix_pairs': ix,
        'B': B_int,
        's': s_int,
        'y': y_int,
        'detail_lines': detail_lines,
    }


def explain_alpha_scan_for_locator_X(target_X: int, j_idx: int) -> list[str]:
    alpha = rc.F.primitive_element
    tx = int(target_X) & 15
    out: list[str] = []
    for k in range(15):
        ak = int(alpha**k)
        if ak == tx:
            out.append(
                f'$\\alpha^{{{k}}} = {ak}$ — **egyezik** $X_{{{j_idx}}} = {tx}$ → **$i_{{{j_idx}}} = {k}$**.'
            )
        else:
            out.append(f'$\\alpha^{{{k}}} = {ak}$')
    return out


def syndrome_column_quotient_markdown_lines(s_f: np.ndarray, H_mat: galois.FieldArray, col_j: int) -> tuple[list[str], list[int]]:
    lines: list[str] = []
    quotients: list[int] = []
    nk = int(H_mat.shape[0])
    for i in range(nk):
        hij = int(H_mat[i, col_j])
        si = int(s_f[i])
        if hij == 0:
            lines.append(
                f'$H_{{{i},{col_j}}}=0$ — $s_{i}/H_{{{i},{col_j}}}$ **nem** értelmezhető (**nevező** $0$).'
            )
            continue
        ai = int(rc.F(si) / rc.F(hij))
        lines.append(f'$s_{i}/H_{{{i},{col_j}}} = {si}/{hij} = {ai}$')
        quotients.append(ai)
    return (lines, quotients)


def _gf16_int_alpha_label(v: int) -> str:
    v = int(v) & 15
    return f'{v} ({rc.INT_TO_ALPHA_POWER_STR[v]})'


def gf16_arithmetic_tables_html() -> str:
    fs = 'font-size:0.875rem;'
    pad = 'padding:4px 6px;'
    td = f'style="border:1px solid rgba(128,128,128,0.45);{pad}{fs}text-align:center;vertical-align:middle;word-wrap:break-word;"'
    th = f'style="border:1px solid rgba(128,128,128,0.45);{pad}{fs}text-align:center;vertical-align:middle;font-weight:600;word-wrap:break-word;"'
    tbl = f'style="width:100%;table-layout:fixed;border-collapse:collapse;margin:0;{fs}line-height:1.4;"'

    def build_table(*, multiply: bool) -> str:
        head = f"<tr><th {th}>{html.escape('a \\ b')}</th>" + ''.join((f'<th {th}>{html.escape(_gf16_int_alpha_label(j))}</th>' for j in range(16))) + '</tr>'
        body_rows: list[str] = []
        for i in range(16):
            cells = [f'<td {td}>{html.escape(_gf16_int_alpha_label(i))}</td>']
            for j in range(16):
                if multiply:
                    v = int(rc.F(i) * rc.F(j))
                else:
                    v = int(rc.F(i) + rc.F(j))
                cells.append(f'<td {td}>{html.escape(_gf16_int_alpha_label(v))}</td>')
            body_rows.append('<tr>' + ''.join(cells) + '</tr>')
        return f"<table {tbl}><thead>{head}</thead><tbody>{''.join(body_rows)}</tbody></table>"

    mul_t = build_table(multiply=True)
    add_t = build_table(multiply=False)
    title = f'{fs}font-weight:600;margin-bottom:0.35rem;text-align:center;'
    return f'<div style="width:100%;box-sizing:border-box;display:grid;grid-template-columns:repeat(auto-fit, minmax(min(100%, 18rem), 1fr));gap:clamp(0.5rem,1.5vw,1.25rem);align-items:start;"><div style="min-width:0;width:100%;"><div style="{title}">Összeadás a+b</div>{add_t}</div><div style="min-width:0;width:100%;"><div style="{title}">Szorzás a·b</div>{mul_t}</div></div>'
