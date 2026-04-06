from __future__ import annotations

import html

import numpy as np
import streamlit as st
import galois

import rs74_core as rc

_C_SUBSCRIPT = ('₀', '₁', '₂', '₃', '₄', '₅', '₆')


def syndrome_sum_markdown(si_sum: int, *, mask: int, alpha_map: dict[int, str]) -> str:
    v = int(si_sum) & mask
    al = alpha_map[v]
    if al == str(v):
        return f'**{si_sum}**'
    return f'**{si_sum}** — **{al}**'


def format_gf_matrix(mat: galois.FieldArray) -> str:
    rows = []
    for i in range(mat.shape[0]):
        rows.append(' & '.join((str(int(mat[i, j])) for j in range(mat.shape[1]))))
    return ' \\\\ '.join(rows)


def markdown_line_cj_from_m_dot_g(m_vals: list[int], G: galois.FieldArray, j: int) -> str:
    k = int(G.shape[0])
    terms_mul: list[str] = []
    terms_val: list[str] = []
    acc = rc.GF(0)
    for i in range(k):
        mi = int(m_vals[i]) & 7
        gij = int(G[i, j]) & 7
        prod = int(rc.GF(mi) * rc.GF(gij))
        acc = acc + rc.GF(prod)
        terms_mul.append(f'{mi}·{gij}')
        terms_val.append(str(prod))
    total = int(acc) & 7
    sub = _C_SUBSCRIPT[j]
    alpha = rc.INT_TO_ALPHA_STR[total]
    return f'**c{sub}** = ' + '+'.join(terms_mul) + ' = ' + '+'.join(terms_val) + f' = **{total}** ({alpha})'


def syndrome_ci_derivation_lines(c_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int) -> tuple[list[str], int]:
    acc = rc.GF(0)
    lines: list[str] = []
    c_flat = rc.GF(c_vec).reshape(1, -1)
    for j in range(rc.N):
        cj = int(c_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = rc.GF(cj) * rc.GF(hij)
        acc = acc + term
        lines.append(f'$c_{{{j}}}\\cdot H_{{{row_i},{j}}} = {cj}\\cdot {hij} = {int(term)}$')
    return (lines, int(acc))


def syndrome_si_derivation_lines(r_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int) -> tuple[list[str], int]:
    acc = rc.GF(0)
    lines: list[str] = []
    r_flat = rc.GF(r_vec).reshape(1, -1)
    for j in range(rc.N):
        rj = int(r_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = rc.GF(rj) * rc.GF(hij)
        acc = acc + term
        lines.append(f'$r_{{{j}}}\\cdot H_{{{row_i},{j}}} = {rj}\\cdot {hij} = {int(term)}$')
    return (lines, int(acc))


def encoding_cj_column_derivation_lines(m_row: galois.FieldArray, G_mat: galois.FieldArray, col_j: int) -> tuple[list[str], int]:
    m_flat = rc.GF(m_row).reshape(1, -1)
    acc = rc.GF(0)
    lines: list[str] = []
    for i in range(rc.K):
        mi = int(m_flat[0, i])
        gij = int(G_mat[i, col_j])
        term = rc.GF(mi) * rc.GF(gij)
        acc = acc + term
        lines.append(f'$m_{{{i}}}\\cdot G_{{{i},{col_j}}} = {mi}\\cdot {gij} = {int(term)}$')
    return (lines, int(acc))


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
        ai = int(rc.GF(si) / rc.GF(hij))
        lines.append(f'$s_{i}/H_{{{i},{col_j}}} = {si}/{hij} = {ai}$')
        quotients.append(ai)
    return (lines, quotients)


def _gf8_int_alpha_label(v: int) -> str:
    v = int(v) & 7
    return f'{v} ({rc.INT_TO_ALPHA_POWER_STR[v]})'


def gf8_arithmetic_tables_html() -> str:
    fs = 'font-size:0.875rem;'
    pad = 'padding:4px 6px;'
    td = f'style="border:1px solid rgba(128,128,128,0.45);{pad}{fs}text-align:center;vertical-align:middle;word-wrap:break-word;"'
    th = f'style="border:1px solid rgba(128,128,128,0.45);{pad}{fs}text-align:center;vertical-align:middle;font-weight:600;word-wrap:break-word;"'
    tbl = f'style="width:100%;table-layout:fixed;border-collapse:collapse;margin:0;{fs}line-height:1.4;"'

    def build_table(*, multiply: bool) -> str:
        head = f"<tr><th {th}>{html.escape('a \\ b')}</th>" + ''.join((f'<th {th}>{html.escape(_gf8_int_alpha_label(j))}</th>' for j in range(8))) + '</tr>'
        body_rows: list[str] = []
        for i in range(8):
            cells = [f'<td {td}>{html.escape(_gf8_int_alpha_label(i))}</td>']
            for j in range(8):
                if multiply:
                    v = int(rc.GF(i) * rc.GF(j))
                else:
                    v = int(rc.GF(i) + rc.GF(j))
                cells.append(f'<td {td}>{html.escape(_gf8_int_alpha_label(v))}</td>')
            body_rows.append('<tr>' + ''.join(cells) + '</tr>')
        return f"<table {tbl}><thead>{head}</thead><tbody>{''.join(body_rows)}</tbody></table>"
    mul_t = build_table(multiply=True)
    add_t = build_table(multiply=False)
    title = f'{fs}font-weight:600;margin-bottom:0.35rem;text-align:center;'
    return f'<div style="width:100%;box-sizing:border-box;display:grid;grid-template-columns:repeat(auto-fit, minmax(min(100%, 18rem), 1fr));gap:clamp(0.5rem,1.5vw,1.25rem);align-items:start;"><div style="min-width:0;width:100%;"><div style="{title}">Összeadás a+b</div>{add_t}</div><div style="min-width:0;width:100%;"><div style="{title}">Szorzás a·b</div>{mul_t}</div></div>'


def streamlit_cols_m_g(left: int = 1, right: int = 5) -> tuple:
    for g in (None, 'small'):
        try:
            return st.columns([left, right], gap=g)
        except TypeError:
            continue
    return st.columns([left, right])


def render_syndrome_r_dot_Ht_expander(r_vec: galois.FieldArray, H_mat: galois.FieldArray) -> None:
    nk = int(H_mat.shape[0])
    with st.expander('Szindróma: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$ — minden **$s_i$** ( $\\mathbf{r}$ és **$H$** soronkénti szorzata )', expanded=False):
        st.markdown('Lépésről lépésre (aktuális **r** és **H** sorai — **s₀**, **s₁**, **s₂**):')
        syn_step_cols = st.columns(nk)
        for i in range(nk):
            with syn_step_cols[i]:
                st.markdown(f'**s_{i}** — **H** **{i}.** sora és **r**:')
                si_lines, si_sum = syndrome_si_derivation_lines(r_vec, H_mat, i)
                for ln in si_lines:
                    st.markdown(ln)
                st.success(f'**Összeg (s_{{{i}}})** (GF(8), **⊕**): {syndrome_sum_markdown(si_sum, mask=7, alpha_map=rc.INT_TO_ALPHA_STR)}')


def render_encoding_m_dot_g_expander(m_row: galois.FieldArray, G_mat: galois.FieldArray, *, parity_right: bool) -> None:
    with st.expander('Előállítás: $\\mathbf{c} = \\mathbf{m} \\cdot G$ — az első kimeneti elem (**$c_0$**): $\\mathbf{m}$ és **$G$** első oszlopa', expanded=False):
        st.markdown('**c** = **m·G** (1×4)·(4×7) = (1×7). Minden **cⱼ** = Σᵢ **mᵢ**·**Gᵢ,ⱼ** (GF(8)): a **j**-edik kimenetet **m** skaláris szorzata adja **G** **j**-edik **oszlopával** (a sorokon végig **i** = 0…3).')
        st.warning('**Nem** a **G** első **sorának** elemeit kell „összeadni” (pl. **G₀,₀ + G₀,₁ + …** — ez **nem** **c₀**). Az első kimenethez **G** **első oszlopját** használjuk: **c₀** = **m₀·G₀,₀** + **m₁·G₁,₀** + **m₂·G₂,₀** + **m₃·G₃,₀**.')
        if parity_right:
            st.caption('**Paritás jobbra:** **G = [I₄ | P]** — az első oszlop gyakran **(1,0,0,0)ᵀ** kezdetű; **c₀** így **m₀**-t adja közvetlenül, ha **G₀,₀ = 1**.')
        else:
            st.caption('**Paritás balra:** **G** első oszlopa a paritásblokk részeit is tartalmazza; **c₀** továbbra a négy **mᵢ·Gᵢ,₀** összege.')
        c0_lines, c0_sum = encoding_cj_column_derivation_lines(m_row, G_mat, 0)
        st.markdown('**Lépésről lépésre** (aktuális **m** és **G** első **oszlopa** → **c₀** = Σᵢ **mᵢ·Gᵢ,₀**):')
        st.caption('A négy szorzat **összege** **GF(8)**-ban: **⊕** (bitenkénti XOR), **nem** a decimális számok egész összege.')
        for ln in c0_lines:
            st.markdown(ln)
        st.success(
            f'**Összeg (c₀)** (GF(8), **⊕**): {syndrome_sum_markdown(c0_sum, mask=7, alpha_map=rc.INT_TO_ALPHA_STR)} '
            f'(ez egyezik **c** első elemével a fenti **m·G** szorzatnál).'
        )


def render_s0_expander(c_vec: galois.FieldArray, H_mat: galois.FieldArray) -> None:
    nk = int(H_mat.shape[0])
    with st.expander('Első szindróma-komponens: **c·Hᵀ = 0** — **s₀**, **s₁**, **s₂** (aktuális **c** és **H** sorai)', expanded=False):
        st.markdown('Lépésről lépésre (aktuális **c** és **H** sorai — **s₀**, **s₁**, **s₂**); kód szónál mindhárom összeg **0** (GF(8), **⊕**):')
        s0_cols = st.columns(nk)
        for i in range(nk):
            with s0_cols[i]:
                st.markdown(f'**s_{i}** — **H** **{i}.** sora és **c**:')
                ci_lines, si_sum = syndrome_ci_derivation_lines(c_vec, H_mat, i)
                for ln in ci_lines:
                    st.markdown(ln)
                st.success(f'**Összeg (s_{{{i}}})** (GF(8), **⊕**): {syndrome_sum_markdown(si_sum, mask=7, alpha_map=rc.INT_TO_ALPHA_STR)}')
