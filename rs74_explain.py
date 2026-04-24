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


def _rs74_generator_g_poly() -> galois.Poly:
    F = rc.GF
    alpha = F.primitive_element
    ONE = F(1)
    Z = F(0)
    g0 = alpha**2 + ONE
    g1 = alpha
    g2 = alpha**2 + ONE
    return galois.Poly([g0, g1, g2, ONE], field=F, order='asc')


def poly_long_divide_by_g_records(*, dividend: galois.Poly, g: galois.Poly) -> tuple[list[dict[str, object]], galois.Poly]:
    """Maradékos osztás GF(2^3)[x]-ben; g monikus, deg(g)=3. Vissza: lépésrekordok és végső maradék."""
    F = dividend.field
    Z = F(0)
    zero = galois.Poly([Z], field=F)
    dg = int(g.degree)
    r = dividend
    steps: list[dict[str, object]] = []
    while r != zero and int(r.degree) >= dg:
        dr = int(r.degree)
        shift = dr - dg
        coeffs_r = list(r.coefficients(order='asc', size=dr + 1))
        cr = coeffs_r[dr]
        coeffs_g = list(g.coefficients(order='asc', size=dg + 1))
        cg = coeffs_g[dg]
        qc = cr / cg
        xs = galois.Poly([Z] * shift + [F(1)], field=F, order='asc')
        sub = qc * xs * g
        r_next = r - sub
        steps.append(
            {
                'dr': dr,
                'shift': shift,
                'qc': int(qc),
                'remainder_before': r,
                'subtrahend': sub,
                'remainder_after': r_next,
            }
        )
        r = r_next
    return (steps, r)


def _poly_display_latex(p: galois.Poly) -> str:
    return str(p)


def _leading_mono_from_poly(p: galois.Poly) -> tuple[int, int]:
    """(fok, együttható) a nemnulla polinom főtagjához."""
    d = int(p.degree)
    c = int(list(p.coefficients(order='asc', size=d + 1))[d])
    return (d, c)


def _mono_latex(c: int, d: int) -> str:
    c = int(c) & 7
    d = int(d)
    if d < 0:
        return '0'
    if d == 0:
        return str(c)
    if d == 1:
        return 'x' if c == 1 else f'{c}x'
    if c == 1:
        return f'x^{{{d}}}'
    return f'{c}x^{{{d}}}'


def _mono_quotient_latex(qc: int, shift: int) -> str:
    qc = int(qc) & 7
    shift = int(shift)
    if shift == 0:
        return str(qc)
    if qc == 1:
        return f'x^{{{shift}}}'
    return f'{qc}x^{{{shift}}}'


def _hányados_rész_mondat(*, step_i: int, n_steps: int) -> str:
    if n_steps == 1:
        return 'Ez lesz a hányadosunk első (és jelen esetben egyetlen) tagja.'
    if step_i == 0:
        return 'Ez lesz a hányadosunk első tagja.'
    if step_i == n_steps - 1:
        return 'Ez lesz a hányadosunk utolsó tagja.'
    return 'Ez lesz a hányadosunk egy újabb tagja.'


def _render_rs74_division_pedagogy_row(*, g_body_tex: str, k: int, steps: list, rem: galois.Poly, dg: int) -> None:
    """Ugyanaz a levezetés / formázás minden bázisra: t(x)=x^k osztása g(x)-szel (GF(8), char 2)."""
    st.markdown(f'Az **$t(x)=x^{{{k}}}$** polinomot elosztjuk a **$g(x)={g_body_tex}$** polinommal.')
    gs = 0
    n_steps = len(steps)
    for si, stp in enumerate(steps):
        rb = stp['remainder_before']
        sub = stp['subtrahend']
        ra = stp['remainder_after']
        qc = int(stp['qc'])
        shift = int(stp['shift'])
        d_lead, c_lead = _leading_mono_from_poly(rb)
        lead_tex = _mono_latex(c_lead, d_lead)
        quot_tex = _mono_quotient_latex(qc, shift)
        gs += 1
        st.markdown(
            f'**{gs}. lépés:** Az első tagok osztása: Vegyük az aktuális polinom legmagasabb fokú tagját (**${lead_tex}$**) '
            'és osszuk el az osztó legmagasabb fokú tagjával (**$x^3$**).'
        )
        st.latex(fr'{lead_tex} : x^3 = {quot_tex}')
        st.markdown(_hányados_rész_mondat(step_i=si, n_steps=n_steps))
        gs += 1
        st.markdown(
            f'**{gs}. lépés:** Visszaszorzás: Szorozzuk vissza az osztót (**$g(x)={g_body_tex}$**) a kapott **${quot_tex}$** taggal: '
            f'$\\displaystyle {_poly_display_latex(sub)}$'
        )
        gs += 1
        st.markdown(
            f'**{gs}. lépés:** Maradék — a GF($2^3$) polinomgyűrű **char 2** teste miatt a „kivonás” **együtthatónként megegyezik a $\\oplus$** művelettel.'
        )
        st.markdown('Az aktuális polinomhoz **hozzáadjuk** a visszaszorzás eredményét:')
        st.latex(
            rf'{_poly_display_latex(rb)} + \left({_poly_display_latex(sub)}\right) = {_poly_display_latex(ra)}'
        )
    d_rem = int(rem.degree)
    st.markdown(
        f'Mivel a kapott maradék foka (**{d_rem}**) kisebb, mint az osztó foka (**{dg}**), az osztás **befejeződött**.'
    )
    st.markdown('**Végeredmény**')
    st.markdown(
        f'A **$t(x)=x^{{{k}}}$** polinom **$g(x)$**-re vett maradéka (a szisztematikus paritás **$c_0,c_1,c_2$** az **$x^0,x^1,x^2$** szerint):'
    )
    st.latex(r'r(x)=' + _poly_display_latex(rem))


def render_g_parity_mod_g_long_division_expander(*, parity_right: bool) -> None:
    """G sorainak paritásblokkja: t(x)=x^{3+i} osztása g(x)-szel; szöveg a paritás bal/jobb nézethez igazítva."""
    F = rc.GF
    Z = F(0)
    g = _rs74_generator_g_poly()
    x = galois.Poly([Z, F(1)], field=F, order='asc')
    col_label = '4…6' if parity_right else '0…2'
    g_layout = '**[I₄ | P]** (paritás jobbra)' if parity_right else '**[P | I₄]** (paritás balra)'
    title = (
        'Polinomosztás **g(x)** szerint: **x³, x⁴, x⁵, x⁶** → a **G** sorok **(c₀,c₁,c₂)** paritása — '
        + f'aktuális elrendezés: {g_layout}, oszlopok **{col_label}**.'
    )
    with st.expander(title, expanded=False):
        _gc = [int(x) for x in g.coefficients(order='asc', size=4)]
        _c0, _c1, _c2, _ = _gc
        _g_int_line = rf'g(x)=x^3+{_c2}x^2+{_c1}x+{_c0}'
        st.latex(
            r'\begin{array}{c}'
            r'g(x)=x^3+(\alpha^2+1)x^2+\alpha x+(\alpha^2+1) \\[0.35em]'
            + _g_int_line
            + r' \\[0.4em]'
            + r'm(x)\in\{1,x,x^2,x^3\},\ \text{majd}\ t(x)=m(x)\cdot x^3'
            + r'\end{array}'
        )
        if parity_right:
            st.markdown(
                'A **G** **i.** sora annak a kódszónak az együtthatói, amikor az üzenet az **i.** bázisvektor '
                '(négy szimbólumból csak az **i.** helyen **1**, máshol **0**). Üzenetpolinom **$m(x)\\in\\{1,x,x^2,x^3\\}$**; '
                'szisztematikus lépés: **$t(x)=m(x)\\cdot x^{n-k}$**, itt **$n-k=3$**, tehát **$t(x)=m(x)\\,x^3$** — az osztandók sorban **$x^3,x^4,x^5,x^6$**. '
                'A **$r(x)=t(x)\\bmod g(x)$** maradék foka **$<3$**; együtthatói **$(c_0,c_1,c_2)$** az **$x^0,x^1,x^2$** tagoknál — ez a **három paritás szimbólum**. '
                '**$G=[\\mathbf{I}_4\\,|\\,\\mathbf{P}]$** (paritás jobbra) esetén a program ezt a **hármas** a mátrix **4., 5., 6.** oszlopába teszi (**P** jobbra); '
                'az **1…4.** oszlop egységmátrixa a **$x^3\\ldots x^6$** helyekhez kötött üzenetrész (nem a polinom legalacsonyabb négy tagja).'
            )
        else:
            st.markdown(
                'A **G** **i.** sora annak a kódszónak az együtthatói, amikor az üzenet az **i.** bázisvektor '
                '(négy szimbólumból csak az **i.** helyen **1**, máshol **0**). Üzenetpolinom **$m(x)\\in\\{1,x,x^2,x^3\\}$**; '
                '**$t(x)=m(x)\\,x^3$**; osztandók **$x^3,x^4,x^5,x^6$**; a **$t(x)\\bmod g(x)$** maradék együtthatói **$(c_0,c_1,c_2)$** adják a paritást. '
                'A tárolt **$[\\mathbf{I}_4\\,|\\,\\mathbf{P}]$** alakban ez a hármas a **4…6.** oszlop; **$[\\mathbf{P}\\,|\\,\\mathbf{I}_4]$** nézetben oszlop-permutáció miatt ugyanezek az értékek a fenti **G** **0., 1., 2.** oszlopa.'
            )

        _g_body = f'x^3+{_c2}x^2+{_c1}x+{_c0}'
        for i in range(4):
            k = 3 + i
            t = x**k
            steps, rem = poly_long_divide_by_g_records(dividend=t, g=g)
            m_latex = '1' if i == 0 else ('x' if i == 1 else f'x^{i}')
            st.markdown(
                f'#### **{i}.** bázis (üzenetpolinom **$m(x)={m_latex}$**) → eltolás: **$t(x)=m(x)\\,x^3=x^{{{k}}}$**'
            )
            _render_rs74_division_pedagogy_row(
                g_body_tex=_g_body, k=k, steps=steps, rem=rem, dg=int(g.degree)
            )
