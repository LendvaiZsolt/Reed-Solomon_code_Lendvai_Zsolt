from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nav_visibility
import rs84_core_gf16 as core
import rs84_explain_gf16 as ex
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
F = core.F
N = core.N
K = core.K
LETTER_ORDER = core.LETTER_ORDER
G_BASE = core.G_BASE
H_BASE = core.H_BASE
INT_TO_ALPHA_STR = core.INT_TO_ALPHA_STR
INT_TO_ALPHA_POWER_STR = core.INT_TO_ALPHA_POWER_STR


def _syndrome_sum_markdown(si_sum: int, *, mask: int, alpha_map: dict[int, str]) -> str:
    v = int(si_sum) & mask
    al = alpha_map[v]
    if al == str(v):
        return f'**{v}**'
    return f'**{v}**; ({al})'


_S_SUB_4 = ('₀', '₁', '₂', '₃')


def _rs84_received_row_markdown(r_ints: list[int], highlight_v: int) -> str:
    parts: list[str] = []
    for idx in range(N):
        val = int(r_ints[idx]) & 15
        if idx == int(highlight_v):
            parts.append(f'**{val}**')
        else:
            parts.append(str(val))
    return '[' + ', '.join(parts) + ']'


def _rs84_int_tuple_markdown(vals: list[int], bold_indices: set[int]) -> str:
    parts: list[str] = []
    for idx, val in enumerate(vals):
        v = int(val) & 15
        if idx in bold_indices:
            parts.append(f'**{v}**')
        else:
            parts.append(str(v))
    return '[' + ', '.join(parts) + ']'


def _rs84_v_from_i_arith_latex(ij: int, vv: int) -> str:
    rd = 7 - int(ij)
    vv = int(vv) & 15
    if 0 <= rd <= 7:
        return f'$7-{ij} = {vv}$'
    return f'$7-{ij} = {rd} \\equiv {vv} \\pmod{{8}}$'


def _render_pgz_rs84_algorithm_steps(r_ints: list[int]) -> None:
    pgz = core.compute_pgz_hankel_state(r_ints)
    st.markdown('### Algoritmus ciklikus Reed–Solomon kódok hibajavítására')
    Si = pgz['S_ints']
    st.markdown(
        '**1)** A csatornán keresztül érkezett $\\overline{v}$ vett jel felhasználásával kiszámítjuk az '
        '$S_i = v(\\alpha^i)$, $i = 1, 2, \\ldots, n-k$ szindrómákat.'
    )
    st.caption('$v \\leftrightarrow R$, $R(x)=\\sum_{v=0}^{7} r_v x^{7-v}$, $S_i=R(\\alpha^i)$.')
    st.latex(
        'S_i = v(\\alpha^i):\\quad '
        + f'S_1={Si[0]},\\; S_2={Si[1]},\\; S_3={Si[2]},\\; S_4={Si[3]}'
        + ' \\quad (n-k=4)'
    )
    with st.expander('Konkrét $S_1,\\ldots,S_4$ számolás', expanded=False):
        st.latex(r'S_j = \bigoplus_{v=0}^{7} r_v\,(\alpha^j)^{7-v}')
        st.caption(
            '**1)** Minden tagban a **$\\cdot$** **GF(16) szorzás**.'
        )
        st.caption('**2)** A **$\\oplus$** a GF(16) összeadása.')
        for jj in range(1, N - K + 1):
            st.markdown(f'**$S_{{{jj}}}$**')
            term_lines, xor_lines, _sj = ex.evaluation_syndrome_sj_derivation_lines(r_ints, jj)
            for ln in term_lines:
                st.markdown(ln)
            for ln in xor_lines:
                st.markdown(ln)
    if pgz.get('all_S_zero'):
        return
    s1, s2, s3, s4 = Si[0], Si[1], Si[2], Si[3]
    d2 = pgz['det_h2']
    st.markdown(
        '**2)** Legyen $h = t$, ahol $t$ a kód hibajavító képessége. Számítsuk ki az $A_h$ mátrix determinánsát!'
    )
    st.latex(
        r'A_h = \begin{pmatrix} '
        r'S_1 & S_2 & \ldots & S_h \\ '
        r'S_2 & S_3 & \ldots & S_{h+1} \\ '
        r'\vdots & \vdots & & \vdots \\ '
        r'S_h & S_{h+1} & \ldots & S_{2h-1} '
        r'\end{pmatrix}'
    )
    st.markdown(
        'Ha $|A_h| = 0$, akkor csökkentsük $h$ értékét eggyel. '
        'Egészen addig folytassuk ezt az eljárást, ameddig az $|A_h|$ determináns értéke már nem lesz $0$. '
        'Ez az utolsó $h$ lesz a hibák száma, az $m$ értéke.'
    )
    st.latex(f't = {pgz["t"]}')
    st.latex(f'A_2 = \\begin{{pmatrix}} {s1} & {s2} \\\\ {s2} & {s3} \\end{{pmatrix}}')
    st.latex(f'|A_2| = {d2}')
    if pgz['det_h2_zero']:
        st.markdown('Ha $|A_2| = 0$, csökkentsük $h$ értékét eggyel.')
        st.latex(f'A_1 = \\begin{{pmatrix}} {s1} \\end{{pmatrix}}')
        st.latex(f'|A_1| = {pgz["det_h1"]}')
    if not pgz['det_h2_zero']:
        st.markdown(
            f'A determináns $|A_2| = {d2}$, azaz nem nulla, ezért **$m = {pgz["m"]}$** (hibák száma).'
        )
    else:
        st.markdown(
            f'$|A_2| = 0$, ezért $h := 1$; **$|A_1| = {pgz["det_h1"]}$**. '
            f'Ez alapján **$m = {pgz["m"]}$** (hibák száma).'
        )
    st.markdown('**3)** A szindrómák és az $m$ ismeretében oldjuk meg az')
    st.latex(r'A\overline{x} = \overline{b}')
    st.markdown('lineáris egyenletrendszert, ahol')
    st.latex(
        r'A = \begin{pmatrix} '
        r'S_1 & S_2 & \cdots & S_m \\ '
        r'S_2 & S_3 & \cdots & S_{m+1} \\ '
        r'\vdots & \vdots & & \vdots \\ '
        r'S_m & S_{m+1} & \cdots & S_{2m-1} '
        r'\end{pmatrix}'
    )
    st.latex(r'\overline{b} = \begin{pmatrix} -S_{m+1} \\ -S_{m+2} \\ \vdots \\ -S_{2m} \end{pmatrix}')
    st.latex(r'\overline{x} = \begin{pmatrix} L_m \\ L_{m-1} \\ \vdots \\ L_1 \end{pmatrix}')
    st.caption('GF($2^4$); char 2: $-S_i = S_i$.')
    st.markdown('Az egyenletrendszer megoldásával ismertté válnak a hibahelypolinom együtthatói:')
    st.latex(r'L(x) = 1 + L_1 x + L_2 x^2 + \cdots + L_m x^m')
    st3 = pgz.get('step3')
    if not st3 or st3.get('error'):
        st.caption('A 3. lépés egyenletrendszere nem oldható (szinguláris).')
        return
    A = st3['A']
    b = st3['b']
    x = st3['x']
    Lasc = st3['L_asc']
    m = pgz['m']
    st.markdown('**Konkrét értékek** (aktuális **r**, **$m$**):')
    if m == 1:
        st.latex(f'A = \\begin{{pmatrix}} {A[0][0]} \\end{{pmatrix}}')
        st.latex(f'\\overline{{b}} = \\begin{{pmatrix}} -S_2 \\end{{pmatrix}} = \\begin{{pmatrix}} {b[0]} \\end{{pmatrix}}')
        st.latex(f'\\overline{{x}} = \\begin{{pmatrix}} {x[0]} \\end{{pmatrix}} = \\begin{{pmatrix}} L_1 \\end{{pmatrix}}')
        st.latex(f'L(x) = 1 + {Lasc[1]} x')
    elif m == 2:
        st.latex(
            f'A = \\begin{{pmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{pmatrix}}'
        )
        st.latex(
            f'\\overline{{b}} = \\begin{{pmatrix}} -S_3 \\\\ -S_4 \\end{{pmatrix}}'
            f' = \\begin{{pmatrix}} {b[0]} \\\\ {b[1]} \\end{{pmatrix}}'
        )
        st.latex(
            f'\\overline{{x}} = \\begin{{pmatrix}} {x[0]} \\\\ {x[1]} \\end{{pmatrix}}'
            f' = \\begin{{pmatrix}} L_2 \\\\ L_1 \\end{{pmatrix}}'
        )
        st.latex(f'L(x) = 1 + {Lasc[1]} x + {Lasc[2]} x^2')
    st.markdown('**4)** Határozzuk meg az')
    st.latex(r'L(x) = 1 + L_1 x + L_2 x^2 + \dots + L_m x^m')
    st.markdown(
        'hibahelypolinom gyökeit. A kiszámolt gyökök inverzei lesznek az '
        '$X_1, X_2, \\dots, X_m$ hibahely lokátorok.'
    )
    st.markdown(f'**Behelyettesítés** ($m = {m}$):')
    if m == 1:
        st.latex(f'L(x) = 1 + {Lasc[1]} x')
    else:
        st.latex(f'L(x) = 1 + {Lasc[1]} x + {Lasc[2]} x^2')
    scan_lines, root_lines = ex.explain_locator_scan_and_roots(Lasc)
    with st.expander('Részletszámítások ($L(z)$ minden elemre, gyökök, $X$)', expanded=False):
        st.markdown('##### $L(z)$, $z \\in \\{0,\\ldots,15\\}$ (GF(16) int)')
        for ln in scan_lines:
            st.markdown(ln)
        st.markdown('##### Gyökök ($L(r)=0$, $r \\neq 0$) és lokátorok $X = r^{-1}$')
        for ln in root_lines:
            st.markdown(ln)
        if not root_lines:
            st.caption('Nincs megjeleníthető gyök (ritka / hibás lokátor).')
    ix_pairs = core.find_locator_ix_pairs(Lasc)
    st.markdown(
        '**5)** Az $X_1, X_2, \\dots, X_m$ hibahely lokátorokból határozzuk meg az '
        '$i_1, i_2, \\dots, i_m$ hibahelyeket az'
    )
    st.latex(r'X_j = \alpha^{i_j} \quad j = 1, 2, \ldots, m')
    st.markdown('összefüggés felhasználásával.')
    st.markdown('**Behelyettesítés:**')
    if not ix_pairs:
        st.caption('Nincs kiolvasható lokátor (gyökök hiánya vagy hibás $L$).')
    else:
        st.caption(
            'A fogadott szót **$\\mathbf{r} = [r_0,\\ldots,r_7]$**; **$v \\in \\{0,\\ldots,7\\}$** '
            'annak a helynek az indexe, ahol a hiba szimbóluma áll. '
            'A Peterson / értékelő polinom **$R(x)=\\sum_{u=0}^{7} r_u\\,x^{7-u}$** — tehát **$r_v$** éppen az **$x^{7-v}$** tag együtthatója. '
            'Innen **$X=\\alpha^{i}$** esetén **$i \\equiv 7-v \\pmod{8}$** (és **$v \\equiv 7-i \\pmod{8}$**).'
        )
        for j, (xv, ij, vv) in enumerate(ix_pairs, start=1):
            xlab = INT_TO_ALPHA_POWER_STR.get(xv & 15, str(xv))
            arith = _rs84_v_from_i_arith_latex(ij, vv)
            r_row = _rs84_received_row_markdown(r_ints, vv)
            st.markdown(
                f'$X_{j} = {xv}$ ({xlab}), $\\alpha^{{{ij}}} = {xv}$, tehát **$i_{j} = {ij}$**; '
                f'**hibahely** **$v \\equiv 7-i \\pmod{{8}}$** = {arith}, **$v = {vv}$** '
                f'$\\Rightarrow$ a javítandó bejegyzés **$\\mathbf{{r}}$** = {r_row}'
            )
    with st.expander('Mellékszámítások ($\\alpha^k$ táblázat, $i_j$ megkeresése)', expanded=False):
        if not ix_pairs:
            st.caption('Nincs lokátor.')
        else:
            aux_cols = st.columns(len(ix_pairs))
            for idx, (xv, _, _) in enumerate(ix_pairs):
                j = idx + 1
                with aux_cols[idx]:
                    st.markdown(f'##### $X_{j} = {xv}$')
                    for ln in ex.explain_alpha_scan_for_locator_X(xv, j):
                        st.markdown(ln)

    st.markdown(
        '**6)** A szindrómák és az $X_1, X_2, \\ldots, X_m$ hibahely lokátorok ismeretében oldjuk meg a'
    )
    st.latex(r'B \overline{y} = \overline{s}')
    st.markdown('lineáris egyenletrendszert, ahol')
    _b6, _y6, _s6 = st.columns(3)
    with _b6:
        st.latex(
            r'B = \begin{pmatrix} '
            r'X_1 & X_2 & \cdots & X_m \\ '
            r'X_1^2 & X_2^2 & \cdots & X_m^2 \\ '
            r'\vdots & \vdots & \ddots & \vdots \\ '
            r'X_1^m & X_2^m & \cdots & X_m^m '
            r'\end{pmatrix}'
        )
    with _y6:
        st.latex(r'\overline{y} = \begin{pmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_m \end{pmatrix}')
    with _s6:
        st.latex(r'\overline{s} = \begin{pmatrix} S_1 \\ S_2 \\ \vdots \\ S_m \end{pmatrix}')
    st.caption('Minden bejegyzés és művelet **GF(16)**; $B_{i,j} = X_j^{\,i}$ ($i = 1,\\ldots,m$).')
    st6 = ex.forney_ui_bundle(r_ints, Lasc)
    st.markdown('**Konkrét értékek** (aktuális **r**, lokátorok **$X_j$**, szindrómák **$S_i$**):')
    if st6 is None:
        st.caption('Nincs lokátor — a $B\\overline{y}=\\overline{s}$ rendszer nem írható fel numerikusan.')
    elif st6.get('error'):
        st.caption('A Vandermonde-rendszer nem oldható (szinguláris / osztási hiba).')
    else:
        Bm = st6['B']
        sm = st6['s']
        ym = st6['y']
        m6 = st6['m']
        if m6 == 1:
            cB, cy, cs = st.columns(3)
            with cB:
                st.latex(f'B = \\begin{{pmatrix}} {Bm[0][0]} \\end{{pmatrix}}')
            with cy:
                st.latex(f'\\overline{{y}} = \\begin{{pmatrix}} {ym[0]} \\end{{pmatrix}}')
            with cs:
                st.latex(f'\\overline{{s}} = \\begin{{pmatrix}} {sm[0]} \\end{{pmatrix}}')
            st.markdown(
                f'Egy egyenlet: **${Bm[0][0]} \\cdot Y_1 = {sm[0]}$** → '
                f'**$Y_1 = {sm[0]} / {Bm[0][0]} = {ym[0]}$** (GF(16)).'
            )
        else:
            cB, cy, cs = st.columns(3)
            with cB:
                st.latex(
                    f'B = \\begin{{pmatrix}} {Bm[0][0]} & {Bm[0][1]} \\\\ {Bm[1][0]} & {Bm[1][1]} \\end{{pmatrix}}'
                )
            with cy:
                st.latex(
                    f'\\overline{{y}} = \\begin{{pmatrix}} {ym[0]} \\\\ {ym[1]} \\end{{pmatrix}}'
                )
            with cs:
                st.latex(
                    f'\\overline{{s}} = \\begin{{pmatrix}} {sm[0]} \\\\ {sm[1]} \\end{{pmatrix}}'
                )
        with st.expander('Mellékszámítások ($B\\overline{y}=\\overline{s}$, ellenőrzés)', expanded=False):
            for ln in st6['detail_lines']:
                st.markdown(ln)

    st.markdown(
        '**7)** A meghatározott $Y_1, Y_2, \\ldots, Y_m$ értékek és az '
        '$i_1, i_2, \\ldots, i_m$ hibahelyek ismeretében a hibavektor'
    )
    if m == 1:
        st.latex(r'\overline{e} = (0,\ldots,0, e_{i_1}=Y_1, 0,\ldots,0),')
    elif m == 2:
        st.latex(
            r'\overline{e} = (0,\ldots,0, e_{i_1}=Y_1, 0,\ldots,0, e_{i_2}=Y_2, 0,\ldots,0),'
        )
    else:
        st.latex(
            r'\overline{e} = (0,\ldots,0, e_{i_1}=Y_1, 0,\ldots,0, \ldots,'
            r'0, e_{i_m}=Y_m, 0,\ldots,0),'
        )
    st.caption(
        'Az $i_j$ itt a hiba **vektorindexe** ($0\\ldots7$), ugyanaz, mint az 5. lépés **$v_j$** értéke '
        '(nem a $\\alpha$-kitevő).'
    )
    st.markdown('az átküldött kódszó pedig')
    st.latex(r'\overline{c} = \overline{v} - \overline{e}.')
    st6_ok7 = bool(st6 and not st6.get('error') and ix_pairs)
    v_vals7: list[int] = []
    e_vals7: list[int] = []
    c_vals7: list[int] = []
    bold_idx7: set[int] = set()
    if st6_ok7:
        ym7 = st6['y']
        bold_idx7 = {int(vv) for _, _, vv in ix_pairs}
        v_vals7 = [int(r_ints[i]) & 15 for i in range(N)]
        e_vals7 = [0] * N
        for (_, _, vv), yv in zip(ix_pairs, ym7):
            e_vals7[int(vv)] = int(yv) & 15
        c_vals7 = [int(F(v_vals7[j]) + F(e_vals7[j])) & 15 for j in range(N)]
        st.markdown('**Konkrét értékek** ($\\overline{v}$ = fogadott **r**):')
        st.markdown('$\\overline{v}$ = ' + _rs84_int_tuple_markdown(v_vals7, bold_idx7))
        st.markdown('$\\overline{e}$ = ' + _rs84_int_tuple_markdown(e_vals7, bold_idx7))
        st.markdown('$\\overline{c}$ = ' + _rs84_int_tuple_markdown(c_vals7, set()))
    else:
        st.caption('A konkrét $\\overline{e}$ és $\\overline{c}$ a $Y_j$ / lokátorok hiányában nem írható fel.')
    with st.expander('Mellékszámítások ($\\overline{v}$, $\\overline{c}=\\overline{v}+\\overline{e}$)', expanded=False):
        st.caption(
            '$\\overline{v}$ a fogadott szó (**$\\mathbf{r}$**). **GF($2^4$)**, char 2: '
            '$\\overline{v}-\\overline{e} = \\overline{v}+\\overline{e}$ (komponensenként **⊕**).'
        )
        if st6_ok7:
            for j in range(N):
                if e_vals7[j] == 0:
                    continue
                st.markdown(
                    f'$c_{{{j}}} = v_{{{j}}} + e_{{{j}}} = {v_vals7[j]} + {e_vals7[j]} = {c_vals7[j]}$ '
                    f'(**GF(16)** összeadás).'
                )
            st.caption('Ahol $e_j=0$, ott $c_j=v_j$.')


def _render_peterson_evaluation_syndromes_section(pet_ints: list[int]) -> None:
    st.markdown('### Peterson-gyök (Hankel-determináns) szindrómák $S_j = R(\\alpha^j)$')
    st.caption(
        'Az **r** vektor **$r_0,\\ldots,r_7$** komponensei az értékelő polinomhoz: $R(x)=\\sum_{v=0}^{7} r_v\\,x^{7-v}$ '
        '(a **v** indexű helyen álló szimbólum együtthatója **$x^{7-v}$**-nek felel meg). '
        'Így $S_j=R(\\alpha^j)=\\sum_v r_v\\,(\\alpha^j)^{7-v}$. Hibátlan kódnál $S_1=\\cdots=S_4=0$.'
    )
    st.markdown(
        '**S₁…S₄** (int 0…15): `' + core.format_int_row(pet_ints) + '`  \n**α hatvány:** '
        + ', '.join((INT_TO_ALPHA_POWER_STR[v & 15] for v in pet_ints))
    )


def streamlit_cols_m_g(left: int = 1, right: int = 5) -> tuple:
    for g in (None, 'small'):
        try:
            return st.columns([left, right], gap=g)
        except TypeError:
            continue
    return st.columns([left, right])


def render_syndrome_r_dot_Ht_expander(r_vec, H_mat) -> None:
    nk = int(H_mat.shape[0])
    with st.expander('Szindróma: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$ — minden **$s_i$** ( $\\mathbf{r}$ és **$H$** soronkénti szorzata )', expanded=False):
        st.markdown('Lépésről lépésre (aktuális **r** és **H** sorai — **s₀**, **s₁**, **s₂**, **s₃**):')
        syn_step_cols = st.columns(nk)
        for i in range(nk):
            with syn_step_cols[i]:
                st.markdown(f'**s{_S_SUB_4[i]}** — **H** **{i}.** sora és **r**:')
                si_lines, si_sum = ex.syndrome_si_derivation_lines(r_vec, H_mat, i)
                for ln in si_lines:
                    st.markdown(ln)
                st.success(f'**Összeg (s{_S_SUB_4[i]})** (GF(16), **⊕**): {_syndrome_sum_markdown(si_sum, mask=15, alpha_map=INT_TO_ALPHA_STR)}')


def render_encoding_m_dot_g_expander(m_row, G_mat, *, m_vals: list[int]) -> None:
    with st.expander('Előállítás: $\\mathbf{c} = \\mathbf{m} \\cdot G_{RS}$ — az első kimeneti elem (**$c_0$**): $\\mathbf{m}$ és **$G_{RS}$** első oszlopa', expanded=False):
        st.markdown('**c** = **m·G_RS** (1×4)·(4×8) = (1×8). Minden **cⱼ** = Σᵢ **mᵢ**·**Gᵢ,ⱼ** (GF(16)): a **j**-edik kimenetet **m** skaláris szorzata adja **G_RS** **j**-edik **oszlopával** (a sorokon végig **i** = 0…3).')
        st.warning('**Nem** a **G_RS** első **sorának** elemeit kell „összeadni” (pl. **G₀,₀ + G₀,₁ + …** — ez **nem** **c₀**). Az első kimenethez **G_RS** **első oszlopját** használjuk: **c₀** = **m₀·G₀,₀** + **m₁·G₁,₀** + **m₂·G₂,₀** + **m₃·G₃,₀**.')
        st.caption('**G_RS = [I₄ | P]** — az első oszlop **(1,0,0,0)ᵀ**; **c₀** = **m₀**.')
        c0_lines, c0_sum = ex.encoding_cj_column_derivation_lines(m_vals, G_mat, 0)
        st.markdown('**Lépésről lépésre** (aktuális **m** és **G** első **oszlopa** → **c₀** = Σᵢ **mᵢ·Gᵢ,₀**):')
        st.caption('A négy szorzat **összege** **GF(16)**-ban: **⊕** (bitenkénti XOR a 4 bites reprezentáción), **nem** a decimális számok egész összege.')
        for ln in c0_lines:
            st.markdown(ln)
        st.success(
            f'**Összeg (c₀)** (GF(16), **⊕**): {_syndrome_sum_markdown(c0_sum, mask=15, alpha_map=INT_TO_ALPHA_STR)} '
            f'(ez egyezik **c** első elemével a fenti **m·G** szorzatnál).'
        )


def render_s0_expander(c_vec, H_mat) -> None:
    nk = int(H_mat.shape[0])
    with st.expander(
        'Küldött szó, azaz hibátlan szindróma: **c·Hᵀ = 0** — **s₀**, **s₁**, **s₂**, **s₃** (aktuális **c** és **H** sorai)',
        expanded=False,
    ):
        st.markdown('Lépésről lépésre (aktuális **c** és **H** sorai — **s₀**…**s₃**); kód szónál mind a négy összeg **0** (GF(16), **⊕**):')
        s0_cols = st.columns(nk)
        for i in range(nk):
            with s0_cols[i]:
                st.markdown(f'**s{_S_SUB_4[i]}** — **H** **{i}.** sora és **c**:')
                ci_lines, si_sum = ex.syndrome_ci_derivation_lines(c_vec, H_mat, i)
                for ln in ci_lines:
                    st.markdown(ln)
                st.success(f'**Összeg (s{_S_SUB_4[i]})** (GF(16), **⊕**): {_syndrome_sum_markdown(si_sum, mask=15, alpha_map=INT_TO_ALPHA_STR)}')


_SIDEBAR_INJ_MODE_KOZVETLEN = 'Közvetlen fogadott érték: r[j] = választott 4 bit (0…15)'
_SIDEBAR_INJ_MODE_OSSZEADAS = 'Összeadás: r[j] = c[j] + e (e ≠ 0)'

_RS84_MINTA_EGYENI = 'Egyéni beállítás'
_RS84_MINTA_1 = '1-es mintahiba'
_RS84_MINTA_2 = '2-es mintahiba'
_RS84_MINTA_RND1 = 'random 1 hiba'
_RS84_MINTA_RND2 = 'random 2 hiba'
_RS84_MINTAHIBA_PRESETS: dict[str, tuple[tuple[int, int], ...]] = {
    _RS84_MINTA_1: ((1, 12), (6, 2)),
    _RS84_MINTA_2: ((0, 15), (7, 1)),
}
_RS84_MINTAHIBA_LETTERS: dict[str, tuple[str, str, str, str]] = {
    _RS84_MINTA_1: ('P', 'A', 'N', 'C'),
    _RS84_MINTA_2: ('D', 'C', 'B', 'A'),
}


def _rs84_apply_minta_from_radio() -> None:
    sel = st.session_state.get('rs84_minta_preset', _RS84_MINTA_EGYENI)
    if sel == _RS84_MINTA_EGYENI:
        return
    if sel in (_RS84_MINTA_RND1, _RS84_MINTA_RND2):
        rng = np.random.default_rng()
        n_err = 1 if sel == _RS84_MINTA_RND1 else 2
        m_vals_loc = [core.letter_to_gf_int(str(st.session_state.get(f'rs84_letter{i}', 'A'))) for i in range(4)]
        c_loc = (F(m_vals_loc).reshape(1, K) @ G_BASE)
        c_ints = [int(x) for x in np.asarray(c_loc).flatten()]
        pos = rng.choice(list(range(N)), size=n_err, replace=False).tolist()
        pairs_list: list[tuple[int, int]] = []
        for p in pos:
            p = int(p)
            good = int(c_ints[p])
            bad = good
            while bad == good:
                bad = int(rng.integers(0, 16))
            pairs_list.append((p, bad))
        pairs = tuple(pairs_list)
        st.session_state['rs84_minta_snapshot'] = pairs
    else:
        pairs = _RS84_MINTAHIBA_PRESETS[sel]
        st.session_state['rs84_minta_snapshot'] = pairs
    st.session_state['rs84_num_symbol_errors'] = len(pairs)
    st.session_state['rs84_error_inj_mode'] = _SIDEBAR_INJ_MODE_KOZVETLEN
    for i, (p, v) in enumerate(pairs):
        st.session_state[f'rs84_err_pos_{i}'] = p
        st.session_state[f'rs84_recv_sym_{i}'] = v
    letters = _RS84_MINTAHIBA_LETTERS.get(sel)
    if letters:
        _set_letters(letters)


def _rs84_minta_clear_if_drift() -> None:
    minta = st.session_state.get('rs84_minta_preset', _RS84_MINTA_EGYENI)
    if minta == _RS84_MINTA_EGYENI:
        return
    if minta in (_RS84_MINTA_RND1, _RS84_MINTA_RND2):
        pairs = st.session_state.get('rs84_minta_snapshot')
        if pairs is None:
            st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
            return
    else:
        pairs = _RS84_MINTAHIBA_PRESETS[minta]
    if st.session_state.get('rs84_num_symbol_errors') != len(pairs):
        st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
        return
    mode = st.session_state.get('rs84_error_inj_mode', _SIDEBAR_INJ_MODE_KOZVETLEN)
    if not mode.startswith('Közvetlen'):
        st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
        return
    for i, (p_exp, v_exp) in enumerate(pairs):
        if int(st.session_state.get(f'rs84_err_pos_{i}', -999)) != p_exp:
            st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
            return
        if int(st.session_state.get(f'rs84_recv_sym_{i}', -999)) != v_exp:
            st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
            return
    letters_exp = _RS84_MINTAHIBA_LETTERS.get(minta)
    if letters_exp:
        for li in range(4):
            if st.session_state.get(f'rs84_letter{li}') != letters_exp[li]:
                st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
                return


def _set_letters(letters: tuple[str, str, str, str]) -> None:
    for i, ch in enumerate(letters):
        st.session_state[f'rs84_letter{i}'] = ch


def on_click_test1() -> None:
    _set_letters(('B', 'C', 'D', 'A'))


def on_click_test2() -> None:
    _set_letters(('A', 'A', 'A', 'A'))


def on_click_test3() -> None:
    rng = np.random.default_rng()
    for i in range(4):
        st.session_state[f'rs84_letter{i}'] = rng.choice(LETTER_ORDER)


st.set_page_config(page_title='RS(8,4) – hibajavító kódolás (GF(16))', layout='wide')
st.title('RS(8,4) – hibajavító kódolás (GF(16))')
_rs_d_min = N - K + 1
_rs_t = (_rs_d_min - 1) // 2
st.markdown(
    f'**Műveleti test:** GF(2⁴), irreducibilis polinom **{core.IRREDUCIBLE_DESC}**. '
    f'**Generátorpolinom:** $g(x)=(x+\\alpha)(x+\\alpha^2)(x+\\alpha^3)(x+\\alpha^4)$. '
    f'**$G_{{RS}}=[I_4\\mid P]$**. '
    f'Peterson / értékelési szindrómákhoz $R(x)=\\sum_{{v=0}}^{{7}} c_v\\,x^{{7-v}}$. '
    f'A kód hossza **n = {N}**, üzenet **k = {K}**, paritás **n − k = {N - K}**. '
    f'A kód hibajavító képessége **t = {_rs_t}**; '
    f'($d_{{\\min}} = n - k + 1 = {N} - {K} + 1 = {_rs_d_min}$ (MDS). '
    f'$t = \\lfloor (d_{{\\min}} - 1) / 2 \\rfloor = \\lfloor ({_rs_d_min} - 1) / 2 \\rfloor = {_rs_t}$).'
)
with st.sidebar:
    st.page_link('app.py', label='RS(7,4)')
    st.page_link('pages/2_rs84.py', label='RS(8,4) GF(9)')
    if nav_visibility.SHOW_RS84_GF16_PAGE_LINK:
        st.page_link('pages/2_rs84_gf16.py', label='RS(8,4) GF(16) - törlésre kerül')
    st.page_link('pages/3_dokumentacio.py', label='Használati útmutató')
    st.divider()
    st.header('Bemenetek')
    st.subheader('4 szimbólum (A–P)')
    st.caption('Minden betűhöz **4 bit** (0…15); mind a négy hely üzenet **m₀…m₃**.')
    chosen: list[str] = []
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            chosen.append(
                st.selectbox(
                    f'#{i + 1}',
                    LETTER_ORDER,
                    index=i + 1,
                    key=f'rs84_letter{i}',
                    on_change=_rs84_minta_clear_if_drift,
                )
            )
    m_vals = [core.letter_to_gf_int(ch) for ch in chosen]
    st.subheader('Hibák injektálása (1–2 szimbólum)')
    corrupt = st.checkbox('Hiba beszúrása', value=False, key='rs84_corrupt')
    inj_mode = _SIDEBAR_INJ_MODE_KOZVETLEN
    num_errors = 1
    err_pos_list: list[int] = [0]
    err_mag_list: list[int] = [1]
    recv_sym_list: list[int] = [0]
    if corrupt:
        if st.session_state.get('rs84_minta_preset') == '3-as mintahiba':
            st.session_state['rs84_minta_preset'] = _RS84_MINTA_EGYENI
        st.radio(
            'Mintahiba (kézi módosítás után: Egyéni beállítás)',
            (_RS84_MINTA_EGYENI, _RS84_MINTA_1, _RS84_MINTA_2, _RS84_MINTA_RND1, _RS84_MINTA_RND2),
            index=0,
            key='rs84_minta_preset',
            on_change=_rs84_apply_minta_from_radio,
        )
        inj_mode = st.radio(
            'Hiba beállítás módja (minden hibára azonos)',
            (_SIDEBAR_INJ_MODE_KOZVETLEN, _SIDEBAR_INJ_MODE_OSSZEADAS),
            index=0,
            key='rs84_error_inj_mode',
            on_change=_rs84_minta_clear_if_drift,
        )
        _ne_opts = [1, 2]
        _ne_prev = int(st.session_state.get('rs84_num_symbol_errors', 1))
        if _ne_prev not in _ne_opts:
            st.session_state['rs84_num_symbol_errors'] = min(max(_ne_prev, 1), 2)
        num_errors = int(
            st.selectbox(
                'Hibák száma',
                _ne_opts,
                index=_ne_opts.index(int(st.session_state.get('rs84_num_symbol_errors', 1))),
                key='rs84_num_symbol_errors',
                on_change=_rs84_minta_clear_if_drift,
            )
        )
        err_pos_list = []
        err_mag_list = []
        recv_sym_list = []
        for i in range(num_errors):
            st.markdown(f'**Hiba {i + 1} / {num_errors}**')
            p = st.selectbox(
                f'Pozíció j (0…7) — hiba {i + 1}',
                list(range(N)),
                index=min(i, N - 1),
                key=f'rs84_err_pos_{i}',
                on_change=_rs84_minta_clear_if_drift,
            )
            err_pos_list.append(int(p))
            if inj_mode.startswith('Összeadás'):
                em = st.selectbox(
                    f'Hiba e (nemnulla) — hiba {i + 1}',
                    list(range(1, 16)),
                    index=0,
                    format_func=core.gf_symbol_select_label,
                    key=f'rs84_err_mag_{i}',
                    on_change=_rs84_minta_clear_if_drift,
                )
                err_mag_list.append(int(em))
            else:
                rs = st.selectbox(
                    f'Fogadott r[j] — hiba {i + 1}',
                    list(range(16)),
                    index=0,
                    format_func=core.gf_symbol_select_label,
                    key=f'rs84_recv_sym_{i}',
                    on_change=_rs84_minta_clear_if_drift,
                )
                recv_sym_list.append(int(rs))
with st.expander('GF(16) elemek: karakter ↔ 4 bit ↔ polinom ↔ α hatvány', expanded=False):
    st.caption(
        'A szimbólumok és a mezőelemek megfeleltetése (primitív elem **α**, irreducibilis polinom **x⁴+x+1**). '
        '**Sorrend:** **A → P** betűrend. A **G_RS** és **H_RS** együtthatói **0…15** egészként számolódnak.'
    )
    _df_gf16 = pd.DataFrame(core.gf16_element_table_rows())
    _df_gf16['GF(16) int (0–15)'] = _df_gf16['GF(16) int (0–15)'].astype(str)
    _sty_gf16 = _df_gf16.style.set_properties(subset=['GF(16) int (0–15)'], **{'text-align': 'left'})
    st.dataframe(_sty_gf16, use_container_width=True, hide_index=True)
with st.expander('GF(2⁴) szorzás és összeadás (16×16)', expanded=False):
    st.caption(
        '**Sor / oszlopfejléc:** **a** és **b** (int **0…15**), zárójelben az **α-hatvány** alak. '
        '**Összeadás:** char 2 → **XOR** a 4 bites reprezentáción. **Szorzás:** mezőbeli **a·b**.'
    )
    st.markdown(ex.gf16_arithmetic_tables_html(), unsafe_allow_html=True)
G, H = G_BASE, H_BASE
m = F(m_vals).reshape(1, K)
c = m @ G
r = c.copy()
if corrupt:
    r = r.copy()
    for hi in range(num_errors):
        pos = err_pos_list[hi]
        if inj_mode.startswith('Összeadás'):
            r[0, pos] = r[0, pos] + F(err_mag_list[hi])
        else:
            r[0, pos] = F(recv_sym_list[hi])
e = r - c
tab_g, tab_enc, tab_err, tab_syn, tab_dec = st.tabs(['Alapadatok', 'Kódolás', 'Fogadott szó és hiba', 'Szindróma', 'Javítás / dekódolás'])
with tab_g:
    st.subheader('Generátorpolinom g(x)')
    st.latex('g(x)=(x+\\alpha)(x+\\alpha^2)(x+\\alpha^3)(x+\\alpha^4)')
    st.caption(
        'A **g(x)** foka **n−k = 4**; a szűk értelmű $[8,4]$ RS kód generátora (gyökök $\\alpha,\\ldots,\\alpha^4$). '
        'A konkrét **P** blokk a **G_RS = [I₄ | P]** mátrixban a **reedsolo** `encode` szisztematikus kimenetéből származik — ugyanahhoz a lineáris kódhoz tartozik, mint ez a **g(x)**.'
    )
    st.subheader('Generátor mátrix $G_{RS}$ (4 × 8)')
    st.caption(
        '**G_RS = [I₄ | P]** (első négy oszlop egységmátrix). **c = m · G_RS** → **[m₀,m₁,m₂,m₃ | paritás négy szimbólum]**; soronként **encode(eᵢ)** a **reedsolo**-ban.'
    )
    st.dataframe(np.array(G, dtype=int), use_container_width=True)
    st.latex('G_{RS} = \\begin{bmatrix} ' + ex.format_gf_matrix(G) + ' \\end{bmatrix}')
    st.subheader('Paritás-mátrix $H_{RS}$ (4 × 8)')
    st.caption('**Szisztematikus pár:** ha **G_RS = [I₄ | P]**, akkor **H_RS = [Pᵀ | I₄]** (4×8).')
    st.dataframe(np.array(H, dtype=int), use_container_width=True)
    st.latex('H_{RS} = \\begin{bmatrix} ' + ex.format_gf_matrix(H) + ' \\end{bmatrix}')
with tab_enc:
    st.subheader('Üzenet és kódolás')
    st.markdown('Választott szimbólumok: **' + ', '.join(chosen) + '** → **m₀…m₃** (A=0, …, P=15).')
    rows_enc = []
    for i, ch in enumerate(chosen):
        v = core.letter_to_gf_int(ch)
        row = {'Pozíció': i + 1, 'Betű': ch}
        row.update(core.gf_int_to_labels(v))
        rows_enc.append(row)
    st.dataframe(rows_enc, use_container_width=True)
    m_ints_enc = core.gf_row_to_ints(m)
    st.markdown('**m** (int, m₀…m₃): ' + core.format_gf16_int_tuple(m_ints_enc))
    st.markdown('**m** (bitsorozat): ' + core.format_gf16_bits_tuple(m_ints_enc))
    st.markdown('### Előállítás: $\\mathbf{c} = \\mathbf{m} \\cdot G_{RS}$')
    c_ints_enc = core.gf_row_to_ints(c)
    m_row_tex = ' & '.join((str(v) for v in m_ints_enc))
    g_tex_inner = ex.format_gf_matrix(G)
    c_tex_inner = ' & '.join((str(v) for v in c_ints_enc))
    row_top_l, row_top_r = streamlit_cols_m_g()
    with row_top_l:
        st.empty()
    with row_top_r:
        st.latex('\\begin{bmatrix} ' + g_tex_inner + ' \\end{bmatrix}')
    row_bot_l, row_bot_r = streamlit_cols_m_g()
    with row_bot_l:
        st.latex('\\begin{bmatrix} ' + m_row_tex + ' \\end{bmatrix}')
    with row_bot_r:
        st.latex('\\begin{bmatrix} ' + c_tex_inner + ' \\end{bmatrix}')
    render_encoding_m_dot_g_expander(m, G, m_vals=m_vals)
    st.subheader('Kód szó α alakban')
    alpha_row = [INT_TO_ALPHA_STR[v] for v in core.gf_row_to_ints(c)]
    st.write(', '.join((f'c{j}={s}' for j, s in enumerate(alpha_row))))
    ci = core.gf_row_to_ints(c)
    st.subheader('Polinom és bit-sorrend (c₀ vs c₇ elöl)' if core.SHOW_KODOLAS_C7_DESCENDING else 'Polinom és bit-sorrend (c₀…c₇)')
    st.markdown(
        '**Értékelő polinom** (Peterson / $S_j$ szindrómák): $R(x) = c_0 x^7 + c_1 x^6 + \\cdots + c_7 = \\sum_{v=0}^{7} c_v\\,x^{7-v}$.'
    )
    st.markdown(
        '**c = m·G_RS** szisztematikus: **c₀…c₃** = **m₀…m₃**, **c₄…c₇** a paritás (ugyanaz a sorrend, mint a **reedsolo** `encode` 8 bájtja). '
        'A bitek és a **32 bit** sorozat továbbra is **c₀ → c₇** index szerint halad.'
    )
    if core.SHOW_KODOLAS_CI_INT_LIST:
        st.write('**Együtthatók c₀…c₇ (integer):**', ci)
    st.write('**Blokkok [c₀,…,c₇]:**', core.bracket_groups_bits(ci, descending=False))
    if core.SHOW_KODOLAS_C7_DESCENDING:
        st.write('**Blokkok [c₇,…,c₀]:**', core.bracket_groups_bits(ci, descending=True))
    st.write('**32 bit, sorrend c₀→c₇:**', core.bits32_c0_to_c7(ci))
    if core.SHOW_KODOLAS_C7_DESCENDING:
        st.write('**32 bit, sorrend c₇→c₀:**', core.bits32_c7_to_c0(ci))
    st.markdown('#### Az együtthatók kiszámolása a GF(2⁴) táblázat alapján (aktuális üzenet) (c₀…c₇)')
    for j in range(N):
        st.markdown(ex.markdown_line_cj_from_m_dot_g(m_vals, G, j))
with tab_err:
    st.subheader('Fogadott vektor és hibavektor')
    c_ints_err = core.gf_row_to_ints(c)
    r_ints_err = core.gf_row_to_ints(r)
    e_ints_err = core.gf_row_to_ints(e)
    st.markdown('**c** (küldött) = `' + core.format_int_row(c_ints_err) + '`  \n**r** (fogadott) = `' + core.format_int_row(r_ints_err) + '`  \n**e** = **r** − **c** = `' + core.format_int_row(e_ints_err) + '`')
    st.subheader('32 bites reprezentáció (c₀…c₇)')
    st.caption('Minden szimbólum 4 bit; összesen 32 bit, sorrend c₀→c₇.')
    c32 = core.bits32_c0_to_c7(c_ints_err)
    r32 = core.bits32_c0_to_c7(r_ints_err)
    e32 = core.bits32_c0_to_c7(e_ints_err)
    c32s = core.bits32_spaced_c0_to_c7(c_ints_err)
    r32s = core.bits32_spaced_c0_to_c7(r_ints_err)
    e32s = core.bits32_spaced_c0_to_c7(e_ints_err)
    lbl = 16
    st.code(f"{'c (küldött):':<{lbl}}{c32}\n{'r (fogadott):':<{lbl}}{r32}\n{'e = r − c:':<{lbl}}{e32}\n\n{'c (küldött):':<{lbl}}{c32s}\n{'r (fogadott):':<{lbl}}{r32s}\n{'e = r − c:':<{lbl}}{e32s}", language=None)
    if corrupt:
        if num_errors >= 3:
            st.warning(
                '**3 szimbólumhiba** esetén ez a [8,4] kód **nem** javítható megbízhatóan; az **S₁…S₄** alapú dekóder legfeljebb **2** hibát kezel.'
            )
        elif num_errors == 2:
            st.info(
                '**Két hiba:** a **Szindróma** fülön az **s = r·Hᵀ** egyetlen **H**-oszlopra illesztése **nem** ad **j**-t. '
                'A **Javítás** fül a **S₁…S₄** (értékelési / Peterson-gyök, **Hankel-determináns**) szindrómák alapján viszont mindkét hibát **megtalálhatjuk és javíthatjuk**.'
            )
    else:
        st.success('Nincs szándékos hiba: r = c.')
with tab_syn:
    st.subheader('Szindróma számítás')
    if corrupt and num_errors == 2:
        st.caption(
            '**Két hiba:** a **Szindróma** fülön az **s = r·Hᵀ** egyetlen **H**-oszlopra illesztése **nem** ad **j**-t. '
            'Az **S₁…S₄** (értékelési / Peterson-gyök, **Hankel-determináns**) szekció és a **Javítás** fül e szindrómák alapján mindkét hibát **megtalálhatjuk és javíthatjuk**.'
        )
    s = core.syndrome_row(r, H)
    s_ints = core.gf_row_to_ints(s)
    r_ints_syn = core.gf_row_to_ints(r)
    c_ints_syn = core.gf_row_to_ints(c)
    r_row_tex = ' & '.join((str(v) for v in r_ints_syn))
    s_row_tex = ' & '.join((str(v) for v in s_ints))
    h_t_tex = ex.format_gf_matrix(H.T)
    s_c = core.syndrome_row(c, H)
    s_c_ints = core.gf_row_to_ints(s_c)
    c_row_tex = ' & '.join((str(v) for v in c_ints_syn))
    s_c_row_tex = ' & '.join((str(v) for v in s_c_ints))
    st.markdown('### Helyes (küldött) kód szó: $\\mathbf{0} = \\mathbf{c} \\, H^{\\mathsf{T}}$')
    _syn_lr = (5, 6)
    c_top_l, c_top_r = streamlit_cols_m_g(*_syn_lr)
    with c_top_l:
        st.empty()
    with c_top_r:
        st.latex('H^{\\mathsf{T}} = \\begin{bmatrix} ' + h_t_tex + ' \\end{bmatrix}')
    c_bot_l, c_bot_r = streamlit_cols_m_g(*_syn_lr)
    with c_bot_l:
        st.latex('\\mathbf{c} = \\begin{bmatrix} ' + c_row_tex + ' \\end{bmatrix}')
    with c_bot_r:
        st.latex('\\mathbf{c} \\, H^{\\mathsf{T}} = \\begin{bmatrix} ' + s_c_row_tex + ' \\end{bmatrix}')
    render_s0_expander(c, H)
    st.markdown('### Fogadott szó: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$')
    syn_top_l, syn_top_r = streamlit_cols_m_g(*_syn_lr)
    with syn_top_l:
        st.empty()
    with syn_top_r:
        st.latex('H^{\\mathsf{T}} = \\begin{bmatrix} ' + h_t_tex + ' \\end{bmatrix}')
    syn_bot_l, syn_bot_r = streamlit_cols_m_g(*_syn_lr)
    with syn_bot_l:
        st.latex('\\mathbf{r} = \\begin{bmatrix} ' + r_row_tex + ' \\end{bmatrix}')
    with syn_bot_r:
        st.latex('\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}} = \\begin{bmatrix} ' + s_row_tex + ' \\end{bmatrix}')
    st.markdown(
        '**α hatvány alak** (s₀…s₃): '
        + ', '.join((INT_TO_ALPHA_POWER_STR[v & 15] for v in s_ints))
        + '  \n**Polinom alak:** '
        + ', '.join((INT_TO_ALPHA_STR[v & 15] for v in s_ints))
    )
    pet_ints = core.compute_evaluation_syndromes_ints(r_ints_syn)
    j_hat, a_hat = core.single_error_from_syndrome(s.flatten(), H)
    defer_peterson_below_blue = j_hat is None and not np.all(s == 0)
    if not defer_peterson_below_blue:
        _render_peterson_evaluation_syndromes_section(pet_ints)
    render_syndrome_r_dot_Ht_expander(r, H)
    if j_hat is not None:
        st.subheader('GF(16) osztások — hibahely meghatározás, oszlop-illesztés (**egy hiba**)')
    st.markdown(
        f'### Szindróma és **H** oszlop (**csak egy hiba** esetén)\n\n**Tehát** `{core.format_int_row(s_ints)}` „**kijelöli**” azt az **H**-**oszlopot** (**j**), amelyre **létezik** olyan **ε ∈ GF(16)**, hogy '
        f'**s₀ = ε·H₀,ⱼ**, …, **s₃ = ε·H₃,ⱼ** (mind GF(16)).'
    )
    if np.all(s == 0):
        st.caption('**s = 0:** nincs oszlophoz illesztendő nemtriviális szindróma.')
    if j_hat is not None:
        s_f = s.flatten()
        h_col = H[:, j_hat]
        s0, s1 = (int(s_f[0]), int(s_f[1]))
        q_ratio_s: Optional[int] = int(F(s0) / F(s1)) if s1 != 0 else None
        ratio_lines = []
        for i in range(N - K):
            hij = int(h_col[i])
            si = int(s_f[i])
            if hij == 0:
                ratio_lines.append(
                    f'$H_{{{i},{j_hat}}}=0$ — $s_{i}/H_{{{i},{j_hat}}}$ **nem** értelmezhető (**nevező** $0$).'
                )
                continue
            ai = F(si) / F(hij)
            ratio_lines.append(f'$s_{i}/H_{{{i},{j_hat}}} = {si}/{hij} = {int(ai)}$')
        st.markdown('  \n'.join(ratio_lines))
        st.markdown(f'Minden **értelmezett** $s_i/H_{{i,j}}$ = **ε** = **{int(a_hat)}**; $j={j_hat}$.')
        with st.expander('További H-oszlopok (j ≠ ' + str(j_hat) + '): ugyanazok a sᵢ/Hᵢ,ⱼ számítások', expanded=False):
            st.markdown('##### **sᵢ/Hᵢ,ⱼ** minden sorra (j ≠ ' + str(j_hat) + ')')
            for j_alt in sorted(j for j in range(N) if j != j_hat):
                alt_lines, alt_qs = ex.syndrome_column_quotient_markdown_lines(s_f, H, j_alt)
                st.markdown(f'**H{j_alt}.** oszlop (**j = {j_alt}**):')
                st.markdown('  \n'.join(alt_lines))
                if len(alt_qs) >= 2 and len(set(alt_qs)) > 1:
                    st.caption(f'**j = {j_alt}:** a hányadosok **nem** egyeznek — **nincs** olyan **ε**, amellyel minden sor stimmelne.')
                elif len(alt_qs) >= 2 and len(set(alt_qs)) == 1:
                    st.caption(f'**j = {j_alt}:** a számolt hányados(ok) megegyeznek, de ez az oszlop **nem** illeszkedik a teljes **s** vektorra (a helyes oszlop **j = {j_hat}**).')
        with st.expander('Első két szindróma-komponens hányadosa s₀/s₁ (és ugyanígy az H mátrix j = ' + str(j_hat) + ' oszlopának első két eleme)', expanded=False):
            st.markdown('**A megfelelő érték.**')
            st.markdown(f'**H{j_hat}.** oszlop (**j = {j_hat}**):')
            if q_ratio_s is not None:
                st.markdown(f'$s_0/s_1 = {s0}/{s1} = {q_ratio_s}$')
            else:
                st.caption('$s_1 = 0$: az $s_0/s_1$ hányados nem értelmezett; használd a fenti $s_i/H_{i,j}$ sorokat.')
            h0c, h1c = (int(h_col[0]), int(h_col[1]))
            if h1c != 0:
                q_h_main = int(F(h0c) / F(h1c))
                st.markdown(f'$H_{{0,{j_hat}}}/H_{{1,{j_hat}}} = {h0c}/{h1c} = {q_h_main}$')
            else:
                st.caption(
                    f'$H_{{1,{j_hat}}}=0$ — **H₀,ⱼ/H₁,ⱼ** nem értelmezett, mert **H₁,ⱼ** = 0 ($j={j_hat}$).'
                )
            st.markdown('**A nem megfelelő értékek.**')
            for j_alt in sorted(j for j in range(N) if j != j_hat):
                st.markdown(f'**H{j_alt}.** oszlop (**j = {j_alt}**):')
                h0j = int(H[0, j_alt])
                h1j = int(H[1, j_alt])
                if h1j == 0:
                    st.caption(
                        f'$H_{{1,{j_alt}}}=0$ — **H₀,ⱼ/H₁,ⱼ** nem értelmezett, mert **H₁,ⱼ** = 0 ($j={j_alt}$).'
                    )
                    continue
                qhj_wrong = int(F(h0j) / F(h1j))
                st.markdown(f'$H_{{0,{j_alt}}}/H_{{1,{j_alt}}} = {h0j}/{h1j} = {qhj_wrong}$')
        pos_bits = []
        for i in range(N):
            if i == j_hat:
                pos_bits.append(f'**[{i}]**')
            else:
                pos_bits.append(f'{i}')
        st.markdown('**Pozíciók (0…7):** ' + '\u2003'.join(pos_bits))
    elif np.all(s == 0):
        st.success('Nulla szindróma, nincs hiba.')
    else:
        st.info(
            'Az **s = r·Hᵀ** egyetlen **H**-oszlopra illesztése **nem** ad **j**-t (gyakori **két hiba** esetén). '
            'A lenti **S₁…S₄** (értékelési / Peterson-gyök, **Hankel-determináns**) szindrómák és a **Javítás** fül ezek alapján mindkét hibát **megtalálhatjuk és javíthatjuk**.'
        )
        _render_peterson_evaluation_syndromes_section(pet_ints)
with tab_dec:
    st.subheader('Hibajavítás (legfeljebb 2 szimbólum), c visszaállítása')
    r_dec_ints = core.gf_row_to_ints(r)
    st.caption('[8,4] RS, $d_{\\min}=5$, legfeljebb 2 szimbólumhiba. Az **s = r·Hᵀ** és az $S_i$ szindrómák nem ugyanazok.')
    st.write('**Fogadott szó r** = [r₀,…,r₇] (int 0…15):', core.format_int_row(r_dec_ints))
    st.write('**r** 32 bites sorozat (r₀→r₇, 4 bitenként szóközzel):', core.bits32_spaced_c0_to_c7(r_dec_ints))
    _render_pgz_rs84_algorithm_steps(r_dec_ints)
    s_dec = core.syndrome_row(r, H)
    if corrupt and num_errors >= 3:
        st.warning('**3** injektált hiba esetén az **S₁…S₄** alapú dekóder nem garantált.')
    dec_result = core.decode_rs84(r_dec_ints)
    if dec_result is not None:
        c_hat_ints, _e_hat_ints = dec_result
        match = list(c_hat_ints) == [int(x) for x in c.flatten()]
        pet_zero = all(v == 0 for v in core.compute_evaluation_syndromes_ints(c_hat_ints))
        if not pet_zero:
            st.error('Belső ellenőrzés: a helyreállított **ĉ** **S₁…S₄** szindrómái nem nullák — dekódolási hiba.')
        elif corrupt and match:
            st.success('A javított kód szó megegyezik az eredeti **c**-vel. A benne szereplő **m̂** megegyezik az eredeti **m**-mel.')
        elif not corrupt:
            st.success('Hibátlan eset (S₁…S₄ = 0, **ĉ** = **r**).')
        else:
            st.error('A javított szó nem egyezik a küldött **c**-vel (túl sok hiba vagy nem javítható minta).')
    elif np.all(s_dec == 0):
        st.info(
            f'**s** = [0, 0, 0, 0] → nincs észlelt hiba az **H** szerint; a helyes kód szó egyezik a fogadottal: **c** = **r** '
            f'(int: {core.format_int_row(r_dec_ints)}; 4 bit / pozíció: {core.format_gf16_symbols_as_bits(r_dec_ints)}; '
            f'32 bit (4 bitenként szóközzel): {core.bits32_spaced_c0_to_c7(r_dec_ints)}).'
        )
    else:
        st.error(
            'Az **S₁…S₄** alapú dekóder **nem** talált megoldást — pl. **3 vagy több** szimbólumhiba, '
            'vagy ritka **hamis** minta (pl. degenerált Hankel-mátrix, nem megfelelő számú gyök a hibahelypolinomnak, szinguláris Vandermonde).'
        )
