from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

import rs74_core as rc
import rs74_explain as ex

APP_DIR = Path(__file__).resolve().parent


def _set_letters(letters: tuple[str, str, str]) -> None:
    for i, ch in enumerate(letters):
        st.session_state[f'letter{i}'] = ch

def on_click_test1() -> None:
    _set_letters(('B', 'C', 'D'))

def on_click_test2() -> None:
    _set_letters(('A', 'A', 'A'))

def on_click_test3() -> None:
    rng = np.random.default_rng()
    for i in range(3):
        st.session_state[f'letter{i}'] = rng.choice(rc.LETTER_ORDER)
_SIDEBAR_PARITY_BALRA_LABEL = 'Balra: [p₀,p₁,p₂ | m₀,m₁,m₂,m₃]  →  G = [P | I₄]'
_SIDEBAR_INJ_MODE_KOZVETLEN = 'Közvetlen fogadott érték: r[j] = választott 3 bit (0…7)'
_SIDEBAR_INJ_MODE_OSSZEADAS = 'Összeadás: r[j] = c[j] + e (e ≠ 0)'

def _apply_dolgozat_alapadatok_preset() -> None:
    st.session_state['letter0'] = 'B'
    st.session_state['letter1'] = 'A'
    st.session_state['letter2'] = 'D'
    st.session_state['parity_order'] = _SIDEBAR_PARITY_BALRA_LABEL
    st.session_state['corrupt'] = True
    st.session_state['num_symbol_errors'] = 1
    st.session_state['error_inj_mode'] = _SIDEBAR_INJ_MODE_KOZVETLEN
    st.session_state['err_pos_0'] = 3
    st.session_state['recv_sym_0'] = 5

def _reset_sidebar_startup_defaults() -> None:
    st.session_state['letter0'] = rc.LETTER_ORDER[1]
    st.session_state['letter1'] = rc.LETTER_ORDER[2]
    st.session_state['letter2'] = rc.LETTER_ORDER[3]
    st.session_state['parity_order'] = _SIDEBAR_PARITY_BALRA_LABEL
    st.session_state['corrupt'] = False
    st.session_state['num_symbol_errors'] = 1
    st.session_state['error_inj_mode'] = _SIDEBAR_INJ_MODE_KOZVETLEN
    for i in range(3):
        st.session_state[f'err_pos_{i}'] = min(i, rc.N - 1)
        st.session_state[f'err_mag_{i}'] = 1
        st.session_state[f'recv_sym_{i}'] = 0
st.set_page_config(page_title='RS(7,4) – hibajavító kódolás', layout='wide')
st.title('RS(7,4) – hibajavító kódolás')
st.markdown(f'**Műveleti test:** GF(2³), irreducibilis polinom **x³ + x + 1**. **Generátorpolinom** (a megadott alak): $g(x)=x^3+(\\alpha^2+1)x^2+\\alpha x+(\\alpha^2+1)$. A kód hossza **n = {rc.N}**, üzenet **k = {rc.K}**, paritás **n − k = {rc.N - rc.K}**.')
with st.sidebar:
    _dolgozat_prev = st.session_state.get('_dolgozat_checkbox_prev', False)
    dolgozat_alap = st.checkbox('A dolgozat hibaértékeinek használata', value=False, key='dolgozat_alapadatok')
    if dolgozat_alap and (not _dolgozat_prev):
        _apply_dolgozat_alapadatok_preset()
    elif not dolgozat_alap and _dolgozat_prev:
        _reset_sidebar_startup_defaults()
    st.session_state['_dolgozat_checkbox_prev'] = dolgozat_alap
    st.header('Bemenetek')
    st.subheader('3 szimbólum (A–H) + padding')
    st.caption('Minden betűhöz 3 bit; a **4. üzenetszimbólum** padding: **A (000) → 0**.')
    chosen: list[str] = []
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            chosen.append(st.selectbox(f'#{i + 1}', rc.LETTER_ORDER, index=i + 1, key=f'letter{i}'))
    pad_int = 0
    m_vals = [rc.letter_to_gf_int(ch) for ch in chosen] + [pad_int]
    st.subheader('G mátrix: paritás bal / jobb')
    parity_sel = st.radio('A kód szó c = m·G sorrendje (ugyanaz a kód, más oszlop-permutáció)', (_SIDEBAR_PARITY_BALRA_LABEL, 'Jobbra: [m₀,m₁,m₂,m₃ | p₀,p₁,p₂]  →  G = [I₄ | P]'), index=0, key='parity_order')
    parity_right = parity_sel.startswith('Jobbra')
    st.subheader('Hibák injektálása (1–3 szimbólum)')
    corrupt = st.checkbox('Hiba beszúrása', value=False, key='corrupt')
    inj_mode = _SIDEBAR_INJ_MODE_KOZVETLEN
    num_errors = 1
    err_pos_list: list[int] = [0]
    err_mag_list: list[int] = [1]
    recv_sym_list: list[int] = [0]
    if corrupt:
        inj_mode = st.radio('Hiba beállítás módja (minden hibára azonos)', (_SIDEBAR_INJ_MODE_KOZVETLEN, _SIDEBAR_INJ_MODE_OSSZEADAS), index=0, key='error_inj_mode')
        num_errors = int(st.selectbox('Hibák száma', [1, 2, 3], index=0, key='num_symbol_errors'))
        err_pos_list = []
        err_mag_list = []
        recv_sym_list = []
        for i in range(num_errors):
            st.markdown(f'**Hiba {i + 1} / {num_errors}**')
            p = st.selectbox(f'Pozíció j (0…6) — hiba {i + 1}', list(range(rc.N)), index=min(i, rc.N - 1), key=f'err_pos_{i}')
            err_pos_list.append(int(p))
            if inj_mode.startswith('Összeadás'):
                em = st.selectbox(f'Hiba e (nemnulla) — hiba {i + 1}', list(range(1, 8)), index=0, format_func=rc.gf_symbol_select_label, key=f'err_mag_{i}')
                err_mag_list.append(int(em))
            else:
                rs = st.selectbox(f'Fogadott r[j] — hiba {i + 1}', list(range(8)), index=0, format_func=rc.gf_symbol_select_label, key=f'recv_sym_{i}')
                recv_sym_list.append(int(rs))
with st.expander('GF(8) elemek: karakter ↔ 3 bit ↔ polinom ↔ α hatvány', expanded=False):
    st.caption('A szimbólumok és a mezőelemek megfeleltetése (primitív elem **α**, irreducibilis polinom **x³+x+1**). **Sorrend:** **A → H** betűrend (a **karakter** oszlop szerint); az **α hatvány** oszlop így nem növekvő kitevő szerinti. A **G** és **H** együtthatói ugyanazokkal a 0–7 értékekkel számolhatók. Utolsó oszlop: a sorhoz tartozó **GF(8) egész** (mind a nyolc érték **0…7** pontosan egyszer).')
    st.dataframe(rc.gf8_element_table_rows(), use_container_width=True, hide_index=True)
with st.expander('GF(2³) szorzás és összeadás (8×8)', expanded=False):
    st.caption('**Sor / oszlopfejléc:** **a** és **b** (int **0…7**), zárójelben az **α-hatvány** alak (ugyanaz, mint a fenti GF(8) táblázatnál). **Összeadás:** char 2 → a bitek **XOR**-ja (ugyanaz, mint **a ⊕ b** az int reprezentáción). **Szorzás:** mezőbeli **a·b**. Irreducibilis polinom: **x³+x+1** (**galois** GF(2³)).')
    st.markdown(ex.gf8_arithmetic_tables_html(), unsafe_allow_html=True)
G, H = rc.permute_columns_parity_order(rc.G_BASE, rc.H_BASE, parity_right)
m = rc.GF(m_vals).reshape(1, rc.K)
c = m @ G
r = c.copy()
if corrupt:
    r = r.copy()
    for hi in range(num_errors):
        pos = err_pos_list[hi]
        if inj_mode.startswith('Összeadás'):
            r[0, pos] = r[0, pos] + rc.GF(err_mag_list[hi])
        else:
            r[0, pos] = rc.GF(recv_sym_list[hi])
e = r - c
tab_g, tab_enc, tab_err, tab_syn, tab_dec = st.tabs(['Alapadatok', 'Kódolás', 'Fogadott szó és hiba', 'Szindróma', 'Javítás / dekódolás'])
with tab_g:
    st.subheader('Generátorpolinom g(x)')
    st.latex('g(x)=(x-\\alpha)(x-\\alpha^2)(x-\\alpha^3)')
    st.latex('g(x)=x^3+(\\alpha^2+1)x^2+\\alpha x+(\\alpha^2+1)')
    st.subheader('Generátor mátrix G (4 × 7)')
    if parity_right:
        st.caption('**Paritás jobbra:** G = [I₄ | P] (első 4 oszlop egységmátrix). c = m · G → [m₀…m₃ | p₀…p₂].')
    else:
        st.caption('Bal oldalt 3 paritásoszlop, jobb oldalt 4×4 egységmátrix. A sorokat úgy kapod, hogy veszed 1, x, x², x³ üzenetbázist, majd mindegyiket megszorzod x³-mal (x^{n−k}, itt n−k=3), és g(x)-re osztva a maradék adja a paritásrészt. Vagyis: x³, x⁴, x⁵, x⁶ maradékait kell kiszámolni g(x)-re modulo.')
    st.dataframe(np.array(G, dtype=int), use_container_width=True)
    st.latex('G = \\begin{bmatrix} ' + ex.format_gf_matrix(G) + ' \\end{bmatrix}')
    st.subheader('Paritás-mátrix H (3 × 7)')
    if parity_right:
        st.caption('**Szisztematikus H a G-ből:** ha **G = [I₄ | P]**, akkor **H = [Pᵀ | I₃]** (3×7). **H = H_base · Π** a paritás bal/jobb váltásnál.')
    else:
        st.caption('**Szisztematikus H a G-ből:** ha **G = [P | I₄]**, akkor **H = [I₃ | Pᵀ]** (3×7). **H = H_base · Π** a paritás bal/jobb váltásnál.')
    st.dataframe(np.array(H, dtype=int), use_container_width=True)
    st.latex('H = \\begin{bmatrix} ' + ex.format_gf_matrix(H) + ' \\end{bmatrix}')
with tab_enc:
    st.subheader('Üzenet és kódolás')
    st.markdown('Választott szimbólumok: **' + ', '.join(chosen) + '**, majd **padding** a 4. helyen (**A** → 0).')
    rows_enc = []
    for i, ch in enumerate(chosen):
        v = rc.letter_to_gf_int(ch)
        row = {'Pozíció': i + 1, 'Betű': ch}
        row.update(rc.gf_int_to_labels(v))
        rows_enc.append(row)
    rowp = {'Pozíció': 4, 'Betű': 'A (padding)'}
    rowp.update(rc.gf_int_to_labels(0))
    rows_enc.append(rowp)
    st.dataframe(rows_enc, use_container_width=True)
    m_ints_enc = rc.gf_row_to_ints(m)
    st.markdown('**m** (int, m₀…m₃): ' + rc.format_gf8_int_tuple(m_ints_enc))
    st.markdown('**m** (bitsorozat): ' + rc.format_gf8_bits_tuple(m_ints_enc))
    st.markdown('### Előállítás: $\\mathbf{c} = \\mathbf{m} \\cdot G$')
    c_ints_enc = rc.gf_row_to_ints(c)
    m_row_tex = ' & '.join((str(v) for v in m_ints_enc))
    g_tex_inner = ex.format_gf_matrix(G)
    c_tex_inner = ' & '.join((str(v) for v in c_ints_enc))
    row_top_l, row_top_r = ex.streamlit_cols_m_g()
    with row_top_l:
        st.empty()
    with row_top_r:
        st.latex('\\begin{bmatrix} ' + g_tex_inner + ' \\end{bmatrix}')
    row_bot_l, row_bot_r = ex.streamlit_cols_m_g()
    with row_bot_l:
        st.latex('\\begin{bmatrix} ' + m_row_tex + ' \\end{bmatrix}')
    with row_bot_r:
        st.latex('\\begin{bmatrix} ' + c_tex_inner + ' \\end{bmatrix}')
    ex.render_encoding_m_dot_g_expander(m, G, parity_right=parity_right)
    st.subheader('Kód szó α alakban')
    alpha_row = [rc.INT_TO_ALPHA_STR[v] for v in rc.gf_row_to_ints(c)]
    st.write(', '.join((f'c{j}={s}' for j, s in enumerate(alpha_row))))
    ci = rc.gf_row_to_ints(c)
    st.subheader('Polinom és bit-sorrend (c₀ vs c₆ elöl)' if rc.SHOW_KODOLAS_C6_DESCENDING else 'Polinom és bit-sorrend (c₀…c₆)')
    st.markdown('**Polinom** $c(x) = m(x)\\cdot x^3 + r(x)$.')
    st.markdown('A kód szó polinomja: $c(x) = c_0 + c_1 x + c_2 x^2 + \\cdots + c_6 x^6$. Minden **cᵢ** ennek az együtthatója (0…7 → 3 bit).')
    if rc.SHOW_KODOLAS_C6_DESCENDING:
        st.info('**Miért tűnik másnak a 21 bit?** Ugyanaz a hét GF(8) szimbólum két szokásos sorrendben: **c₀→c₆** (konstans először) vs **c₆→c₀** (x⁶ együttható először). A **blokkok** `[111,011,…]` típusú felírás általában **c₆…c₀** (csökkenő fok). A korábbi `001000011000111011000` példa **c₀…c₆** sorrendű összefűzés; nem ugyanaz, mint a `111,011,…` **c₆…c₀** listából összerakott sorozat — nem fordítás, **más a 3×7 blokk sorrendje**.')
    if rc.SHOW_KODOLAS_CI_INT_LIST:
        st.write('**Együtthatók c₀…c₆ (integer):**', ci)
    st.write('**Blokkok [c₀,…,c₆]:**', rc.bracket_groups_bits(ci, descending=False))
    if rc.SHOW_KODOLAS_C6_DESCENDING:
        st.write('**Blokkok [c₆,…,c₀] (x⁶ együttható balra):**', rc.bracket_groups_bits(ci, descending=True))
    st.write('**21 bit, sorrend c₀→c₆ (konstans először):**', rc.bits21_c0_to_c6(ci))
    if rc.SHOW_KODOLAS_C6_DESCENDING:
        st.write('**21 bit, sorrend c₆→c₀ (x⁶ először, jegyzet-barát):**', rc.bits21_c6_to_c0(ci))
    st.markdown('#### Az együtthatók kiszámolása a GF(2³) szorzás és összeadás (8×8) táblázat alapján (aktuális üzenet) (c₀…c₆)')
    for j in range(rc.N):
        st.markdown(ex.markdown_line_cj_from_m_dot_g(m_vals, G, j))
    if parity_right:
        st.info('**Paritás jobbra:** **c = m·G** = **[m₀,m₁,m₂,m₃ | r₀,r₁,r₂]** — a polinom **c₀…c₆** sorrendje ettől eltér (először a maradék alacsony fokú tagjai). Kézi ellenőrzéshez használd a polinom **c₀…c₆** együtthatóit (növekvő fok szerint).')
with tab_err:
    st.subheader('Fogadott vektor és hibavektor')
    c_ints_err = rc.gf_row_to_ints(c)
    r_ints_err = rc.gf_row_to_ints(r)
    e_ints_err = rc.gf_row_to_ints(e)
    st.markdown('**c** (küldött) = `' + rc.format_int_row(c_ints_err) + '`  \n**r** (fogadott) = `' + rc.format_int_row(r_ints_err) + '`  \n**e** = **r** − **c** = `' + rc.format_int_row(e_ints_err) + '`')
    st.subheader('21 bites reprezentáció (c₀…c₆)')
    st.caption('Minden szimbólum 3 bit (b₂ b₁ b₀); összesen 21 bit, sorrend c₀→c₆. Alább: folytonos bitsor; a második blokk szóközzel tagolva (szimbólumonként 3 bit).')
    c21 = rc.bits21_c0_to_c6(c_ints_err)
    r21 = rc.bits21_c0_to_c6(r_ints_err)
    e21 = rc.bits21_c0_to_c6(e_ints_err)
    c21s = rc.bits21_spaced_c0_to_c6(c_ints_err)
    r21s = rc.bits21_spaced_c0_to_c6(r_ints_err)
    e21s = rc.bits21_spaced_c0_to_c6(e_ints_err)
    lbl = 16
    st.code(f"{'c (küldött):':<{lbl}}{c21}\n{'r (fogadott):':<{lbl}}{r21}\n{'e = r − c:':<{lbl}}{e21}\n\n{'c (küldött):':<{lbl}}{c21s}\n{'r (fogadott):':<{lbl}}{r21s}\n{'e = r − c:':<{lbl}}{e21s}", language=None)
    if corrupt:
        lines: list[str] = []
        for hi in range(num_errors):
            j = err_pos_list[hi]
            cj = int(c[0, j])
            rj = int(r[0, j])
            ej = int(r[0, j] - c[0, j])
            if inj_mode.startswith('Összeadás'):
                em = err_mag_list[hi]
                lines.append(f'Hiba **{hi + 1}**: j = **{j}**, **e** = **{em}** = **{rc.INT_TO_ALPHA_STR[em]}** ({rc.int_to_bits3(em)}); c[j]={cj} ({rc.int_to_bits3(cj)}), r[j]={rj} ({rc.int_to_bits3(rj)}).')
            else:
                lines.append(f'Hiba **{hi + 1}**: j = **{j}**, r[j] = **{rj}** = **{rc.INT_TO_ALPHA_STR[rj]}** ({rc.int_to_bits3(rj)}); c[j]={cj} ({rc.int_to_bits3(cj)}); **e** = r−c = **{ej}** = **{rc.INT_TO_ALPHA_STR[ej & 7]}**.')
        st.markdown('  \n'.join(lines))
        if num_errors >= 2:
            st.warning('**2 vagy 3 szimbólumhiba** esetén a [7,4] RS kód **nem** tud javítani (a minimális távolság 4 → legfeljebb **egy** hiba javítható).')
    else:
        st.success('Nincs szándékos hiba: r = c.')
with tab_syn:
    st.subheader('Szindróma számítás')
    if corrupt and num_errors >= 2:
        st.caption('Több szimbólumhiba esetén a szindróma általában **nem** írható le egyetlen [pozíció, hiba] párral; az „egy-hibás” illesztés nem megbízható.')
    s = rc.syndrome_row(r, H)
    s_ints = rc.gf_row_to_ints(s)
    r_ints_syn = rc.gf_row_to_ints(r)
    c_ints_syn = rc.gf_row_to_ints(c)
    r_row_tex = ' & '.join((str(v) for v in r_ints_syn))
    s_row_tex = ' & '.join((str(v) for v in s_ints))
    h_t_tex = ex.format_gf_matrix(H.T)
    s_c = rc.syndrome_row(c, H)
    s_c_ints = rc.gf_row_to_ints(s_c)
    c_row_tex = ' & '.join((str(v) for v in c_ints_syn))
    s_c_row_tex = ' & '.join((str(v) for v in s_c_ints))
    st.markdown('### Helyes (küldött) kód szó: $\\mathbf{0} = \\mathbf{c} \\, H^{\\mathsf{T}}$')
    _syn_lr = (5, 6)
    c_top_l, c_top_r = ex.streamlit_cols_m_g(*_syn_lr)
    with c_top_l:
        st.empty()
    with c_top_r:
        st.latex('H^{\\mathsf{T}} = \\begin{bmatrix} ' + h_t_tex + ' \\end{bmatrix}')
    c_bot_l, c_bot_r = ex.streamlit_cols_m_g(*_syn_lr)
    with c_bot_l:
        st.latex('\\mathbf{c} = \\begin{bmatrix} ' + c_row_tex + ' \\end{bmatrix}')
    with c_bot_r:
        st.latex('\\mathbf{c} \\, H^{\\mathsf{T}} = \\begin{bmatrix} ' + s_c_row_tex + ' \\end{bmatrix}')
    ex.render_s0_expander(c, H)
    st.markdown('### Fogadott szó: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$')
    syn_top_l, syn_top_r = ex.streamlit_cols_m_g(*_syn_lr)
    with syn_top_l:
        st.empty()
    with syn_top_r:
        st.latex('H^{\\mathsf{T}} = \\begin{bmatrix} ' + h_t_tex + ' \\end{bmatrix}')
    syn_bot_l, syn_bot_r = ex.streamlit_cols_m_g(*_syn_lr)
    with syn_bot_l:
        st.latex('\\mathbf{r} = \\begin{bmatrix} ' + r_row_tex + ' \\end{bmatrix}')
    with syn_bot_r:
        st.latex('\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}} = \\begin{bmatrix} ' + s_row_tex + ' \\end{bmatrix}')
    st.markdown('**s** GF(8) int (0–7): `' + rc.format_int_row(s_ints) + '`  \n**α hatvány alak** (s₀, s₁, s₂): ' + ', '.join((rc.INT_TO_ALPHA_POWER_STR[v] for v in s_ints)) + '  \n**Polinom alak** (ugyanazok az elemek): ' + ', '.join((rc.INT_TO_ALPHA_STR[v] for v in s_ints)) + '  \n**Megjegyzés:** **s = r·Hᵀ** a **szisztematikus H**-val (lásd **Alapadatok** fül).')
    ex.render_syndrome_r_dot_Ht_expander(r, H)
    j_hat, a_hat = rc.single_error_from_syndrome(s.flatten(), H)
    if j_hat is not None:
        st.subheader('GF(8) osztások — hibahely meghatározás, oszlop-illesztés')
    st.markdown(f'### Szindróma és **H** oszlop (egy hibánál)\n\n**Tehát** `{rc.format_int_row(s_ints)}` „**kijelöli**” azt az **H**-**oszlopot** (**j**), amelyre **létezik** olyan **ε ∈ GF(8)**, hogy **s₀ = ε·H₀,ⱼ**, **s₁ = ε·H₁,ⱼ**, **s₂ = ε·H₂,ⱼ** (mind GF(8)).')
    if np.all(s == 0):
        st.caption('**s = 0:** nincs oszlophoz illesztendő nemtriviális szindróma.')
    if j_hat is not None:
        s_f = s.flatten()
        h_col = H[:, j_hat]
        s0, s1 = (int(s_f[0]), int(s_f[1]))
        q_ratio_s: Optional[int] = int(rc.GF(s0) / rc.GF(s1)) if s1 != 0 else None
        ratio_lines = []
        for i in range(rc.N - rc.K):
            hij = int(h_col[i])
            si = int(s_f[i])
            if hij == 0:
                ratio_lines.append(
                    f'$H_{{{i},{j_hat}}}=0$ — $s_{i}/H_{{{i},{j_hat}}}$ **nem** értelmezhető (**nevező** $0$).'
                )
                continue
            ai = rc.GF(si) / rc.GF(hij)
            ratio_lines.append(f'$s_{i}/H_{{{i},{j_hat}}} = {si}/{hij} = {int(ai)}$')
        st.markdown('  \n'.join(ratio_lines))
        st.markdown(f'Minden **értelmezett** $s_i/H_{{i,j}}$ = **ε** = **{int(a_hat)}**; $j={j_hat}$.')
        with st.expander('További H-oszlopok (j ≠ ' + str(j_hat) + '): ugyanazok a sᵢ/Hᵢ,ⱼ számítások', expanded=False):
            st.markdown('##### **sᵢ/Hᵢ,ⱼ** minden sorra (j ≠ ' + str(j_hat) + ')')
            for j_alt in sorted(j for j in range(rc.N) if j != j_hat):
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
                q_h_main = int(rc.GF(h0c) / rc.GF(h1c))
                st.markdown(f'$H_{{0,{j_hat}}}/H_{{1,{j_hat}}} = {h0c}/{h1c} = {q_h_main}$')
            else:
                st.caption(
                    f'$H_{{1,{j_hat}}}=0$ — **H₀,ⱼ/H₁,ⱼ** nem értelmezett, mert **H₁,ⱼ** = 0 ($j={j_hat}$).'
                )
            st.markdown('**A nem megfelelő értékek.**')
            for j_alt in sorted(j for j in range(rc.N) if j != j_hat):
                st.markdown(f'**H{j_alt}.** oszlop (**j = {j_alt}**):')
                h0j = int(H[0, j_alt])
                h1j = int(H[1, j_alt])
                if h1j == 0:
                    st.caption(
                        f'$H_{{1,{j_alt}}}=0$ — **H₀,ⱼ/H₁,ⱼ** nem értelmezett, mert **H₁,ⱼ** = 0 ($j={j_alt}$).'
                    )
                    continue
                qhj_wrong = int(rc.GF(h0j) / rc.GF(h1j))
                st.markdown(f'$H_{{0,{j_alt}}}/H_{{1,{j_alt}}} = {h0j}/{h1j} = {qhj_wrong}$')
        pos_bits = []
        for i in range(rc.N):
            if i == j_hat:
                pos_bits.append(f'**[{i}]**')
            else:
                pos_bits.append(f'{i}')
        st.markdown('**Pozíciók (0…6):** ' + '\u2003'.join(pos_bits))
    elif np.all(s == 0):
        st.success('Nulla szindróma, nincs hiba.')
    else:
        st.warning('A szindrómához nem található "j" érték.')
with tab_dec:
    st.subheader('Egy hiba javítása, c visszaállítása.')
    s_dec = rc.syndrome_row(r, H)
    s_dec_ints = rc.gf_row_to_ints(s_dec)
    r_dec_ints = rc.gf_row_to_ints(r)
    st.markdown('A fogadott szó **r** és a szindróma együtt határozza meg a javítást. Egy nemnulla szimbólumhiba esetén $\\mathbf{r}=\\mathbf{c}+\\boldsymbol{\\varepsilon}$, ahol $\\boldsymbol{\\varepsilon}$ csak a **j**. pozíción nem **0**; A **j** és $\\varepsilon$ megtalálása után: $\\hat{c}_i=r_i$ ha $i\\neq j$, és $\\hat{c}_j=r_j-\\varepsilon$ (GF(8)).')
    st.write('**Fogadott szó r** = [r₀,…,r₆] (int 0…7):', rc.format_int_row(r_dec_ints))
    st.write('**r** 21 bites sorozat (r₀→r₆, 3 bitenként szóközzel):', rc.bits21_spaced_c0_to_c6(r_dec_ints))
    st.write('**Szindróma s** = [s₀, s₁, s₂] (int):', rc.format_int_row(s_dec_ints))
    if corrupt and num_errors >= 2:
        st.warning('**2 vagy 3** szimbólumhiba nem javítható.')
    j_hat, a_hat = rc.single_error_from_syndrome(s_dec.flatten(), H)
    if j_hat is not None:
        a_hat_int = int(a_hat)
        st.markdown('#### A szindrómából kapott hibahely és nagyság')
        st.write(f'**ĵ** = {j_hat} (hiba pozíció), **ε̂** = {a_hat_int} (hiba nagyság, int)')
        e_manual = rc.GF([0] * rc.N)
        e_manual[j_hat] = a_hat
        c_manual = r.flatten() - e_manual
        c_hat_ints = rc.gf_row_to_ints(c_manual.reshape(1, -1))
        e_hat_ints = rc.gf_row_to_ints(e_manual.reshape(1, -1))
        st.markdown('**r** − **ε̂** = **c** GF(8)-ban; int értékekkel.')
        st.code(rc.format_r_epsilon_hat_c_aligned_block(r_dec_ints, e_hat_ints, c_hat_ints), language=None)
        st.write('**Helyreállított kódszó c** = [c₀,…,c₆] (int 0…7):', rc.format_int_row(c_hat_ints))
        st.write('**c** 21 bites sorozat (c₀→c₆, 3 bitenként szóközzel):', rc.bits21_spaced_c0_to_c6(c_hat_ints))
        cf = c_manual.flatten()
        if parity_right:
            m_dec_ints = [int(cf[i]) & 7 for i in range(rc.K)]
        else:
            m_dec_ints = [int(cf[i]) & 7 for i in range(rc.N - rc.K, rc.N)]
        st.markdown('**m** (int, m₀…m₃): ' + rc.format_gf8_int_tuple(m_dec_ints))
        st.markdown('**m** (bitsorozat): ' + rc.format_gf8_bits_tuple(m_dec_ints))
        match = np.array_equal(c_manual, c.flatten())
        if corrupt and match:
            st.success('A javított kód szó megegyezik az eredeti **c**-vel. A benne szereplő **m** érték megegyezik az eredeti **m** értékkel.')
        elif not corrupt:
            st.success('Hibátlan eset.')
        else:
            st.error('A javított szó nem egyezik a küldött **c**-vel.')
    elif np.all(s_dec == 0):
        st.info(f'**s** = [0, 0, 0] → nincs észlelt hiba az **H** szerint; a helyes kód szó egyezik a fogadottal: **c** = **r** (int: {rc.format_int_row(r_dec_ints)}; 3 bit / pozíció: {rc.format_gf8_symbols_as_bits(r_dec_ints)}; 21 bit (3 bitenként szóközzel): {rc.bits21_spaced_c0_to_c6(r_dec_ints)}).')
st.caption(f'`cd "{APP_DIR}"` — `py -m streamlit run "rs74_app.py"` (vagy `app.py`)')
