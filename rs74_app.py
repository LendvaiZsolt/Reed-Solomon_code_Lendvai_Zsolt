from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import nav_visibility
import rs74_core as rc
import rs74_explain as ex

# Vandermonde / kiértékelési H (s = r·Hᵀ); Streamlit Cloud-on néha régi rs74_core marad a cache-ben.
_H_RS74_PARITY_EVAL_ROWS = [[1, 2, 4, 3, 6, 7, 5], [1, 4, 6, 5, 2, 3, 7], [1, 3, 5, 4, 7, 2, 6]]
if not hasattr(rc, 'H_RS74_PARITY_EVAL'):
    try:
        importlib.reload(rc)
    except Exception:
        pass
if not hasattr(rc, 'H_RS74_PARITY_EVAL'):
    rc.H_RS74_PARITY_EVAL = rc.GF(_H_RS74_PARITY_EVAL_ROWS)

importlib.reload(ex)

APP_DIR = Path(__file__).resolve().parent


def _call_render_g_parity_mod_g_long_division_expander(*, parity_right: bool) -> None:
    """Streamlit Cloud: néha elavult `rs74_explain` marad a cache-ben; reload + opcionális fájlból exec."""
    fn = getattr(ex, 'render_g_parity_mod_g_long_division_expander', None)
    disk_err: str | None = None
    if fn is None:
        path = APP_DIR / 'rs74_explain.py'
        spec = importlib.util.spec_from_file_location('_rs74_explain_disk', path)
        if spec is not None and spec.loader is not None:
            disk_mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(disk_mod)
            except Exception as e:
                disk_err = repr(e)
            else:
                fn = getattr(disk_mod, 'render_g_parity_mod_g_long_division_expander', None)
    if fn is not None:
        fn(parity_right=parity_right)
        return
    with st.expander('Polinomosztás g(x) szerint (részletek)', expanded=False):
        st.warning(
            'A Cloud még nem a friss kódot futtatja, vagy nem a jó GitHub-repó / **main** ág van bekötve. '
            'Streamlit: **App settings → Branch = main**, majd **Reboot app** (Manage app).'
        )
        st.caption('Betöltött `rs74_explain` (debug):')
        st.code(getattr(ex, '__file__', 'ismeretlen'))
        if disk_err is not None:
            st.caption('Másodlagos betöltés a lemezről:')
            st.code(disk_err)


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


def _gf8_log_alpha(v: int) -> int | None:
    x = int(v) & 7
    if x == 0:
        return None
    alpha = rc.GF.primitive_element
    xv = rc.GF(x)
    for k in range(7):
        if alpha**k == xv:
            return k
    return None


def _gf8_to_alpha_latex(v: int) -> str:
    x = int(v) & 7
    if x == 0:
        return '0'
    k = _gf8_log_alpha(x)
    if k is None:
        return str(x)
    return f'\\alpha^{{{k}}}'


def _gf8_int_to_alpha_poly_latex(v: int) -> str:
    """GF(8) elem **polinom alakja α-val** (LaTeX): `INT_TO_ALPHA_STR` → `\\alpha`, kitevő `²` → `^{2}`."""
    t = rc.INT_TO_ALPHA_STR[int(v) & 7].replace('α', r'\alpha').replace('²', '^{2}')
    return t


def _render_dec_tab_single_error_position_derivation(*, s_ints: list[int], j_hat: int, r_ints: list[int], eps_int: int) -> None:
    """Egy szimbólumhiba: S₂/S₁ = S₁/S₀ = αʲ (Vandermonde H mellett); majd ε és javítás (GF(8), ⊕)."""
    s0, s1, s2 = (int(s_ints[0]) & 7, int(s_ints[1]) & 7, int(s_ints[2]) & 7)
    st.markdown('**Hiba pozíciójának meghatározása:**')
    st.latex(r'\frac{S_2}{S_1} = \frac{S_1}{S_0} = \alpha^j')
    if s0 == 0 or s1 == 0:
        st.caption('**S₀** vagy **S₁** = 0 esetén ez az egyszerű arány nem mindig használható; a **ĵ** értéket az alábbi **H**-oszlop illesztés adja.')
        return
    s0t = _gf8_to_alpha_latex(s0)
    s1t = _gf8_to_alpha_latex(s1)
    s2t = _gf8_to_alpha_latex(s2)
    q21 = int(rc.GF(s2) / rc.GF(s1))
    q10 = int(rc.GF(s1) / rc.GF(s0))
    st.latex(r'\frac{S_2}{S_1} = \frac{' + s2t + '}{' + s1t + '} = ' + _gf8_to_alpha_latex(q21))
    st.latex(r'\frac{S_1}{S_0} = \frac{' + s1t + '}{' + s0t + '} = ' + _gf8_to_alpha_latex(q10))
    st.markdown('**Tehát**')
    st.latex(r'\alpha^{j} = ' + _gf8_to_alpha_latex(q21) + r' \Rightarrow j = ' + str(int(j_hat)))
    rj = int(r_ints[int(j_hat)]) & 7
    bits = rc.int_to_bits3(rj)
    st.markdown(
        f'**0**-tól indexelve a hiba **j = {j_hat}** (a **{j_hat + 1}.** szimbólum a **c₀…c₆** / **r₀…r₆** sorrendben). '
        f'A fogadott **r[{j_hat}] = {rj}** int (**{bits}** három bit) a hibás szimbólum a fogadott vektorban.'
    )
    _render_dec_tab_error_magnitude_and_xor_correction(
        j_hat=int(j_hat), s0=s0, s1=s1, s2=s2, rj=rj, eps_int=int(eps_int) & 7
    )


def _render_dec_tab_error_magnitude_and_xor_correction(*, j_hat: int, s0: int, s1: int, s2: int, rj: int, eps_int: int) -> None:
    """ε = S₀·α^{-j} = S₁·α^{-2j} = S₂·α^{-3j} (mod α⁷=1); cⱼ = rⱼ + ε a GF(8)-ben (XOR a biteken)."""
    F = rc.GF
    a = F.primitive_element
    jj = int(j_hat)
    e = int(eps_int) & 7
    rj = int(rj) & 7
    exp_mj = (-jj) % 7
    exp_m2j = (-2 * jj) % 7
    exp_m3j = (-3 * jj) % 7
    chk0 = int(F(s0) * (a**exp_mj))
    chk1 = int(F(s1) * (a**exp_m2j))
    chk2 = int(F(s2) * (a**exp_m3j))
    if not (chk0 == chk1 == chk2 == e):
        st.caption('**Megjegyzés:** az alábbi lánc a szokásos egy-hibás képleteket illusztrálja; az **ε̂** értéke a fenti oszlop-illesztésből számolva: **' + str(e) + '**.')
    k0 = _gf8_log_alpha(s0)
    k1 = _gf8_log_alpha(s1)
    k2 = _gf8_log_alpha(s2)
    if k0 is None or k1 is None or k2 is None:
        return
    sum_exp01 = (k0 + exp_mj) % 7
    sum_exp12 = (k1 + exp_m2j) % 7
    sum_exp23 = (k2 + exp_m3j) % 7
    st.markdown('**Az eltérés meghatározása:**')
    st.latex(
        r'e_{'
        + str(jj)
        + r'} = S_0\,\alpha^{-j} = S_1\,\alpha^{-2j} = S_2\,\alpha^{-3j}\quad'
        + r'(\mathrm{mod}\ \alpha^7=1;\ j='
        + str(jj)
        + r')'
    )
    st.latex(
        r'e_{'
        + str(jj)
        + r'} = '
        + _gf8_to_alpha_latex(s0)
        + r'\,\alpha^{-'
        + str(jj)
        + r'} = '
        + _gf8_to_alpha_latex(s0)
        + r'\,\alpha^{'
        + str(exp_mj)
        + r'} = \alpha^{'
        + str(k0 + exp_mj)
        + r'} = \alpha^{'
        + str(sum_exp01)
        + r'} = '
        + _gf8_to_alpha_latex(e)
    )
    st.markdown('**Tehát:**')
    st.latex(
        r'\boxed{e_{'
        + str(jj)
        + r'} = '
        + _gf8_to_alpha_latex(e)
        + r' \leftrightarrow '
        + rc.int_to_bits3(e)
        + r' \leftrightarrow '
        + str(int(e))
        + r'}'
    )
    st.markdown(f'**A fogadott {jj + 1}. szimbólum** (**r_{{{jj}}}**):')
    st.latex(
        r'\boxed{r_{'
        + str(jj)
        + r'} = '
        + _gf8_int_to_alpha_poly_latex(rj)
        + r' \leftrightarrow '
        + rc.int_to_bits3(rj)
        + r' \leftrightarrow '
        + str(int(rj))
        + r'}'
    )
    cj = int(F(rj) + F(e))
    st.markdown('**Javítás (GF(8) összeadás = bitek XOR):**')
    st.latex(
        r'\boxed{c_{'
        + str(jj)
        + r'} = r_{'
        + str(jj)
        + r'} + e_{'
        + str(jj)
        + r'} = \bigl('
        + _gf8_int_to_alpha_poly_latex(rj)
        + r'\bigr) + \bigl('
        + _gf8_int_to_alpha_poly_latex(e)
        + r'\bigr) = '
        + _gf8_int_to_alpha_poly_latex(cj)
        + r' \leftrightarrow '
        + rc.int_to_bits3(cj)
        + r' \leftrightarrow '
        + str(int(cj) & 7)
        + r'}'
    )
    st.success('**Visszaáll az eredeti** kódszó-jegy a **j** pozíción (ellenőrzés: **c** ugyanitt).')


def _poly_y_latex_from_r(r_ints: list[int]) -> str:
    terms: list[str] = []
    for i, c in enumerate(r_ints):
        cc = _gf8_to_alpha_latex(c)
        if i == 0:
            terms.append(cc)
        elif i == 1:
            terms.append(f'{cc}x')
        else:
            terms.append(f'{cc}x^{i}')
    return ' + '.join(terms)


def _render_dynamic_syndrome_poly_eval_tab(r_ints: list[int]) -> None:
    alpha = rc.GF.primitive_element
    r_row = rc.GF([int(c) & 7 for c in r_ints]).reshape(1, -1)
    s_vals = rc.gf_row_to_ints(r_row @ rc.H_RS74_PARITY_EVAL.T)
    st.caption('**Szindróma int (0…7)** — **r·Hᵀ** (a LaTeX alatti számítással egyezik): `' + rc.format_int_row(s_vals) + '`')
    st.markdown('Adott:')
    st.latex(r'y(x)=' + _poly_y_latex_from_r(r_ints))
    st.markdown('GF(8)-ban:')
    st.latex(r'\alpha^3=\alpha+1,\qquad \alpha^7=1')
    st.markdown('A szindrómák:')
    st.latex(r'S_0=y(\alpha),\qquad S_1=y(\alpha^2),\qquad S_2=y(\alpha^3)')
    st.markdown('---')

    for s_idx in range(3):
        k = s_idx + 1
        eval_point = r'\alpha' if k == 1 else rf'\alpha^{{{k}}}'
        st.subheader(f'{s_idx + 1}. szindróma')
        st.latex(rf'S_{s_idx}=y({eval_point})')

        sub_terms: list[str] = []
        expo_terms: list[str] = []
        mod_terms: list[str] = []
        total = rc.GF(0)

        for i, c in enumerate(r_ints):
            ci = int(c) & 7
            if ci == 0:
                continue
            c_ltx = _gf8_to_alpha_latex(ci)
            sub_terms.append(f'{c_ltx}({eval_point})^{{{i}}}')

            c_pow = _gf8_log_alpha(ci)
            if c_pow is None:
                continue
            exp_raw = c_pow + k * i
            expo_terms.append(rf'\alpha^{{{exp_raw}}}')
            exp_mod = exp_raw % 7
            mod_terms.append(rf'\alpha^{{{exp_mod}}}')
            total += rc.GF(ci) * (rc.GF.primitive_element ** (k * i))

        has_mellek = bool(sub_terms) or bool(expo_terms)
        if has_mellek:
            with st.expander('Mellékszámítások', expanded=False):
                if sub_terms:
                    st.latex(rf'S_{s_idx}=' + ' + '.join(sub_terms))
                if expo_terms:
                    st.latex(rf'S_{s_idx}=' + ' + '.join(expo_terms))
                if mod_terms and mod_terms != expo_terms:
                    st.markdown('Modulo 7 szerint:')
                    st.latex(',\quad '.join((f'{a}={b}' for a, b in zip(expo_terms, mod_terms))))
                    st.markdown('tehát:')
                    st.latex(rf'S_{s_idx}=' + ' + '.join(mod_terms))
                st.markdown('így:')

        s_curr = int(s_vals[s_idx])
        st.latex(rf'\boxed{{S_{s_idx}={_gf8_to_alpha_latex(s_curr)}}}')
        if s_idx < 2:
            st.markdown('---')

    st.markdown('---')
    st.subheader('Végeredmény')
    s0, s1, s2 = s_vals
    st.latex(
        rf'\boxed{{S_0={_gf8_to_alpha_latex(s0)},\qquad S_1={_gf8_to_alpha_latex(s1)},\qquad S_2={_gf8_to_alpha_latex(s2)}}}'
    )
    st.markdown('Vagyis:')
    st.latex(rf'\boxed{{S=({_gf8_to_alpha_latex(s0)},\,{_gf8_to_alpha_latex(s1)},\,{_gf8_to_alpha_latex(s2)})}}')
    st.latex(rf'\boxed{{S=({s0},\,{s1},\,{s2})}}')
    p0, p1, p2 = (_gf8_int_to_alpha_poly_latex(s0), _gf8_int_to_alpha_poly_latex(s1), _gf8_int_to_alpha_poly_latex(s2))
    st.latex(rf'\boxed{{S=({p0},\,{p1},\,{p2})}}')


def _render_g_ht_zero_derivation(G, H) -> None:
    """GF(8)-ban: (G·Hᵀ)_{i,j} = Σ_k G_{i,k} H_{j,k} — mind a 12 elem 0."""
    st.subheader('Ellenőrzés: G · Hᵀ = 0')
    st.markdown(
        'A **(i, j)** elem a szorzatban: **(G·Hᵀ)ᵢ,ⱼ** = Σₖ **Gᵢ,ₖ · Hⱼ,ₖ** (szorzás és összeg **GF(8)**-ban). '
        'Ha mind a **4×3** elem **0**, akkor **G·Hᵀ** a nullamátrix — vagyis **H** minden sora ortogonális **G** minden sorára, tehát **H** paritás-ellenőrző mátrix ehhez a **G**-hez (oszloprendben egyező **c** = **m·G** mellett **c·Hᵀ = 0**).'
    )
    st.caption('Az alábbi egyenletekben a **+** jel **GF(8)-beli összeadást** jelent (karakterisztika 2 → bitek XOR-ja a 0…7 reprezentáción); a **·** szorzás **GF(8)-beli** szorzás.')
    for i in range(int(G.shape[0])):
        st.markdown(f'**G** **{i}.** sora (sorindex **{i}**):')
        for j in range(int(H.shape[0])):
            acc = rc.GF(0)
            parts: list[str] = []
            for k in range(int(G.shape[1])):
                gik = int(G[i, k])
                hjk = int(H[j, k])
                acc += rc.GF(gik) * rc.GF(hjk)
                parts.append(f'{gik}\\cdot {hjk}')
            joined = ' + '.join(parts)
            st.latex(rf'(G H^{{\mathsf{{T}}}})_{{{i},{j}}} = {joined} = {int(acc)}')


st.set_page_config(page_title='RS(7,4) – hibajavító kódolás', layout='wide')
st.title('RS(7,4) – hibajavító kódolás')
st.markdown(f'**Műveleti test:** GF(2³), irreducibilis polinom **x³ + x + 1**. **Generátorpolinom** (a megadott alak): $g(x)=x^3+(\\alpha^2+1)x^2+\\alpha x+(\\alpha^2+1)$. A kód hossza **n = {rc.N}**, üzenet **k = {rc.K}**, paritás **n − k = {rc.N - rc.K}**.')
with st.sidebar:
    st.page_link('app.py', label='RS(7,4)')
    st.page_link('pages/2_rs84.py', label='RS(8,4) GF(9)')
    if nav_visibility.SHOW_RS84_GF16_PAGE_LINK:
        st.page_link('pages/2_rs84_gf16.py', label='RS(8,4) GF(16) - törlésre kerül')
    st.page_link('pages/4_veletlen_hibateszt.py', label='Véletlen hibateszt')
    st.page_link('pages/3_dokumentacio.py', label='Használati útmutató')
    st.divider()
    _dolgozat_prev = st.session_state.get('_dolgozat_checkbox_prev', False)
    dolgozat_alap = st.checkbox('A dolgozat hibaértékeinek használata', value=False, key='dolgozat_alapadatok')
    if dolgozat_alap and (not _dolgozat_prev):
        _apply_dolgozat_alapadatok_preset()
    elif not dolgozat_alap and _dolgozat_prev:
        _reset_sidebar_startup_defaults()
    st.session_state['_dolgozat_checkbox_prev'] = dolgozat_alap
    _dolgozat_lock = bool(dolgozat_alap)
    st.header('Bemenetek')
    if _dolgozat_lock:
        st.info('**Dolgozat mód:** a bemenetek **zárolva** (a dolgozat példa szerint). Kapcsold ki a fenti jelölőnégyzetet a szerkesztéshez.')
    st.subheader('3 szimbólum (A–H) + padding')
    st.caption('Minden betűhöz 3 bit; a **4. üzenetszimbólum** padding: **A (000) → 0**.')
    chosen: list[str] = []
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            chosen.append(
                st.selectbox(f'#{i + 1}', rc.LETTER_ORDER, index=i + 1, key=f'letter{i}', disabled=_dolgozat_lock)
            )
    pad_int = 0
    m_vals = [rc.letter_to_gf_int(ch) for ch in chosen] + [pad_int]
    st.subheader('G mátrix: paritás bal / jobb')
    parity_sel = st.radio(
        'A kód szó c = m·G sorrendje (ugyanaz a kód, más oszlop-permutáció)',
        (_SIDEBAR_PARITY_BALRA_LABEL, 'Jobbra: [m₀,m₁,m₂,m₃ | p₀,p₁,p₂]  →  G = [I₄ | P]'),
        index=0,
        key='parity_order',
        disabled=_dolgozat_lock,
    )
    parity_right = parity_sel.startswith('Jobbra')
    st.subheader('Hibák injektálása (1 szimbólum)')
    corrupt = st.checkbox('Hiba beszúrása', value=False, key='corrupt', disabled=_dolgozat_lock)
    inj_mode = _SIDEBAR_INJ_MODE_KOZVETLEN
    num_errors = 1
    err_pos_list: list[int] = [0]
    err_mag_list: list[int] = [1]
    recv_sym_list: list[int] = [0]
    if corrupt:
        inj_mode = st.radio(
            'Hiba beállítás módja (minden hibára azonos)',
            (_SIDEBAR_INJ_MODE_KOZVETLEN, _SIDEBAR_INJ_MODE_OSSZEADAS),
            index=0,
            key='error_inj_mode',
            disabled=_dolgozat_lock,
        )
        if int(st.session_state.get('num_symbol_errors', 1)) != 1:
            st.session_state['num_symbol_errors'] = 1
        num_errors = int(st.selectbox('Hibák száma', [1], index=0, key='num_symbol_errors', disabled=_dolgozat_lock))
        err_pos_list = []
        err_mag_list = []
        recv_sym_list = []
        for i in range(num_errors):
            st.markdown(f'**Hiba {i + 1} / {num_errors}**')
            p = st.selectbox(
                f'Pozíció j (0…6) — hiba {i + 1}',
                list(range(rc.N)),
                index=min(i, rc.N - 1),
                key=f'err_pos_{i}',
                disabled=_dolgozat_lock,
            )
            err_pos_list.append(int(p))
            if inj_mode.startswith('Összeadás'):
                em = st.selectbox(
                    f'Hiba e (nemnulla) — hiba {i + 1}',
                    list(range(1, 8)),
                    index=0,
                    format_func=rc.gf_symbol_select_label,
                    key=f'err_mag_{i}',
                    disabled=_dolgozat_lock,
                )
                err_mag_list.append(int(em))
            else:
                rs = st.selectbox(
                    f'Fogadott r[j] — hiba {i + 1}',
                    list(range(8)),
                    index=0,
                    format_func=rc.gf_symbol_select_label,
                    key=f'recv_sym_{i}',
                    disabled=_dolgozat_lock,
                )
                recv_sym_list.append(int(rs))
with st.expander('GF(8) elemek: karakter ↔ 3 bit ↔ polinom ↔ α hatvány ↔ int (0-7)', expanded=False):
    st.caption('A szimbólumok és a mezőelemek megfeleltetése (primitív elem **α**, irreducibilis polinom **x³+x+1**). **Sorrend:** **A → H** betűrend (a **karakter** oszlop szerint); az **α hatvány** oszlop így nem növekvő kitevő szerinti. A **G** és **H** együtthatói ugyanazokkal a 0–7 értékekkel számolhatók. Utolsó oszlop: a sorhoz tartozó **GF(8) egész** (mind a nyolc érték **0…7** pontosan egyszer).')
    _gf8_df = pd.DataFrame(rc.gf8_element_table_rows())
    _gf8_int_col = 'GF(8) int (0–7)'
    _gf8_df[_gf8_int_col] = _gf8_df[_gf8_int_col].astype(str)
    st.dataframe(_gf8_df, use_container_width=True, hide_index=True)
with st.expander('GF(2³) szorzás és összeadás (8×8)', expanded=False):
    st.caption('**Sor / oszlopfejléc:** **a** és **b** (int **0…7**), zárójelben az **α-hatvány** alak (ugyanaz, mint a fenti GF(8) táblázatnál). **Összeadás:** char 2 → a bitek **XOR**-ja (ugyanaz, mint **a ⊕ b** az int reprezentáción). **Szorzás:** mezőbeli **a·b**. Irreducibilis polinom: **x³+x+1** (**galois** GF(2³)).')
    st.markdown(ex.gf8_arithmetic_tables_html(), unsafe_allow_html=True)
G, _H_sys_discarded = rc.permute_columns_parity_order(rc.G_BASE, rc.H_BASE, parity_right)
H = rc.H_RS74_PARITY_EVAL
m = rc.GF(m_vals).reshape(1, rc.K)
c = m @ G
# Új GF(8) vektor minden futáskor (int listából) — elkerüljük a nézet/másolás miatti „beragadt” r / szindróma megjelenítést.
c_ints_list = rc.gf_row_to_ints(c)
r_ints_list = list(c_ints_list)
if corrupt:
    for hi in range(num_errors):
        pos = err_pos_list[hi]
        if inj_mode.startswith('Összeadás'):
            r_ints_list[pos] = int(rc.GF(r_ints_list[pos]) + rc.GF(err_mag_list[hi])) & 7
        else:
            r_ints_list[pos] = int(recv_sym_list[hi]) & 7
r = rc.GF(r_ints_list).reshape(1, rc.N)
e = r - c
s_row_live = rc.syndrome_row(r, H)
s_ints_live = rc.gf_row_to_ints(s_row_live)
# Ne adj meg `key`-t a füleknek `on_change="ignore"` mellett: a 1.56+ verziókban a stabil
# block_id + nem állapotkövető tab összeegyeztetése miatt a nem aktív fülek tartalma elavulhat.
with st.sidebar:
    st.divider()
    st.markdown('**Élő számítás** (sidebar → **r**, **s**):')
    st.caption('**r** int: `' + rc.format_int_row(r_ints_list) + '`')
    st.caption('**s = r·Hᵀ** int: `' + rc.format_int_row(s_ints_live) + '`')
tab_g, tab_enc, tab_err, tab_syn, tab_syn0, tab_dec = st.tabs(
    ['Alapadatok', 'Kódolás', 'Fogadott szó és hiba', 'Szindroma polinom', 'Szindroma H', 'Javítás / dekódolás'],
)
with tab_g:
    st.subheader('Generátorpolinom g(x)')
    st.latex('g(x)=(x-\\alpha)(x-\\alpha^2)(x-\\alpha^3)')
    st.latex('g(x)=x^3+(\\alpha^2+1)x^2+\\alpha x+(\\alpha^2+1)')
    st.subheader('Generátor mátrix G (4 × 7)')
    if parity_right:
        st.caption('**Paritás jobbra:** G = [I₄ | P] (első 4 oszlop egységmátrix). c = m · G → [m₀…m₃ | p₀…p₂].')
    else:
        st.caption('Bal oldalt 3 paritásoszlop, jobb oldalt 4×4 egységmátrix. A sorokat úgy kapod, hogy veszed 1, x, x², x³ üzenetbázist, majd mindegyiket megszorzod x³-mal ($x^{n-k}$, itt n−k=3), és g(x)-re osztva a maradék adja a paritásrészt. Vagyis: x³, x⁴, x⁵, x⁶ maradékait kell kiszámolni g(x)-re modulo.')
    st.dataframe(np.array(G, dtype=int), use_container_width=True)
    st.latex('G = \\begin{bmatrix} ' + ex.format_gf_matrix(G) + ' \\end{bmatrix}')
    st.subheader('Paritás-mátrix H (3 × 7)')
    st.caption(
        '**Kiértékelési (Vandermonde) alak:** a **j**-edik oszlop elemei **1, αʲ, α²ʲ, α³ʲ** (mod **α⁷ = 1**), ahol **j = 0…6** az **aktuális G** oszlopindexeivel egyezik. '
        'Ezzel **s = r·Hᵀ** komponensei megegyeznek **y(α), y(α²), y(α³)** értékekkel (**y(x) = Σᵢ rᵢ xⁱ**). A szisztematikus **[I|P]** / **[Pᵀ|I]** alak helyett itt ezt a fix **H**-t használjuk a szindróma- és javításfülekben is.'
    )
    st.dataframe(np.array(H, dtype=int), use_container_width=True)
    st.latex('H = \\begin{bmatrix} ' + ex.format_gf_matrix(H) + ' \\end{bmatrix}')
    st.markdown('**Hatványalakban:**')
    st.latex(
        r'H = \begin{bmatrix}'
        r'1 & \alpha & \alpha^2 & \alpha^3 & \alpha^4 & \alpha^5 & \alpha^6 \\'
        r'1 & \alpha^2 & \alpha^4 & \alpha^6 & \alpha & \alpha^3 & \alpha^5 \\'
        r'1 & \alpha^3 & \alpha^6 & \alpha^2 & \alpha^5 & \alpha & \alpha^4'
        r'\end{bmatrix}'
    )
    _call_render_g_parity_mod_g_long_division_expander(parity_right=parity_right)
    _render_g_ht_zero_derivation(G, H)
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
    with st.expander('Az együtthatók kiszámolása a GF(2³) szorzás és összeadás (8×8) táblázat alapján (aktuális üzenet) (c₀…c₆)', expanded=False):
        for j in range(rc.N):
            st.markdown(ex.markdown_line_cj_from_m_dot_g(m_vals, G, j))
        if parity_right:
            st.info('**Paritás jobbra:** **c = m·G** = **[m₀,m₁,m₂,m₃ | r₀,r₁,r₂]** — a polinom **c₀…c₆** sorrendje ettől eltér (először a maradék alacsony fokú tagjai). Kézi ellenőrzéshez használd a polinom **c₀…c₆** együtthatóit (növekvő fok szerint).')
with tab_err:
    st.subheader('Fogadott vektor és hibavektor')
    c_ints_err = rc.gf_row_to_ints(c)
    r_ints_err = r_ints_list
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
    _render_dynamic_syndrome_poly_eval_tab(r_ints_list)
with tab_syn0:
    st.subheader('Szindróma számítás')
    if corrupt and num_errors >= 2:
        st.caption('Több szimbólumhiba esetén a szindróma általában **nem** írható le egyetlen [pozíció, hiba] párral; az „egy-hibás” illesztés nem megbízható.')
    s = s_row_live
    s_ints = s_ints_live
    r_ints_syn = r_ints_list
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
    st.markdown('**s** GF(8) int (0–7): `' + rc.format_int_row(s_ints) + '`  \n**α hatvány alak** (s₀, s₁, s₂): ' + ', '.join((rc.INT_TO_ALPHA_POWER_STR[v] for v in s_ints)) + '  \n**Polinom alak** (ugyanazok az elemek): ' + ', '.join((rc.INT_TO_ALPHA_STR[v] for v in s_ints)) + '  \n**Megjegyzés:** **s = r·Hᵀ** ugyanazzal a **H**-val, mint az **Alapadatok** fülön (kiértékelési / Vandermonde alak); megegyezik a **Szindroma polinom** fül **y(α), y(α²), y(α³)** számításával.')
    ex.render_syndrome_r_dot_Ht_expander(r, H)
    j_hat, _a_hat = rc.single_error_from_syndrome(s.flatten(), H)
    if np.all(s == 0):
        st.caption('**s = 0:** nincs oszlophoz illesztendő nemtriviális szindróma.')
        st.success('Nulla szindróma, nincs hiba.')
    elif j_hat is None:
        st.warning('A szindrómához nem található "j" érték.')
with tab_dec:
    st.subheader('Egy hiba javítása, c visszaállítása.')
    s_dec = s_row_live
    s_dec_ints = s_ints_live
    r_dec_ints = r_ints_list
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
        _render_dec_tab_single_error_position_derivation(
            s_ints=s_dec_ints, j_hat=int(j_hat), r_ints=r_dec_ints, eps_int=a_hat_int
        )
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
