from __future__ import annotations

import sys
from pathlib import Path

import galois
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nav_visibility
import rs84_core_gf9 as rs84_core_gf9

# GF(9) = GF(3^2), irreducibilis polinom: x^2 + 1
F = galois.GF(3**2, irreducible_poly="x^2 + 1")
ALPHA = F.primitive_element

N = 8
K = 4
LETTER_ORDER = tuple("ABCDEFGHI")

# Harmadfoku, tetszolegesen valasztott generatorpolinom (keres szerint)
G_POLY = galois.Poly([F(1), ALPHA, ALPHA**2, ALPHA**3], field=F, order="asc")


def _alpha_power_label(v: int) -> str:
    supers = "⁰¹²³⁴⁵⁶⁷⁸⁹"

    def sup_int(n: int) -> str:
        if n == 0:
            return supers[0]
        out = ""
        x = n
        while x > 0:
            out = supers[x % 10] + out
            x //= 10
        return out

    x = int(v)
    if x == 0:
        return "0"
    xv = F(x)
    for k in range(8):
        if ALPHA**k == xv:
            return "α" + sup_int(k)
    return str(x)


def _gf9_label(v: int) -> str:
    x = int(v)
    return f"{x} ({_alpha_power_label(x)})"


def _poly_label_for_int(v: int) -> str:
    x = int(v)
    if x == 0:
        return "0"
    p = galois.Poly(F(x).vector(), field=galois.GF(3))
    t = str(p).replace("x", "α")
    supers = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    out: list[str] = []
    i = 0
    while i < len(t):
        if t[i] == "^":
            i += 1
            num: list[str] = []
            if i < len(t) and t[i] == "-":
                num.append("-")
                i += 1
            while i < len(t) and t[i].isdigit():
                num.append(t[i])
                i += 1
            out.append("".join(num).translate(supers))
            continue
        out.append(t[i])
        i += 1
    return "".join(out)


def _poly_x_label_for_int(v: int) -> str:
    x = int(v)
    if x == 0:
        return "0"
    p = galois.Poly(F(x).vector(), field=galois.GF(3))
    return str(p).replace(" ", "")


def _symbol_select_label(v: int) -> str:
    ch = LETTER_ORDER[int(v)]
    return f"{ch} = {v} ({_alpha_power_label(v)})"


def _build_g_vandermonde() -> galois.FieldArray:
    # K x N: G[i,j] = α^(i*j), i=0..K-1, j=0..N-1
    g_mat = F.Zeros((K, N))
    for i in range(K):
        for j in range(N):
            g_mat[i, j] = ALPHA ** (i * j)
    return g_mat


def _build_h_eval_transpose() -> galois.FieldArray:
    # N x (N-K): H^T[v,j] = α^(v*(j+1))
    # igy G @ H^T = 0 a Vandermonde-felu G mellett
    # es s_j = sum_v r_v * α^(v*(j+1)) = r(α^(j+1))
    h_t = F.Zeros((N, N - K))
    for v in range(N):
        for j in range(N - K):
            h_t[v, j] = ALPHA ** (v * (j + 1))
    return h_t


def _format_row(vec: galois.FieldArray) -> str:
    return "[" + ", ".join(str(int(x)) for x in np.asarray(vec).flatten()) + "]"


def _matrix_df_with_alpha_labels(mat: galois.FieldArray) -> pd.DataFrame:
    arr = np.array(mat, dtype=int)
    rows: list[list[str]] = []
    for row in arr:
        rows.append([f"{v}   ({_alpha_power_label(int(v))})" for v in row])
    return pd.DataFrame(rows, columns=[str(i) for i in range(arr.shape[1])])


_S_SUB_4 = ("₀", "₁", "₂", "₃")


def _streamlit_cols_m_g(left: int = 1, right: int = 5) -> tuple:
    for g in (None, "small"):
        try:
            return st.columns([left, right], gap=g)
        except TypeError:
            continue
    return st.columns([left, right])


def _format_gf_matrix_inner(mat: galois.FieldArray) -> str:
    arr = np.array(mat, dtype=int)
    rows = [" & ".join(str(int(arr[i, j])) for j in range(arr.shape[1])) for i in range(arr.shape[0])]
    return " \\\\ ".join(rows)


def _gf9_syndrome_si_derivation_lines(
    r_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int
) -> tuple[list[str], int]:
    acc = F(0)
    lines: list[str] = []
    r_flat = F(r_vec).reshape(1, -1)
    for j in range(N):
        rj = int(r_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = F(rj) * F(hij)
        acc = acc + term
        lines.append(f"$r_{{{j}}}\\cdot H_{{{row_i},{j}}} = {rj}\\cdot {hij} = {int(term)}$")
    return (lines, int(acc))


def _gf9_syndrome_ci_derivation_lines(
    c_vec: galois.FieldArray, H_mat: galois.FieldArray, row_i: int
) -> tuple[list[str], int]:
    acc = F(0)
    lines: list[str] = []
    c_flat = F(c_vec).reshape(1, -1)
    for j in range(N):
        cj = int(c_flat[0, j])
        hij = int(H_mat[row_i, j])
        term = F(cj) * F(hij)
        acc = acc + term
        lines.append(f"$c_{{{j}}}\\cdot H_{{{row_i},{j}}} = {cj}\\cdot {hij} = {int(term)}$")
    return (lines, int(acc))


def _gf9_syndrome_sum_markdown(si_sum: int) -> str:
    v = int(si_sum)
    al = _poly_label_for_int(v)
    if al == str(v):
        return f"**{v}**"
    return f"**{v}**; ({al})"


def _render_syndrome_r_dot_Ht_expander_gf9(r_vec: galois.FieldArray, H_mat: galois.FieldArray) -> None:
    nk = int(H_mat.shape[0])
    with st.expander(
        "Szindróma: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$ — minden **$s_i$** ( $\\mathbf{r}$ és **$H$** soronkénti szorzata )",
        expanded=False,
    ):
        st.markdown("Lépésről lépésre (aktuális **r** és **H** sorai — **s₀**, **s₁**, **s₂**, **s₃**):")
        syn_step_cols = st.columns(nk)
        for i in range(nk):
            with syn_step_cols[i]:
                st.markdown(f"**s{_S_SUB_4[i]}** — **H** **{i}.** sora és **r**:")
                si_lines, si_sum = _gf9_syndrome_si_derivation_lines(r_vec, H_mat, i)
                for ln in si_lines:
                    st.markdown(ln)
                st.success(
                    f"**Összeg (s{_S_SUB_4[i]})** (GF(9), **⊕**): {_gf9_syndrome_sum_markdown(si_sum)}"
                )


def _render_s0_expander_gf9(c_vec: galois.FieldArray, H_mat: galois.FieldArray) -> None:
    nk = int(H_mat.shape[0])
    with st.expander(
        "Küldött szó, azaz hibátlan szindróma: **c·Hᵀ = 0** — **s₀**, **s₁**, **s₂**, **s₃** (aktuális **c** és **H** sorai)",
        expanded=False,
    ):
        st.markdown(
            "Lépésről lépésre (aktuális **c** és **H** sorai — **s₀**…**s₃**); kód szónál mind a négy összeg **0** (GF(9), **⊕**):"
        )
        s0_cols = st.columns(nk)
        for i in range(nk):
            with s0_cols[i]:
                st.markdown(f"**s{_S_SUB_4[i]}** — **H** **{i}.** sora és **c**:")
                ci_lines, si_sum = _gf9_syndrome_ci_derivation_lines(c_vec, H_mat, i)
                for ln in ci_lines:
                    st.markdown(ln)
                st.success(
                    f"**Összeg (s{_S_SUB_4[i]})** (GF(9), **⊕**): {_gf9_syndrome_sum_markdown(si_sum)}"
                )


def _render_v_at_alpha_powers_expander_gf9(r_row: galois.FieldArray, *, expander_label: str) -> None:
    """v(x)=Σ r_i x^i kiértékelése α^k-nál (k=1…4), tagonkénti GF(9) összegzés."""
    r_flat = [int(x) for x in np.asarray(r_row).flatten()][:N]
    with st.expander(expander_label, expanded=False):
        st.caption("Minden szorzás és összegzés a **GF(9)** testben.")
        cols = st.columns(4)
        for k in range(1, N - K + 1):
            ak = ALPHA**k
            terms: list[int] = []
            lines: list[str] = []
            acc = F(0)
            for i in range(N):
                ri = F(r_flat[i])
                pw = ak**i
                t = ri * pw
                acc = acc + t
                ti = int(t)
                terms.append(ti)
                lines.append(
                    f"$r_{{{i}}} \\cdot (\\alpha^{{{k}}})^{{{i}}} = {int(ri)} \\cdot {int(pw)} = {ti}$"
                )
            summary = (
                f"$v(\\alpha^{{{k}}}) = "
                + " + ".join(str(t) for t in terms)
                + f" = **{int(acc)}**$ (GF(9) összeg)"
            )
            parts = [f"**$v(\\alpha^{{{k}}})$**", *lines, summary]
            col_body = "\n\n".join(parts)
            with cols[k - 1]:
                st.markdown(
                    '<div style="font-size: clamp(0.58rem, min(2.6vw, 0.82rem), 1rem); line-height: 1.32;">\n\n'
                    + col_body
                    + "\n\n</div>",
                    unsafe_allow_html=True,
                )


def _render_cramer_mellek_gf9_columns(s1: int, s2: int, s3: int, s4: int) -> None:
    """GF(9): Cramer mellékszámítás — kulcslépések, nevező, számlálók, hányadosok, összegzés."""
    FS1, FS2, FS3, FS4 = F(s1), F(s2), F(s3), F(s4)
    nS3, nS4 = -FS3, -FS4
    in3, in4 = int(nS3), int(nS4)
    D = FS1 * FS3 - FS2 * FS2
    N2 = nS3 * FS3 - FS2 * nS4
    N1 = FS1 * nS4 - nS3 * FS2
    i_d, i_n2, i_n1 = int(D), int(N2), int(N1)
    if D == F(0):
        st.caption("A nevező determináns 0 — nincs értelmezhető Cramer-lépés.")
        return
    l2 = int(N2 / D)
    l1 = int(N1 / D)
    t13 = int(FS1 * FS3)
    t22 = int(FS2 * FS2)
    t_n2a = int(nS3 * FS3)
    t_n2b = int(FS2 * nS4)
    t_n1a = int(FS1 * nS4)
    t_n1b = int(nS3 * FS2)
    fs = "font-size: clamp(0.58rem, min(2.35vw, 0.88rem), 1rem); line-height: 1.38;"
    if l1 == 0 and l2 == 0:
        lx_tail = "0"
    elif l2 == 0:
        lx_tail = f"{l1}x"
    elif l1 == 0:
        lx_tail = f"{l2}x^2"
    else:
        lx_tail = f"{l1}x + {l2}x^2"
    body = (
        "Minden szorzás, összeg és hányados a **GF(9)** testben (decimális címkék 0…8).\n\n"
        "##### A kulcslépések\n\n"
        f"$-S_3$: $S_3 = {s3}$ → additív inverz **$-S_3 = {in3}$**.\n\n"
        f"$-S_4$: $S_4 = {s4}$ → additív inverz **$-S_4 = {in4}$**.\n\n"
        "##### Nevező $D$\n\n"
        f"$D = \\begin{{vmatrix}} {s1} & {s2} \\\\ {s2} & {s3} \\end{{vmatrix}}"
        f" = {s1}\\cdot{s3} - {s2}\\cdot{s2} = {t13} - {t22} = **{i_d}**$.\n\n"
        "##### $L_2$ számlálója\n\n"
        f"$\\begin{{vmatrix}} {in3} & {s2} \\\\ {in4} & {s3} \\end{{vmatrix}}"
        f" = {in3}\\cdot{s3} - {s2}\\cdot({in4}) = {t_n2a} - {t_n2b} = **{i_n2}**$.\n\n"
        f"$L_2 = \\dfrac{{{i_n2}}}{{{i_d}}} = **{l2}**$.\n\n"
        "##### $L_1$ számlálója\n\n"
        f"$\\begin{{vmatrix}} {s1} & {in3} \\\\ {s2} & {in4} \\end{{vmatrix}}"
        f" = {s1}\\cdot({in4}) - ({in3})\\cdot{s2} = {t_n1a} - {t_n1b} = **{i_n1}**$.\n\n"
        f"$L_1 = \\dfrac{{{i_n1}}}{{{i_d}}} = **{l1}**$.\n\n"
        "##### Tehát\n\n"
        f"$L_1 = {l1},\\quad L_2 = {l2}$.\n\n"
        "##### és\n\n"
        f"$L(x) = 1 + {lx_tail}$"
        + ("." if lx_tail != "0" else " (triviális eset).")
    )
    st.markdown(f'<div style="{fs}">\n\n{body}\n\n</div>', unsafe_allow_html=True)


def _render_cramer_mellek_gf9_m1(s1: int, s2: int) -> None:
    """GF(9): $m=1$ Cramer mellékszámítás — $L_1=-S_2/S_1$; $L_2$ nincs."""
    FS1, FS2 = F(s1), F(s2)
    nS2 = -FS2
    in2 = int(nS2)
    if FS1 == F(0):
        st.caption("$S_1 = 0$ — nincs értelmezhető Cramer-lépés.")
        return
    l1 = int(nS2 / FS1)
    fs = "font-size: clamp(0.58rem, min(2.35vw, 0.88rem), 1rem); line-height: 1.38;"
    body = (
        "Minden művelet a **GF(9)** testben (decimális címkék 0…8).\n\n"
        "##### A kulcslépés\n\n"
        f"$-S_2$: $S_2 = {s2}$ → additív inverz **$-S_2 = {in2}$**.\n\n"
        "##### Nevező (Cramer 1×1)\n\n"
        f"$D = |S_1| = S_1 = **{s1}**$.\n\n"
        "##### $L_1$ számlálója\n\n"
        f"$|-S_2| = -S_2 = **{in2}**$.\n\n"
        f"$L_1 = \\dfrac{{-S_2}}{{S_1}} = \\dfrac{{{in2}}}{{{s1}}} = **{l1}**$.\n\n"
        "##### Tehát\n\n"
        f"$L_1 = {l1}$. **$L_2$** nincs ($m=1$, egyetlen hiba).\n\n"
        "##### és\n\n"
        f"$L(x) = 1 + {l1}x$."
    )
    st.markdown(f'<div style="{fs}">\n\n{body}\n\n</div>', unsafe_allow_html=True)


def _st_latex_lx_poly_framed(l1: int, l2: int | None) -> None:
    """Konkrét L(x): ugyanaz a KaTeX / méret, mint a szimbolikus $L(x)$ sor; \\boxed{} = szűk keret."""
    if l2 is None:
        inner = rf"L(x) = 1 + {l1}\,x"
    else:
        inner = rf"L(x) = 1 + {l1}\,x + {l2}\,x^{{2}}"
    st.latex("\\boxed{" + inner + "}")


def _render_step4_locator_roots_formula_gf9(l1: int, l2: int) -> None:
    """4. pont alatt: $L(x)=0$ megoldóképlete GF(9)-ben; diszkriminánsban $(2\\cdot 2)$, nem a decimális $4$."""
    q = rs84_core_gf9.locator_quadratic_solution_gf9(l1, l2)
    if q["kind"] == "no_degree":
        st.caption("Az $L_2=L_1=0$ eset itt nem jelenik meg.")
        return
    if q["kind"] == "linear":
        st.markdown(
            f"A hibahelypolinom **elsőfokú**: $L(x)=1+{l1}\\,x$. Az $L(x)=0$ megoldása "
            f"$x=-\\dfrac{{1}}{{L_1}}=-\\dfrac{{1}}{{{l1}}}={q['z']}$ (GF(9) osztás). "
            f"A hibahely lokátor: $X_1={q['X']}$."
        )
        return
    if q["kind"] == "no_sqrt":
        st.warning(
            f"A diszkrimináns ($D={q['D']}$) nem négyzetszám a GF(9) testben — a megoldóképlet nem ad gyököt; "
            "ellenőrizd a szindrómákat / a lokátort."
        )
        return
    a, b, c = q["a"], q["b"], q["c"]
    st.markdown(
        fr"A $L(x)=0$ egyenlet **$ax^2+bx+c=0$** alakban: $a={a}$, $b={b}$, $c=1$. "
        r"A megoldóképlet $(2a)^2=(2\cdot 2)\,a^2$ alakjában a diszkriminánsban $(2\cdot 2)$ testbeli szorzást kell használni "
        r"($2\cdot 2=1\neq$ a decimális $4$ címke GF(9)-ben); tehát $D=b^2-(2\cdot 2)\,ac$, és nem $b^2-4ac$."
    )
    st.latex(r"x_{1,2} = \frac{-b \pm \sqrt{b^2 - (2\cdot 2)\,ac}}{2a}")
    fb, fa, fc = F(b), F(a), F(c)
    _two2 = F(2) * F(2)
    _b2 = int(fb * fb)
    _acf = int(_two2 * fa * fc)
    _neg_acf = int(-(_two2 * fa * fc))
    st.latex(
        rf"D = b^2 - (2\cdot 2)\,ac = {b}^2 - (2\cdot 2)\cdot {a}\cdot {c} = {_b2} - {_acf} "
        rf"= {_b2} + \bigl(-({_acf})\bigr) = {_b2} + {_neg_acf} = {q['D']}."
    )
    st.latex(rf"\sqrt{{D}} = \sqrt{{{q['D']}}} = {q['sqrt_D']}.")
    _s = int(q["sqrt_D"])
    _neg_b = int(-fb)
    _neg_s = int(-F(_s))
    _np, _nm = int(q["neg_b_plus_sqrt"]), int(q["neg_b_minus_sqrt"])
    _ta = int(q["two_a"])
    st.latex(rf"2a = 2\cdot a = 2\cdot {a} = {_ta}.")
    if q["z1"] == q["z2"]:
        st.latex(
            rf"x_1 = x_2 = \dfrac{{-b + \sqrt{{D}}}}{{2a}} = \dfrac{{-b + {_s}}}{{2a}} = \dfrac{{-({b}) + {_s}}}{{2a}} "
            rf"= \dfrac{{{_neg_b} + {_s}}}{{2a}} = \dfrac{{{_np}}}{{{_ta}}} = {q['z1']}."
        )
    else:
        st.latex(
            rf"x_1 = \dfrac{{-b + \sqrt{{D}}}}{{2a}} = \dfrac{{-b + {_s}}}{{2a}} = \dfrac{{-({b}) + {_s}}}{{2a}} "
            rf"= \dfrac{{{_neg_b} + {_s}}}{{2a}} = \dfrac{{{_np}}}{{{_ta}}} = {q['z1']}."
        )
        st.latex(
            rf"x_2 = \dfrac{{-b - \sqrt{{D}}}}{{2a}} = \dfrac{{-b - {_s}}}{{2a}} = \dfrac{{-({b}) - {_s}}}{{2a}} "
            rf"= \dfrac{{-({b}) + (-({_s}))}}{{2a}} = \dfrac{{{_neg_b} + ({_neg_s})}}{{2a}} = \dfrac{{{_nm}}}{{{_ta}}} = {q['z2']}."
        )
    st.markdown(
        "A hibahely lokátorok a gyökök **multiplikatív inverzei** ($X_i=z_i^{-1}$) a GF(9) testben:"
    )
    if q["z1"] == q["z2"]:
        st.latex(rf"X_1 = X_2 = z_1^{{-1}} = {q['z1']}^{{-1}} = {q['X1']}.")
    else:
        st.latex(
            rf"X_1 = z_1^{{-1}} = {q['z1']}^{{-1}} = {q['X1']},\quad "
            rf"X_2 = z_2^{{-1}} = {q['z2']}^{{-1}} = {q['X2']}."
        )


def _gf9_discrete_log_alpha(xi: int) -> int | None:
    """$\\alpha^d$ ($0\\le d<8$) decimális címkéje megegyezik-e $X$-szel; visszaad $d$-t vagy ``None``."""
    xv = F(int(xi) % 9)
    for d in range(8):
        if int(ALPHA**d) == int(xv):
            return d
    return None


def _render_step5_hibahelyek_gf9(l1: int, l2_pad: int, m_step: int) -> None:
    """5. pont: $X_k=\\alpha^{i_k}$ — a kitevő $i_k$ (diszkrét log $\alpha$ alapon, $0\\le d<8$)."""
    q = rs84_core_gf9.locator_quadratic_solution_gf9(l1, int(l2_pad))
    if m_step == 1:
        if q.get("kind") != "linear" or q.get("X") is None:
            return
        x1 = int(q["X"])
        st.markdown(
            "**5) Az** $X_1$ **hibahely lokátorból határozzuk meg az** $i_1$ **hibahelyet az** "
            r"$X_1 = \alpha^{i_1}$ **összefüggés felhasználásával.**"
        )
        d1 = _gf9_discrete_log_alpha(x1)
        if d1 is None:
            st.warning(
                f"$X_1={x1}$ nem egyezik $\\alpha^d$-vel egyetlen $d\\in\\{{0,\\ldots,7\\}}$ mellett sem (decimális címkék)."
            )
            return
        st.latex(
            rf"X_1 = {x1} = \alpha^{{{d1}}} \;\Rightarrow\; "
            + r"\boxed{\boxed{i_1 = " + str(d1) + "}}."
        )
        return
    if q.get("kind") != "quadratic" or q.get("X1") is None or q.get("X2") is None:
        return
    x1, x2 = int(q["X1"]), int(q["X2"])
    st.markdown(
        "**5) Az** $X_1, X_2$ **hibahely lokátorokból határozzuk meg az** $i_1, i_2$ **hibahelyeket az** "
        r"$X_1 = \alpha^{i_1},\quad X_2 = \alpha^{i_2}$ **összefüggések felhasználásával.**"
    )
    st.markdown(
        r"(A lokátor $0, 1, 2, 3, \ldots$ értékeket vehet fel, tehát a 0-ás érték valójában az 1. pozíció!)"
    )
    d1, d2 = _gf9_discrete_log_alpha(x1), _gf9_discrete_log_alpha(x2)
    if d1 is None or d2 is None:
        st.warning(
            f"Nem sikerült mindkét lokátort ($X_1={x1}$, $X_2={x2}$) $\\alpha^d$ alakban felírni $d\\in\\{{0,\\ldots,7\\}}$ mellett."
        )
        return
    st.latex(
        rf"X_1 = {x1} = \alpha^{{{d1}}} \;\Rightarrow\; "
        + r"\boxed{\boxed{i_1 = " + str(d1) + "}}"
        + rf",\qquad X_2 = {x2} = \alpha^{{{d2}}} \;\Rightarrow\; "
        + r"\boxed{\boxed{i_2 = " + str(d2) + "}}."
    )


def _gf9_y1_y2_vandermonde_cramer(x1: int, x2: int, s1: int, s2: int) -> tuple[int, int] | None:
    """$B\\bar{y}=\\bar{s}$ megoldása: $(Y_1,Y_2)$ GF(9)-ben; ``None``, ha a Vandermonde-nevező 0."""
    Fx1, Fx2 = F(int(x1) % 9), F(int(x2) % 9)
    Fx1sq, Fx2sq = Fx1 * Fx1, Fx2 * Fx2
    D = Fx1 * Fx2sq - Fx2 * Fx1sq
    if D == F(0):
        return None
    FS1, FS2 = F(int(s1) % 9), F(int(s2) % 9)
    N1 = FS1 * Fx2sq - FS2 * Fx2
    N2 = Fx1 * FS2 - Fx1sq * FS1
    return (int(N1 / D), int(N2 / D))


def _render_step6_By_s_cramer_gf9(
    l1: int, l2: int, s1: int, s2: int, r_ints: list[int], c_ints: list[int]
) -> None:
    """6. pont ($m=2$): $B\\bar{y}=\\bar{s}$ Vandermonde-rendszer $Y_1,Y_2$-re (Cramer)."""
    q = rs84_core_gf9.locator_quadratic_solution_gf9(l1, l2)
    if q.get("kind") != "quadratic" or q.get("X1") is None or q.get("X2") is None:
        return
    x1, x2 = int(q["X1"]), int(q["X2"])
    x1sq = int(F(x1) * F(x1))
    x2sq = int(F(x2) * F(x2))
    y_pair = _gf9_y1_y2_vandermonde_cramer(x1, x2, s1, s2)
    st.markdown(
        "**6) A szindrómák és az** $X_1, X_2$ **hibahely lokátorok ismeretében oldjuk meg a** "
        r"$B\bar{y} = \bar{s}$ **lineáris egyenletrendszert, ahol**"
    )
    col_sym, col_num, col_y_cramer = st.columns(3, gap="small")
    with col_sym:
        st.latex(r"B = \begin{pmatrix} X_1 & X_2 \\ X_1^2 & X_2^2 \end{pmatrix}")
        st.latex(r"\bar{s} = \begin{pmatrix} S_1 \\ S_2 \end{pmatrix}")
        st.latex(r"\bar{y} = \begin{pmatrix} Y_1 \\ Y_2 \end{pmatrix}")
    with col_num:
        st.latex(
            rf"B = \begin{{pmatrix}} {x1} & {x2} \\ {x1sq} & {x2sq} \end{{pmatrix}}"
        )
        st.latex(
            r"\bar{s} = \begin{pmatrix} " + f"{s1} \\\\ {s2}" + r"\end{pmatrix}"
        )
        st.markdown("**Hibaértékek**")
        if y_pair is not None:
            y1, y2 = y_pair
            st.latex(
                r"\boxed{\boxed{Y_1 = " + str(y1) + "}}"
                + r",\quad \boxed{\boxed{Y_2 = " + str(y2) + "}}."
            )
        else:
            st.caption("A Vandermonde-nevező 0, ezért $Y_1$, $Y_2$ nem számolhatók.")
    with col_y_cramer:
        st.markdown("Az egyenletrendszer megoldása szintén a Cramer-szabály alkalmazásával:")
        st.latex(
            r"Y_1 = \frac{\begin{vmatrix} S_1 & X_2 \\ S_2 & X_2^2 \end{vmatrix}}"
            r"{\begin{vmatrix} X_1 & X_2 \\ X_1^2 & X_2^2 \end{vmatrix}}"
        )
        st.latex(
            r"Y_2 = \frac{\begin{vmatrix} X_1 & S_1 \\ X_1^2 & S_2 \end{vmatrix}}"
            r"{\begin{vmatrix} X_1 & X_2 \\ X_1^2 & X_2^2 \end{vmatrix}}."
        )
    with st.expander("Mellékszámítás", expanded=False):
        _render_step6_mellekszamitas_gf9(s1, s2, x1, x2, x1sq, x2sq)
    if y_pair is not None:
        y1, y2 = y_pair
        _render_step7_e_v_c_gf9(r_ints, c_ints, y1, y2, x1, x2)


def _render_step7_mellekszamitas_r_minus_e_gf9(
    r_use: list[int], e_list: list[int], c_hat: list[int]
) -> None:
    """$\\overline{c}=\\overline{r}-\\overline{e}$: kivonás felírása, $-\\overline{e}$, majd $\\overline{r}+(-\\overline{e})$ GF(9)-ben."""
    neg_e = [int(-F(e_list[j])) for j in range(N)]
    r_str = rs84_core_gf9.format_int_row(r_use)
    e_str = rs84_core_gf9.format_int_row(e_list)
    neg_str = rs84_core_gf9.format_int_row(neg_e)
    fs = "font-size: clamp(0.58rem, min(2.35vw, 0.88rem), 1.38); line-height: 1.38;"
    row_lines: list[str] = []
    for j in range(N):
        rj, ej, nej, cj = r_use[j], e_list[j], neg_e[j], c_hat[j]
        js = str(j)
        if ej == 0:
            row_lines.append(
                rf"$c_{{{js}}} = r_{{{js}}} + (-e_{{{js}}}) = {rj} + 0 = **{cj}**$."
            )
        else:
            row_lines.append(
                rf"$e_{{{js}}}={ej}$ additív inverze: **$-e_{{{js}}}={nej}$**. "
                rf"$c_{{{js}}} = r_{{{js}}} + (-e_{{{js}}}) = {rj} + ({nej}) = **{cj}**$."
            )
    body = (
        "Minden koordináta a **GF(9)** testben (decimális címkék 0…8). **Kivonás:** $a-b=a+(-b)$, "
        "ahol $-b$ a $b$ additív inverze.\n\n"
        r"##### 1) A kivonás kijelölése ($\overline{c} = \overline{r} - \overline{e}$)" + "\n\n"
        f"$\\overline{{r}} - \\overline{{e}} = {r_str} - {e_str}$.\n\n"
        r"##### 2) Additív inverz: $-\overline{e}$" + "\n\n"
        f"$-\\overline{{e}} = {neg_str}$ (koordinátánként $-e_j$ a GF(9)-ben).\n\n"
        r"##### 3) Összeadás: $\overline{r} + (-\overline{e})$" + "\n\n"
        + "\n\n".join(row_lines)
    )
    with st.expander("Mellékszámítás", expanded=False):
        st.markdown(f'<div style="{fs}">\n\n{body}\n\n</div>', unsafe_allow_html=True)


def _render_step7_e_v_c_gf9(
    r_ints: list[int],
    c_ints: list[int],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> None:
    """7. pont ($m=2$): $\\overline{e}$, $\\overline{c}$; $\\overline{c}=\\overline{r}-\\overline{e}$ ellenőrzése."""
    st.markdown(
        "**7) A meghatározott** $Y_1, Y_2$ **értékek és az** $i_1, i_2$ **hibahelyek ismeretében határozzuk meg az** "
        r"$\overline{e}$ **hibavektort és az átküldött kódszót**"
    )
    st.latex(r"\overline{c} = \overline{r} - \overline{e}")
    d1 = _gf9_discrete_log_alpha(x1)
    d2 = _gf9_discrete_log_alpha(x2)
    if d1 is None or d2 is None:
        st.warning(
            "A 7. pont vektorai nem számolhatók: $X_1$ vagy $X_2$ nem $\\alpha^d$ alakú $d\\in\\{0,\\ldots,7\\}$ mellett."
        )
        return
    # Az 5. pont $X_k=\alpha^{i_k}$ alapján kapott $i_k$ = a kódszó **0…7** indexe ($r[j]$, $c[j]$),
    # nem a lokátorpolinom foka ↔ index leképezés (`poly_degree_to_vector_index`).
    v1, v2 = int(d1) % N, int(d2) % N
    e_list = [0] * N
    e_list[v1] = int(y1) % 9
    e_list[v2] = int(y2) % 9
    r_use = [int(r_ints[i]) % 9 for i in range(N)]
    c_hat = [int(F(r_use[j]) - F(e_list[j])) for j in range(N)]
    c_ref = [int(c_ints[i]) % 9 for i in range(N)]
    ok = c_hat == c_ref
    r_row = rs84_core_gf9.format_int_row(r_use)
    e_row = rs84_core_gf9.format_int_row(e_list)
    c_row = rs84_core_gf9.format_int_row(c_hat)
    st.subheader("Fogadott szó és hibavektor")
    st.caption(
        f"$\\overline{{e}}$: $X_1=\\alpha^{{{d1}}}$, $X_2=\\alpha^{{{d2}}}$ ⇒ hibák a **$j={v1}$** és **$j={v2}$** indexen "
        f"(0…7, az 5. pont $i_1,i_2$ értékei; 1-től számolva **{v1 + 1}**. és **{v2 + 1}**. hely); "
        f"$Y_1={e_list[v1]}$, $Y_2={e_list[v2]}$ (GF(9))."
    )
    for label, vec in (
        (r"$\overline{r}$ (fogadott):", r_row),
        (r"$\overline{e}$ (X1, X2 hibahely; Y1, Y2):", e_row),
        (r"$\overline{c} = \overline{r} - \overline{e}$:", c_row),
    ):
        lc, vc = st.columns([1.2, 5.8], gap="small")
        with lc:
            st.markdown(label)
        with vc:
            st.markdown(
                f"<div style='font-family:Consolas, monospace; white-space:nowrap;'>{vec}</div>",
                unsafe_allow_html=True,
            )
    _render_step7_mellekszamitas_r_minus_e_gf9(r_use, e_list, c_hat)
    if ok:
        st.success(
            "Ellenőrzés: a $\\overline{c} = \\overline{r} - \\overline{e}$ vektor **megegyezik** a **Kódolás** fülön számolt küldött $\\overline{c}$ kódszóval."
        )
    else:
        st.error(
            "Ellenőrzés: a $\\overline{c} = \\overline{r} - \\overline{e}$ vektor **nem egyezik** a küldött kódszóval. "
            f"javítás: {rs84_core_gf9.format_int_row(c_hat)}; küldött: {rs84_core_gf9.format_int_row(c_ref)}."
        )


def _gf9_y1_from_S1_over_X1(s1: int, x1: int) -> int | None:
    """Egyhiba: $Y_1 = S_1 / X_1$ GF(9)-ben (Vandermonde 1×1); ``None``, ha $X_1=0$."""
    fx = F(int(x1) % 9)
    if fx == F(0):
        return None
    return int(F(int(s1) % 9) / fx)


def _render_step6_mellekszamitas_m1_y1(s1: int, x1: int) -> None:
    """$m=1$: $Y_1 = S_1 / X_1$ mellékszámítás GF(9)-ben."""
    y_opt = _gf9_y1_from_S1_over_X1(s1, x1)
    fs = "font-size: clamp(0.58rem, min(2.35vw, 0.88rem), 1rem); line-height: 1.38;"
    if y_opt is None:
        body = "$X_1 = 0$ — az $S_1/X_1$ hányados nem értelmezhető."
    else:
        body = (
            "Minden művelet a **GF(9)** testben (decimális címkék 0…8).\n\n"
            "##### $Y_1$\n\n"
            f"$Y_1 = \\dfrac{{S_1}}{{X_1}} = \\dfrac{{{s1}}}{{{x1}}} = **{y_opt}**$."
        )
    st.markdown(f'<div style="{fs}">\n\n{body}\n\n</div>', unsafe_allow_html=True)


def _render_step6_By_s_single_gf9(
    l1: int, s1: int, r_ints: list[int], c_ints: list[int]
) -> None:
    """6. pont ($m=1$): $B\\bar{y}=\\bar{s}$ egy egyenlet — $Y_1=S_1/X_1$."""
    q = rs84_core_gf9.locator_quadratic_solution_gf9(l1, 0)
    if q.get("kind") != "linear" or q.get("X") is None:
        return
    x1 = int(q["X"])
    y1_opt = _gf9_y1_from_S1_over_X1(s1, x1)
    st.markdown(
        "**6) A szindrómák és az** $X_1$ **hibahely lokátor ismeretében oldjuk meg a** "
        r"$B\bar{y} = \bar{s}$ **lineáris egyenletrendszert (egyetlen egyenlet), ahol**"
    )
    col_sym, col_num, col_y_cramer = st.columns(3, gap="small")
    with col_sym:
        st.latex(r"B = \begin{pmatrix} X_1 \end{pmatrix}")
        st.latex(r"\bar{s} = \begin{pmatrix} S_1 \end{pmatrix}")
        st.latex(r"\bar{y} = \begin{pmatrix} Y_1 \end{pmatrix}")
    with col_num:
        st.latex(rf"B = \begin{{pmatrix}} {x1} \end{{pmatrix}}")
        st.latex(rf"\bar{{s}} = \begin{{pmatrix}} {s1} \end{{pmatrix}}")
        st.markdown("**Hibaértékek**")
        if y1_opt is not None:
            st.latex(r"\boxed{\boxed{Y_1 = " + str(y1_opt) + "}}.")
        else:
            st.caption("$X_1 = 0$ — $Y_1$ nem számolható osztással.")
    with col_y_cramer:
        st.markdown("Az egyenletrendszer megoldása szintén a Cramer-szabály alkalmazásával:")
        st.latex(r"Y_1 = \frac{|S_1|}{|X_1|} = \frac{S_1}{X_1}")
    with st.expander("Mellékszámítás", expanded=False):
        _render_step6_mellekszamitas_m1_y1(s1, x1)
    if y1_opt is not None:
        _render_step7_e_v_c_gf9_m1(r_ints, c_ints, y1_opt, x1)


def _render_step7_e_v_c_gf9_m1(
    r_ints: list[int], c_ints: list[int], y1: int, x1: int
) -> None:
    """7. pont ($m=1$): $\\overline{e}$, $\\overline{c}$; $\\overline{c}=\\overline{r}-\\overline{e}$ ellenőrzése."""
    st.markdown(
        "**7) A meghatározott** $Y_1$ **érték és az** $i_1$ **hibahely ismeretében határozzuk meg az** "
        r"$\overline{e}$ **hibavektort és az átküldött kódszót**"
    )
    st.latex(r"\overline{c} = \overline{r} - \overline{e}")
    d1 = _gf9_discrete_log_alpha(x1)
    if d1 is None:
        st.warning(
            "A 7. pont vektorai nem számolhatók: $X_1$ nem $\\alpha^d$ alakú $d\\in\\{0,\\ldots,7\\}$ mellett."
        )
        return
    v1 = int(d1) % N
    e_list = [0] * N
    e_list[v1] = int(y1) % 9
    r_use = [int(r_ints[i]) % 9 for i in range(N)]
    c_hat = [int(F(r_use[j]) - F(e_list[j])) for j in range(N)]
    c_ref = [int(c_ints[i]) % 9 for i in range(N)]
    ok = c_hat == c_ref
    r_row = rs84_core_gf9.format_int_row(r_use)
    e_row = rs84_core_gf9.format_int_row(e_list)
    c_row = rs84_core_gf9.format_int_row(c_hat)
    st.subheader("Fogadott szó és hibavektor")
    st.caption(
        f"$\\overline{{e}}$: $X_1=\\alpha^{{{d1}}}$ ⇒ hiba a **$j={v1}$** indexen "
        f"(0…7, az 5. pont $i_1$ értéke; 1-től számolva a **{v1 + 1}**. hely); "
        f"$Y_1={e_list[v1]}$ (GF(9))."
    )
    for label, vec in (
        (r"$\overline{r}$ (fogadott):", r_row),
        (r"$\overline{e}$ (X1; Y1):", e_row),
        (r"$\overline{c} = \overline{r} - \overline{e}$:", c_row),
    ):
        lc, vc = st.columns([1.2, 5.8], gap="small")
        with lc:
            st.markdown(label)
        with vc:
            st.markdown(
                f"<div style='font-family:Consolas, monospace; white-space:nowrap;'>{vec}</div>",
                unsafe_allow_html=True,
            )
    _render_step7_mellekszamitas_r_minus_e_gf9(r_use, e_list, c_hat)
    if ok:
        st.success(
            "Ellenőrzés: a $\\overline{c} = \\overline{r} - \\overline{e}$ vektor **megegyezik** a **Kódolás** fülön számolt küldött $\\overline{c}$ kódszóval."
        )
    else:
        st.error(
            "Ellenőrzés: a $\\overline{c} = \\overline{r} - \\overline{e}$ vektor **nem egyezik** a küldött kódszóval. "
            f"javítás: {rs84_core_gf9.format_int_row(c_hat)}; küldött: {rs84_core_gf9.format_int_row(c_ref)}."
        )


def _render_step6_mellekszamitas_gf9(
    s1: int, s2: int, x1: int, x2: int, x1sq: int, x2sq: int
) -> None:
    """$Y_1$, $Y_2$: Cramer 2×2 determinánsok kifejtése és hányadosok GF(9)-ben."""
    FS1, FS2 = F(s1), F(s2)
    Fx1, Fx2 = F(x1), F(x2)
    Fx1sq, Fx2sq = Fx1 * Fx1, Fx2 * Fx2
    D = Fx1 * Fx2sq - Fx2 * Fx1sq
    N1 = FS1 * Fx2sq - FS2 * Fx2
    N2 = Fx1 * FS2 - Fx1sq * FS1
    i_d = int(D)
    if D == F(0):
        st.caption(
            "A nevező determináns (Vandermonde) **0** — a $Y_1$, $Y_2$ Cramer-hányadosok nem értelmezhetők."
        )
        return
    y1 = int(N1 / D)
    y2 = int(N2 / D)
    i_n1, i_n2 = int(N1), int(N2)
    t_d_a = int(Fx1 * Fx2sq)
    t_d_b = int(Fx2 * Fx1sq)
    t_n1_a = int(FS1 * Fx2sq)
    t_n1_b = int(FS2 * Fx2)
    t_n2_a = int(Fx1 * FS2)
    t_n2_b = int(Fx1sq * FS1)
    fs = "font-size: clamp(0.58rem, min(2.35vw, 0.88rem), 1rem); line-height: 1.38;"
    body = (
        "Minden szorzás, összeg és hányados a **GF(9)** testben (decimális címkék 0…8).\n\n"
        "##### Közös nevező $D$\n\n"
        f"$D = \\begin{{vmatrix}} {x1} & {x2} \\\\ {x1sq} & {x2sq} \\end{{vmatrix}}"
        f" = {x1}\\cdot{x2sq} - {x2}\\cdot{x1sq} = {t_d_a} - {t_d_b} = **{i_d}**$.\n\n"
        "##### $Y_1$ számlálója\n\n"
        f"$\\begin{{vmatrix}} {s1} & {x2} \\\\ {s2} & {x2sq} \\end{{vmatrix}}"
        f" = {s1}\\cdot{x2sq} - {s2}\\cdot{x2} = {t_n1_a} - {t_n1_b} = **{i_n1}**$.\n\n"
        f"$Y_1 = \\dfrac{{{i_n1}}}{{{i_d}}} = **{y1}**$.\n\n"
        "##### $Y_2$ számlálója\n\n"
        f"$\\begin{{vmatrix}} {x1} & {s1} \\\\ {x1sq} & {s2} \\end{{vmatrix}}"
        f" = {x1}\\cdot{s2} - {x1sq}\\cdot{s1} = {t_n2_a} - {t_n2_b} = **{i_n2}**$.\n\n"
        f"$Y_2 = \\dfrac{{{i_n2}}}{{{i_d}}} = **{y2}**$.\n\n"
        "##### Tehát\n\n"
        f"$Y_1 = {y1},\\quad Y_2 = {y2}$."
    )
    st.markdown(f'<div style="{fs}">\n\n{body}\n\n</div>', unsafe_allow_html=True)


def _gf9_element_rows() -> list[dict[str, str | int]]:
    rows = []
    for i, ch in enumerate(LETTER_ORDER):
        rows.append(
            {
                "Karakter": ch,
                "GF(9) decimális (0-8)": str(i),
                "Polinom alak": _poly_label_for_int(i),
                "α-hatvány alak": _alpha_power_label(i),
            }
        )
    return rows


def _gf9_poly_text(poly: galois.Poly, *, size: int = 3) -> str:
    coeffs = [int(c) for c in poly.coefficients(order="asc", size=size)]
    parts: list[str] = []
    supers = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    def _sup(n: int) -> str:
        if n == 0:
            return supers[0]
        out = ""
        x = n
        while x > 0:
            out = supers[x % 10] + out
            x //= 10
        return out
    for p in range(size - 1, -1, -1):
        c = coeffs[p]
        if c == 0:
            continue
        coef = _alpha_power_label(c)
        if p == 0:
            parts.append(coef)
        elif p == 1:
            parts.append("x" if c == 1 else f"{coef}x")
        else:
            parts.append(f"x{_sup(p)}" if c == 1 else f"{coef}x{_sup(p)}")
    return " + ".join(parts) if parts else "0"


def _poly_long_divide_records(dividend: galois.Poly, divisor: galois.Poly) -> tuple[list[dict[str, galois.Poly | int]], galois.Poly]:
    steps: list[dict[str, galois.Poly | int]] = []
    r = dividend
    zero = galois.Poly([F(0)], field=F, order="asc")
    ddeg = int(divisor.degree)
    dlead = divisor.coefficients(order="asc", size=ddeg + 1)[ddeg]
    while r != zero and int(r.degree) >= ddeg:
        rdeg = int(r.degree)
        shift = rdeg - ddeg
        rlead = r.coefficients(order="asc", size=rdeg + 1)[rdeg]
        qcoeff = rlead / dlead
        x_shift = galois.Poly([F(0)] * shift + [F(1)], field=F, order="asc")
        sub = qcoeff * x_shift * divisor
        r_next = r - sub
        steps.append(
            {
                "shift": int(shift),
                "qcoeff": int(qcoeff),
                "before": r,
                "sub": sub,
                "after": r_next,
            }
        )
        r = r_next
    return steps, r


G = _build_g_vandermonde()
H_T = _build_h_eval_transpose()
H = H_T.T

_GF9_INJ_KOZVETLEN = "Közvetlen: r[j] = érték"
_GF9_INJ_OSSZEADAS = "Összeadás: r[j] = c[j] + e"
_GF9_MINTA_EGYENI = "Egyéni beállítás"
_GF9_MINTA_1 = "1-es mintahiba"
_GF9_MINTA_2 = "2-es mintahiba"
_GF9_MINTA_3 = "3-as mintahiba"
_GF9_MINTA_4 = "4-es mintahiba"
_GF9_MINTA_5 = "5-ös mintahiba"
_GF9_MINTA_RND1 = "random 1 hiba"
_GF9_MINTA_RND2 = "random 2 hiba"
_GF9_MINTAHIBA_PRESETS: dict[str, tuple[tuple[int, int], ...]] = {
    _GF9_MINTA_1: ((1, 5), (4, 2)),
    _GF9_MINTA_2: ((0, 3), (7, 6)),
    _GF9_MINTA_3: ((2, 8), (5, 1)),
    _GF9_MINTA_4: ((3, 4), (6, 7)),
    _GF9_MINTA_5: ((1, 7), (6, 2)),
}
_GF9_MINTAHIBA_LETTERS: dict[str, tuple[str, str, str, str]] = {
    _GF9_MINTA_1: ("B", "C", "D", "E"),
    _GF9_MINTA_2: ("C", "D", "E", "F"),
    _GF9_MINTA_3: ("A", "B", "C", "D"),
    _GF9_MINTA_4: ("D", "E", "F", "G"),
    _GF9_MINTA_5: ("E", "F", "G", "H"),
}


def _gf9_set_letters(letters: tuple[str, str, str, str]) -> None:
    for i, ch in enumerate(letters):
        st.session_state[f"gf9_letter_{i}"] = ch


def _gf9_codeword_ints_from_letters() -> list[int]:
    m_vals_loc = [LETTER_ORDER.index(str(st.session_state.get(f"gf9_letter_{i}", "A"))) for i in range(K)]
    m_row = F(m_vals_loc).reshape(1, K)
    c_row = m_row @ G
    return [int(x) for x in np.asarray(c_row).flatten()]


def _gf9_random_error_pairs(*, n_err: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    c_ints = _gf9_codeword_ints_from_letters()
    pos_opts = list(range(N))
    positions = rng.choice(pos_opts, size=n_err, replace=False).tolist()
    pairs: list[tuple[int, int]] = []
    for p in positions:
        p = int(p)
        good = int(c_ints[p])
        bad = good
        while bad == good:
            bad = int(rng.integers(0, 9))
        pairs.append((p, bad))
    return tuple(pairs)


def _gf9_apply_minta_from_radio() -> None:
    sel = str(st.session_state.get("gf9_minta_preset", _GF9_MINTA_EGYENI))
    if sel == _GF9_MINTA_EGYENI:
        st.session_state.pop("gf9_minta_snapshot", None)
        st.session_state.pop("gf9_minta_letters_snapshot", None)
        return
    rng = np.random.default_rng()
    if sel == _GF9_MINTA_RND1:
        pairs = _gf9_random_error_pairs(n_err=1, rng=rng)
    elif sel == _GF9_MINTA_RND2:
        pairs = _gf9_random_error_pairs(n_err=2, rng=rng)
    else:
        pairs = _GF9_MINTAHIBA_PRESETS[sel]
    st.session_state["gf9_num_errors"] = len(pairs)
    st.session_state["gf9_inj_mode"] = _GF9_INJ_KOZVETLEN
    for i, (p, v) in enumerate(pairs):
        st.session_state[f"gf9_err_pos_{i}"] = int(p)
        st.session_state[f"gf9_recv_{i}"] = int(v)
    letters = _GF9_MINTAHIBA_LETTERS.get(sel)
    if letters:
        _gf9_set_letters(letters)
    st.session_state["gf9_minta_snapshot"] = pairs
    st.session_state["gf9_minta_letters_snapshot"] = tuple(
        str(st.session_state.get(f"gf9_letter_{i}", "A")) for i in range(K)
    )


def _gf9_minta_clear_if_drift() -> None:
    minta = str(st.session_state.get("gf9_minta_preset", _GF9_MINTA_EGYENI))
    if minta == _GF9_MINTA_EGYENI:
        return
    snap = st.session_state.get("gf9_minta_snapshot")
    if snap is None:
        st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
        st.session_state.pop("gf9_minta_letters_snapshot", None)
        return
    pairs = snap
    if int(st.session_state.get("gf9_num_errors", -1)) != len(pairs):
        st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
        st.session_state.pop("gf9_minta_snapshot", None)
        st.session_state.pop("gf9_minta_letters_snapshot", None)
        return
    if str(st.session_state.get("gf9_inj_mode", "")) != _GF9_INJ_KOZVETLEN:
        st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
        st.session_state.pop("gf9_minta_snapshot", None)
        st.session_state.pop("gf9_minta_letters_snapshot", None)
        return
    for i, (p_exp, v_exp) in enumerate(pairs):
        if int(st.session_state.get(f"gf9_err_pos_{i}", -999)) != int(p_exp):
            st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
            st.session_state.pop("gf9_minta_snapshot", None)
            st.session_state.pop("gf9_minta_letters_snapshot", None)
            return
        if int(st.session_state.get(f"gf9_recv_{i}", -999)) != int(v_exp):
            st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
            st.session_state.pop("gf9_minta_snapshot", None)
            st.session_state.pop("gf9_minta_letters_snapshot", None)
            return
    letters_snap = st.session_state.get("gf9_minta_letters_snapshot")
    if letters_snap is not None:
        cur = tuple(str(st.session_state.get(f"gf9_letter_{i}", "A")) for i in range(K))
        if cur != letters_snap:
            st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
            st.session_state.pop("gf9_minta_snapshot", None)
            st.session_state.pop("gf9_minta_letters_snapshot", None)
            return
    letters_exp = _GF9_MINTAHIBA_LETTERS.get(minta)
    if letters_exp:
        for li in range(4):
            if str(st.session_state.get(f"gf9_letter_{li}")) != letters_exp[li]:
                st.session_state["gf9_minta_preset"] = _GF9_MINTA_EGYENI
                st.session_state.pop("gf9_minta_snapshot", None)
                st.session_state.pop("gf9_minta_letters_snapshot", None)
                return


st.set_page_config(page_title="RS(8,4) - hibajavító kódolás (GF(9))", layout="wide")
st.title("RS(8,4) - hibajavító kódolás (GF(9))")

with st.sidebar:
    st.page_link("app.py", label="RS(7,4)")
    st.page_link("pages/2_rs84.py", label="RS(8,4) GF(9)")
    if nav_visibility.SHOW_RS84_GF16_PAGE_LINK:
        st.page_link("pages/2_rs84_gf16.py", label="RS(8,4) GF(16) - törlésre kerül")
    st.page_link("pages/3_dokumentacio.py", label="Használati útmutató")
    st.divider()

    st.header("Bemenetek")
    st.subheader("4 szimbólum (A-I)")
    chosen: list[str] = []
    cols = st.columns(K)
    for i in range(K):
        with cols[i]:
            chosen.append(
                st.selectbox(
                    f"#{i+1}",
                    LETTER_ORDER,
                    index=min(i + 1, len(LETTER_ORDER) - 1),
                    key=f"gf9_letter_{i}",
                    on_change=_gf9_minta_clear_if_drift,
                )
            )
    m_vals = [LETTER_ORDER.index(ch) for ch in chosen]

    st.subheader("Hibák injektálása (1-2 szimbólum)")
    corrupt = st.checkbox("Hiba beszúrása", value=False, key="gf9_corrupt")
    inj_mode = _GF9_INJ_KOZVETLEN
    num_errors = 1
    err_pos: list[int] = []
    err_mag: list[int] = []
    recv_vals: list[int] = []
    if corrupt:
        st.radio(
            "Mintahiba (kézi módosítás után: Egyéni beállítás)",
            (
                _GF9_MINTA_EGYENI,
                _GF9_MINTA_1,
                _GF9_MINTA_2,
                _GF9_MINTA_3,
                _GF9_MINTA_4,
                _GF9_MINTA_5,
                _GF9_MINTA_RND1,
                _GF9_MINTA_RND2,
            ),
            index=0,
            key="gf9_minta_preset",
            on_change=_gf9_apply_minta_from_radio,
        )
        inj_mode = st.radio(
            "Hiba beállítás módja (minden hibára azonos)",
            (_GF9_INJ_KOZVETLEN, _GF9_INJ_OSSZEADAS),
            key="gf9_inj_mode",
            on_change=_gf9_minta_clear_if_drift,
        )
        num_errors = int(
            st.selectbox(
                "Hibák száma",
                [1, 2],
                key="gf9_num_errors",
                on_change=_gf9_minta_clear_if_drift,
            )
        )
        for i in range(num_errors):
            st.markdown(f"**Hiba {i+1} / {num_errors}**")
            p = int(
                st.selectbox(
                    f"Pozíció j (0..{N-1})",
                    list(range(N)),
                    key=f"gf9_err_pos_{i}",
                    on_change=_gf9_minta_clear_if_drift,
                )
            )
            err_pos.append(p)
            if inj_mode.startswith("Összeadás"):
                e = int(
                    st.selectbox(
                        "Hiba e (1..8)",
                        list(range(1, 9)),
                        format_func=_symbol_select_label,
                        key=f"gf9_err_mag_{i}",
                        on_change=_gf9_minta_clear_if_drift,
                    )
                )
                err_mag.append(e)
            else:
                rv = int(
                    st.selectbox(
                        "Fogadott r[j]",
                        list(range(9)),
                        format_func=_symbol_select_label,
                        key=f"gf9_recv_{i}",
                        on_change=_gf9_minta_clear_if_drift,
                    )
                )
                recv_vals.append(rv)

with st.expander("Elméleti alapok", expanded=False):
    st.header("Reed-Solomon kód RS(8,4) a GF(3²) testben")
    st.write("A kód a következő véges testen van értelmezve:")
    st.latex(r"GF(3^2)=GF(3)[x]/(x^2+1)")
    st.write("Mivel az irreducibilis polinom foka 2, a test elemei az alábbi alakúak:")
    st.latex(r"a+bx,\quad a,b\in\{0,1,2\}")
    st.write("ahol a szorzás során teljesül:")
    st.latex(r"x^2=-1=2\ (\mathrm{mod}\ 3)")
    st.write("Ez összesen 9 különböző elemet ad.")

    st.subheader("Generátormátrix (Vandermonde-alak)")
    st.write("A Reed-Solomon kódot kiértékelési (evaluation) formában adjuk meg. A generátormátrix:")
    st.latex(r"G_{i,j}=\alpha^{i\cdot j},\quad i=0,\ldots,k-1,\ j=0,\ldots,n-1")
    st.write("ahol α a test egy primitív eleme.")
    st.write("RS(8,4) esetén:")
    st.markdown(f"- kódszó hossza: $n={N}$")
    st.markdown(f"- üzenethossz: $k={K}$")
    st.markdown(f"- paritás: $n-k={N-K}$")
    st.write("A generátormátrix mérete:")
    st.latex(r"G\in GF(9)^{4\times 8}")

    st.subheader("A kód jelentése")
    st.write("Az üzenetet egy legfeljebb harmadfokú polinomként értelmezzük:")
    st.latex(r"m(x)=m_0+m_1x+m_2x^2+m_3x^3")
    st.write("A kódszó a polinom kiértékelése a következő pontokban:")
    st.latex(r"1,\alpha,\alpha^2,\ldots,\alpha^7")
    st.write("azaz:")
    st.latex(r"c=(m(1),m(\alpha),m(\alpha^2),\ldots,m(\alpha^7))")
    st.write("Ez pontosan a Vandermonde-mátrixos szorzással áll elő:")
    st.latex(r"c=m\cdot G")

    st.subheader("Minimális távolság és hibajavítás")
    st.latex(r"d_{\min}=n-k+1=8-4+1=5")
    st.write("A kód MDS (Maximum Distance Separable), így:")
    st.latex(r"t=\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor=2")
    st.write("tehát legfeljebb 2 hiba javítható.")

with st.expander("GF(9) elemek: karakter <-> decimális (0-8) <-> α-hatvány", expanded=False):
    _df_gf9 = pd.DataFrame(_gf9_element_rows())
    _sty_gf9 = _df_gf9.style.set_properties(subset=["GF(9) decimális (0-8)"], **{"text-align": "left"})
    st.dataframe(_sty_gf9, use_container_width=True, hide_index=True)

with st.expander("GF(3²) szorzás és összeadás (9x9)", expanded=False):
    add_tbl = []
    mul_tbl = []
    for a in range(9):
        add_row = {"a\\b (decimális)": _gf9_label(a)}
        mul_row = {"a\\b (decimális)": _gf9_label(a)}
        for b in range(9):
            add_row[_gf9_label(b)] = _gf9_label(int(F(a) + F(b)))
            mul_row[_gf9_label(b)] = _gf9_label(int(F(a) * F(b)))
        add_tbl.append(add_row)
        mul_tbl.append(mul_row)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Összeadás a+b**")
        st.dataframe(pd.DataFrame(add_tbl), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Szorzás a*b**")
        st.dataframe(pd.DataFrame(mul_tbl), use_container_width=True, hide_index=True)

m = F(m_vals).reshape(1, K)
c = m @ G
r = c.copy()
if corrupt:
    r = r.copy()
    for i in range(num_errors):
        p = err_pos[i]
        if inj_mode.startswith("Összeadás"):
            r[0, p] = r[0, p] + F(err_mag[i])
        else:
            r[0, p] = F(recv_vals[i])

tab_g, tab_enc, tab_err, tab_syn, tab_dec = st.tabs(
    ["Alapadatok", "Kódolás", "Fogadott szó és hiba", "Szindróma", "Javítás / Dekódolás"]
)

with tab_g:
    st.subheader("Generátor mátrix G (Vandermonde forma)")
    with st.expander("G általános alak", expanded=False):
        st.latex(
            r"G=\begin{pmatrix}"
            r"1 & 1 & 1 & \cdots & 1\\ "
            r"1 & \alpha & \alpha^2 & \cdots & \alpha^{n-1}\\ "
            r"1 & \alpha^2 & \alpha^4 & \cdots & \alpha^{2(n-1)}\\ "
            r"\vdots & \vdots & \vdots & \ddots & \vdots\\ "
            r"1 & \alpha^{k-1} & \alpha^{2(k-1)} & \cdots & \alpha^{(k-1)(n-1)}"
            r"\end{pmatrix}"
        )
        st.latex(r"G_{i,j}=\alpha^{i\cdot j},\quad i=0,\ldots,k-1,\ j=0,\ldots,n-1")
    st.markdown("**G (4 × 8)**")
    st.dataframe(_matrix_df_with_alpha_labels(G), use_container_width=True, hide_index=True)

    st.subheader("Paritás-mátrix H (4 × 8)")
    with st.expander("H általános alak", expanded=False):
        st.latex(
            r"H_{j,v}=\alpha^{v(j+1)},\quad j=0,\ldots,n-k-1,\ v=0,\ldots,n-1"
        )
        st.latex(
            r"H=\begin{pmatrix}"
            r"1 & \alpha & \alpha^2 & \cdots & \alpha^{n-1}\\ "
            r"1 & \alpha^2 & \alpha^4 & \cdots & \alpha^{2(n-1)}\\ "
            r"\vdots & \ddots & \vdots & \vdots\\ "
            r"1 & \alpha^{n-k} & \alpha^{2(n-k)} & \cdots & \alpha^{(n-k)(n-1)}"
            r"\end{pmatrix}"
        )
    st.markdown("**H (4 × 8)**")
    st.dataframe(_matrix_df_with_alpha_labels(H), use_container_width=True, hide_index=True)
    st.subheader(r"Transzponált paritás-mátrix $H^t$")
    with st.expander("Hᵀ általános alak", expanded=False):
        st.latex(
            r"H^T=\begin{pmatrix}"
            r"1 & 1 & \cdots & 1\\ "
            r"\alpha & \alpha^2 & \cdots & \alpha^{n-k}\\ "
            r"\alpha^2 & \alpha^4 & \cdots & \alpha^{2(n-k)}\\ "
            r"\vdots & \vdots & \ddots & \vdots\\ "
            r"\alpha^{n-1} & \alpha^{2(n-1)} & \cdots & \alpha^{(n-1)(n-k)}"
            r"\end{pmatrix}"
        )
        st.latex(r"H^T_{v,j}=\alpha^{v(j+1)},\quad s_j=r(\alpha^{j+1})")
    st.markdown("**Hᵀ (8 × 4)**")
    st.dataframe(_matrix_df_with_alpha_labels(H_T), use_container_width=True, hide_index=True)
    with st.expander(
        "Mellékszámítások",
        expanded=False,
    ):
        st.subheader("Összeadás mellékszámolása")
        st.write("Általánosan:")
        st.latex(r"(a+bx)+(c+dx)=(a+c)+(b+d)x")
        st.write("ahol minden együttható mod 3 szerint számolódik.")
        st.write("Példa:")
        st.latex(r"(1+x)+(2+2x)=3+3x=0+0x=0")
        st.write("tehát:")
        st.latex(r"4+8=0")

        poly_cols = [_poly_x_label_for_int(i) for i in range(9)]

        st.markdown("**Összeadási mátrix polinom-alakban**")
        add_poly_rows = []
        for a in range(9):
            row = {"+": _poly_x_label_for_int(a)}
            for b in range(9):
                row[poly_cols[b]] = _poly_x_label_for_int(int(F(a) + F(b)))
            add_poly_rows.append(row)
        st.dataframe(pd.DataFrame(add_poly_rows), use_container_width=True, hide_index=True)

        st.markdown("**Ugyanez decimális kódokkal**")
        add_dec_rows = []
        for a in range(9):
            row = {"+": a}
            for b in range(9):
                row[str(b)] = int(F(a) + F(b))
            add_dec_rows.append(row)
        st.dataframe(pd.DataFrame(add_dec_rows), use_container_width=True, hide_index=True)

        st.subheader("Szorzás mellékszámolása")
        st.write("Általánosan:")
        st.latex(r"(a+bx)(c+dx)=ac+(ad+bc)x+bdx^2")
        st.write("mivel:")
        st.latex(r"x^2=2")
        st.write("ezért:")
        st.latex(r"(a+bx)(c+dx)=(ac+2bd)+(ad+bc)x")
        st.write("mod 3 szerint.")
        st.write("Példa:")
        st.latex(r"(1+x)(1+x)=1+2x+x^2=1+2x+2=3+2x=2x")
        st.write("tehát:")
        st.latex(r"4\cdot4=6")

        st.markdown("**Szorzási mátrix polinom-alakban**")
        mul_poly_rows = []
        for a in range(9):
            row = {"×": _poly_x_label_for_int(a)}
            for b in range(9):
                row[poly_cols[b]] = _poly_x_label_for_int(int(F(a) * F(b)))
            mul_poly_rows.append(row)
        st.dataframe(pd.DataFrame(mul_poly_rows), use_container_width=True, hide_index=True)

        st.markdown("**Szorzási mátrix decimális kódokkal**")
        mul_dec_rows = []
        for a in range(9):
            row = {"×": a}
            for b in range(9):
                row[str(b)] = int(F(a) * F(b))
            mul_dec_rows.append(row)
        st.dataframe(pd.DataFrame(mul_dec_rows), use_container_width=True, hide_index=True)

with tab_enc:
    st.subheader("Üzenet és kódolás")
    st.markdown("Választott szimbólumok: **" + ", ".join(chosen) + "** → **m₀…m₃** (A=0, …, I=8).")

    rows_enc = []
    for i, ch in enumerate(chosen):
        v = LETTER_ORDER.index(ch)
        rows_enc.append(
            {
                "Pozíció": str(i + 1),
                "Betű": ch,
                "GF(9) decimális (0-8)": str(v),
                "Polinom alak": _poly_label_for_int(v),
                "α-hatvány alak": _alpha_power_label(v),
            }
        )
    _enc_df = pd.DataFrame(rows_enc)
    _enc_sty = (
        _enc_df.style
        .set_properties(subset=_enc_df.columns, **{"text-align": "left"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("text-align", "left")]},
            ]
        )
    )
    st.dataframe(_enc_sty, use_container_width=True, hide_index=True)

    m_ints = [int(x) for x in np.asarray(m).flatten()]
    c_ints = [int(x) for x in np.asarray(c).flatten()]
    st.markdown("**m** (int, m₀…m₃): " + "[" + ", ".join(str(v) for v in m_ints) + "]")

    st.markdown("### Előállítás: $\\mathbf{c} = \\mathbf{m} \\cdot G_{RS}$")
    m_row_tex = " & ".join(str(v) for v in m_ints)
    g_tex_rows = []
    g_arr = np.array(G, dtype=int)
    for rr in g_arr:
        g_tex_rows.append(" & ".join(str(int(v)) for v in rr))
    g_tex_inner = r" \\ ".join(g_tex_rows)
    c_row_tex = " & ".join(str(v) for v in c_ints)
    top_l, top_r = st.columns([2, 3])
    with top_l:
        st.empty()
    with top_r:
        st.latex(r"\begin{bmatrix}" + g_tex_inner + r"\end{bmatrix}")

    bot_l, bot_r = st.columns([2, 3])
    with bot_l:
        st.latex(r"\begin{bmatrix}" + m_row_tex + r"\end{bmatrix}")
    with bot_r:
        st.latex(r"\begin{bmatrix}" + c_row_tex + r"\end{bmatrix}")

    st.subheader("Kódszó α alakban")
    st.write(", ".join(f"c{j}={_alpha_power_label(v)}" for j, v in enumerate(c_ints)))

    st.markdown(
        "**Értékelő polinom**: "
        "$R(x)=c_0x^7+c_1x^6+\\cdots+c_7=\\sum_{v=0}^{7}c_vx^{7-v}$."
    )

    with st.expander("Kódszó együtthatók mellékszámítása", expanded=False):
        st.markdown("#### Az együtthatók kiszámolása a GF(9) táblázat alapján (aktuális üzenet) (c₀…c₇)")
        for j in range(N):
            term_parts = [f"{m_ints[i]}·{int(G[i, j])}" for i in range(K)]
            prod_vals = [int(F(m_ints[i]) * G[i, j]) for i in range(K)]
            acc = F(0)
            for pv in prod_vals:
                acc += F(pv)
            st.markdown(
                f"$c_{j} = " + " + ".join(term_parts) + f" = " + " + ".join(str(pv) for pv in prod_vals) + f" = {int(acc)}$"
            )

with tab_err:
    st.subheader("Fogadott szó és hibavektor")
    e = r - c
    c_row = _format_row(c)
    r_row = _format_row(r)
    e_row = _format_row(e)
    for label, vec in [
        (r"$\overline{c}$ (küldött):", c_row),
        (r"$\overline{r}$ (fogadott):", r_row),
        (r"$\overline{e} = \overline{r} - \overline{c}$:", e_row),
    ]:
        lc, vc = st.columns([1.2, 5.8], gap="small")
        with lc:
            st.markdown(label)
        with vc:
            st.markdown(
                f"<div style='font-family:Consolas, monospace; white-space:nowrap;'>{vec}</div>",
                unsafe_allow_html=True,
            )

with tab_syn:
    st.subheader("Szindróma számítás")
    if corrupt and num_errors == 2:
        st.caption(
            "**Két hiba:** a **Szindróma** fülön az **s = r·Hᵀ** egyetlen **H**-oszlopra illesztése **nem** ad **j**-t. "
            "Az **S₁…S₄** (értékelési / Peterson-gyök, **Hankel-determináns**) szekció és a **Javítás** fül e szindrómák alapján mindkét hibát **megtalálhatjuk és javíthatjuk**."
        )
    s_syn = r @ H_T
    s_ints_syn = [int(x) for x in np.asarray(s_syn).flatten()]
    r_ints_syn = [int(x) for x in np.asarray(r).flatten()]
    c_ints_syn = [int(x) for x in np.asarray(c).flatten()]
    r_row_tex = " & ".join(str(v) for v in r_ints_syn)
    s_row_tex = " & ".join(str(v) for v in s_ints_syn)
    h_t_tex = _format_gf_matrix_inner(H_T)
    s_c_syn = c @ H_T
    s_c_ints_syn = [int(x) for x in np.asarray(s_c_syn).flatten()]
    c_row_tex = " & ".join(str(v) for v in c_ints_syn)
    s_c_row_tex = " & ".join(str(v) for v in s_c_ints_syn)
    st.markdown("### Helyes (küldött) kód szó: $\\mathbf{0} = \\mathbf{c} \\, H^{\\mathsf{T}}$")
    _syn_lr = (5, 6)
    c_top_l, c_top_r = _streamlit_cols_m_g(*_syn_lr)
    with c_top_l:
        st.empty()
    with c_top_r:
        st.latex(r"H^{\mathsf{T}} = \begin{bmatrix} " + h_t_tex + r"\end{bmatrix}")
    c_bot_l, c_bot_r = _streamlit_cols_m_g(*_syn_lr)
    with c_bot_l:
        st.latex(r"\mathbf{c} = \begin{bmatrix} " + c_row_tex + r"\end{bmatrix}")
    with c_bot_r:
        st.latex(r"\mathbf{c} \, H^{\mathsf{T}} = \begin{bmatrix} " + s_c_row_tex + r"\end{bmatrix}")
    _render_s0_expander_gf9(c, H)
    st.markdown("### Fogadott szó: $\\mathbf{s} = \\mathbf{r} \\, H^{\\mathsf{T}}$")
    syn_top_l, syn_top_r = _streamlit_cols_m_g(*_syn_lr)
    with syn_top_l:
        st.empty()
    with syn_top_r:
        st.latex(r"H^{\mathsf{T}} = \begin{bmatrix} " + h_t_tex + r"\end{bmatrix}")
    syn_bot_l, syn_bot_r = _streamlit_cols_m_g(*_syn_lr)
    with syn_bot_l:
        st.latex(r"\mathbf{r} = \begin{bmatrix} " + r_row_tex + r"\end{bmatrix}")
    with syn_bot_r:
        st.latex(r"\mathbf{s} = \mathbf{r} \, H^{\mathsf{T}} = \begin{bmatrix} " + s_row_tex + r"\end{bmatrix}")
    st.markdown(
        "**α hatvány alak** (s₀…s₃): **"
        + "[" + ", ".join(str(v) for v in s_ints_syn) + "]** "
        + ", ".join(_alpha_power_label(v) for v in s_ints_syn)
        + "  \n**Polinom alak:** "
        + ", ".join(_poly_label_for_int(v) for v in s_ints_syn)
    )
    _render_syndrome_r_dot_Ht_expander_gf9(r, H)

with tab_dec:
    s_dec_tab = [int(x) for x in np.asarray(r @ H_T).flatten()]
    st.markdown(
        r"**1) s értéke a Szindróma fül alapján ($s=\mathbf{r}\,H^{\mathsf{T}}$, decimális): "
        f"[{', '.join(str(x) for x in s_dec_tab)}]. "
        r"Nyilván, ha $S_1 = S_2 = S_3 = S_4 = 0$, akkor nem történt hiba.**"
    )
    _ai = int(ALPHA)
    st.markdown(
        f"**Primitív elem** az **Alapadatok** fül alapján: **$\\alpha^1 = {_ai}$**. "
        "A csatornán keresztül érkezett $\\overline{v}$ vett jel felhasználásával **is** kiszámolhatjuk az"
    )
    st.latex(
        r"S_1 = v(\alpha),\quad S_2 = v(\alpha^2),\quad S_3 = v(\alpha^3),\quad S_4 = v(\alpha^4)"
    )
    r_fd = [int(x) for x in np.asarray(r).flatten()][:N]
    v_p = galois.Poly([F(x) for x in r_fd], field=F, order="asc")
    sk_v = [int(v_p(ALPHA**k)) for k in range(1, N - K + 1)]
    _v_exp_label = (
        r"$v(\alpha^k)$ ($k=1,2,3,4$) értékek megegyeznek a fent írt, a **Szindróma** fül alatt számolt értékekkel."
        if sk_v == s_dec_tab
        else r"$v(\alpha^k)$ kiértékelés részletei ($v(x)=\sum_{i=0}^{7} r_i x^i$, $k=1,2,3,4$, aktuális $\mathbf{r}$)"
    )
    _render_v_at_alpha_powers_expander_gf9(r, expander_label=_v_exp_label)
    if sk_v != s_dec_tab:
        st.warning(
            f"Eltérés: $v(\\alpha^k)$ sorozat **{sk_v}**, szindróma-fül **{s_dec_tab}**."
        )

    st.divider()
    st.markdown(
        r"**2) Hibák számának meghatározása. Legyen $h=t$, ahol $t$ a kód hibajavító képessége. "
        r"Számítsuk ki az $A_h$ mátrix determinánsát!**"
    )
    st.latex(
        r"A_h = \begin{pmatrix} "
        r"S_1 & S_2 & \ldots & S_h \\ "
        r"S_2 & S_3 & \ldots & S_{h+1} \\ "
        r"\vdots & \vdots & & \vdots \\ "
        r"S_h & S_{h+1} & \ldots & S_{2h-1} "
        r"\end{pmatrix}"
    )
    st.markdown(
        "Ha $|A_h| = 0$, akkor csökkentsük $h$ értékét eggyel. "
        "Egészen addig folytassuk ezt az eljárást, ameddig az $|A_h|$ determináns értéke már nem lesz $0$. "
        "Ez az utolsó $h$ lesz a hibák száma, az $m$ értéke."
    )
    _t_rs = (N - K) // 2
    st.latex(f"t = {_t_rs}")
    if all(x == 0 for x in sk_v):
        st.info("Minden $S_1=S_2=S_3=S_4=0$ — az értékelő szindrómák szerint nincs hiba ($m=0$).")
    else:
        s1, s2, s3, s4 = (int(sk_v[0]), int(sk_v[1]), int(sk_v[2]), int(sk_v[3]))
        d2 = int(F(s1) * F(s3) - F(s2) * F(s2))
        st.latex(r"A_2 = \begin{pmatrix} " + f"{s1} & {s2} \\\\ {s2} & {s3}" + r"\end{pmatrix}")
        st.latex(f"|A_2| = {d2}")
        _s1s3 = int(F(s1) * F(s3))
        _s2sq = int(F(s2) * F(s2))
        _neg_s2sq = int(-(F(s2) * F(s2)))
        with st.expander("Mellékszámítások", expanded=False):
            _q = '"'
            st.markdown(
                "A szorzások és összegek a fenti **GF($3^2$)** szorzás- és összeadás **(9×9)** táblázat szerint "
                "értendők (decimális címkék 0…8).\n\n"
                r"$|A_2| = S_1 S_3 - S_2^2 = S_1 S_3 + \bigl(-(S_2^2)\bigr)$ "
                r"a GF($3^2$) testben: $a - b = a + (-b)$, ahol $-b$ a $b$ additív inverze."
                "\n\n"
                f"$S_1 S_3 = {s1}\\cdot {s3} = **{_s1s3}**$.\n\n"
                f"$S_2^2 = S_2\\cdot S_2 = {s2}\\cdot {s2} = {_s2sq}.\\quad "
                f"\\text{{A {_s2sq} {_q}-{_q} előjel miatti additív inverze: }}{_neg_s2sq}.$\n\n"
                f"$|A_2| = {_s1s3} + ({_neg_s2sq}) = **{d2}**$ (GF(9) összeadás)."
            )
        det_h2_zero = d2 == 0
        if det_h2_zero:
            st.markdown("Ha $|A_2| = 0$, csökkentsük $h$ értékét eggyel.")
            st.latex(r"A_1 = \begin{pmatrix} " + f"{s1}" + r"\end{pmatrix}")
            st.latex(f"|A_1| = {s1}")
            st.markdown(
                f"$|A_2| = 0$, ezért $h := 1$; **$|A_1| = {s1}$**. "
                f"Ez alapján **$m = 1$** (hibák száma)."
            )
        else:
            st.markdown(
                f"A determináns $|A_2| = {d2}$, azaz nem nulla, ezért **$m = 2$** (hibák száma)."
            )

    pgz = rs84_core_gf9.compute_pgz_hankel_state(r_fd)
    if not pgz.get("all_S_zero") and int(pgz.get("m", 0)) >= 1:
        st.markdown(
            r"**3) A szindrómák és az $m$ ismeretében oldjuk meg az $A\overline{x} = \overline{b}$ lineáris egyenletrendszert, ahol**"
        )
        st3 = pgz.get("step3")
        m_step = int(pgz.get("m", 0))
        s1, s2, s3, s4 = (int(s_dec_tab[0]), int(s_dec_tab[1]), int(s_dec_tab[2]), int(s_dec_tab[3]))
        _show_cramer_col = bool(st3 and not st3.get("error") and m_step in (1, 2))
        if _show_cramer_col:
            col_sym, col_num, col_cramer = st.columns(3, gap="small")
        else:
            col_sym, col_num = st.columns(2, gap="medium")
        with col_sym:
            st.latex(
                r"A = \begin{pmatrix} "
                r"S_1 & S_2 & \cdots & S_m \\ "
                r"S_2 & S_3 & \cdots & S_{m+1} \\ "
                r"\vdots & \vdots & & \vdots \\ "
                r"S_m & S_{m+1} & \cdots & S_{2m-1} "
                r"\end{pmatrix}"
            )
            st.latex(r"\overline{b} = \begin{pmatrix} -S_{m+1} \\ -S_{m+2} \\ \vdots \\ -S_{2m} \end{pmatrix}")
            st.latex(r"\overline{x} = \begin{pmatrix} L_m \\ L_{m-1} \\ \vdots \\ L_1 \end{pmatrix}")
            st.latex(r"L(x) = 1 + L_1 x + L_2 x^2 + \cdots + L_m x^m")
        with col_num:
            if not st3 or st3.get("error"):
                st.caption("A 3. lépés egyenletrendszere nem oldható (szinguláris).")
            else:
                A = st3["A"]
                x = st3["x"]
                Lasc = st3["L_asc"]
                if m_step == 1:
                    st.latex(f"A = \\begin{{pmatrix}} {A[0][0]} \\end{{pmatrix}}")
                    st.latex(
                        f"\\overline{{b}} = \\begin{{pmatrix}} -S_2 \\end{{pmatrix}} "
                        f"= \\begin{{pmatrix}} {int(-F(s2))} \\end{{pmatrix}}"
                    )
                    st.latex(
                        f"\\overline{{x}} = \\begin{{pmatrix}} {x[0]} \\end{{pmatrix}} "
                        f"= \\begin{{pmatrix}} L_1 \\end{{pmatrix}}"
                    )
                    _st_latex_lx_poly_framed(int(Lasc[1]), None)
                elif m_step == 2:
                    st.latex(
                        f"A = \\begin{{pmatrix}} {A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]} \\end{{pmatrix}}"
                    )
                    st.latex(
                        f"\\overline{{b}} = \\begin{{pmatrix}} -S_3 \\\\ -S_4 \\end{{pmatrix}} "
                        f"= \\begin{{pmatrix}} {int(-F(s3))} \\\\ {int(-F(s4))} \\end{{pmatrix}}"
                    )
                    st.latex(
                        f"\\overline{{x}} = \\begin{{pmatrix}} {x[0]} \\\\ {x[1]} \\end{{pmatrix}} "
                        f"= \\begin{{pmatrix}} L_2 \\\\ L_1 \\end{{pmatrix}}"
                    )
                    _st_latex_lx_poly_framed(int(Lasc[1]), int(Lasc[2]))
        if _show_cramer_col:
            with col_cramer:
                if m_step == 1:
                    st.markdown(
                        "A **hibahelypolinom együtthatói** ($m=1$): **$L_2$** nincs; "
                        "a **$S_1 L_1 = -S_2$** egyenlet megoldása Cramerrel (1×1), "
                        "az **1. pont** $S_1$, $S_2$ értékeivel:"
                    )
                    st.latex(r"L_1 = \frac{|-S_2|}{|S_1|} = \frac{-S_2}{S_1}")
                else:
                    st.markdown(
                        "A **hibahelypolinom együtthatói.** A lineáris egyenletrendszer megoldásai a Cramer-szabállyal:"
                    )
                    st.latex(
                        r"L_2 = \frac{\begin{vmatrix} -S_3 & S_2 \\ -S_4 & S_3 \end{vmatrix}}"
                        r"{\begin{vmatrix} S_1 & S_2 \\ S_2 & S_3 \end{vmatrix}}"
                    )
                    st.latex(
                        r"L_1 = \frac{\begin{vmatrix} S_1 & -S_3 \\ S_2 & -S_4 \end{vmatrix}}"
                        r"{\begin{vmatrix} S_1 & S_2 \\ S_2 & S_3 \end{vmatrix}}"
                    )
        if st3 and not st3.get("error") and m_step == 2:
            with st.expander("Mellékszámítások", expanded=False):
                _render_cramer_mellek_gf9_columns(s1, s2, s3, s4)
        elif st3 and not st3.get("error") and m_step == 1:
            with st.expander("Mellékszámítások", expanded=False):
                _render_cramer_mellek_gf9_m1(s1, s2)
        if st3 and not st3.get("error"):
            _Lasc = st3["L_asc"]
            if m_step == 2:
                _l1, _l2 = int(_Lasc[1]), int(_Lasc[2])
                st.markdown(
                    rf"**4) Határozzuk meg az $L(x) = 1 + {_l1}\,x + {_l2}\,x^2$ hibahelypolinom gyökeit. "
                    r"A kiszámolt gyökök inverzei lesznek az $X_1, X_2$ hibahely lokátorok.**"
                )
                _render_step4_locator_roots_formula_gf9(_l1, _l2)
                _render_step5_hibahelyek_gf9(_l1, _l2, 2)
                _c_dec = [int(x) for x in np.asarray(c).flatten()][:N]
                _render_step6_By_s_cramer_gf9(_l1, _l2, s1, s2, r_fd, _c_dec)
            elif m_step == 1:
                _l1 = int(_Lasc[1])
                st.markdown(
                    rf"**4) Határozzuk meg az $L(x) = 1 + {_l1}\,x$ hibahelypolinom gyökeit. "
                    r"A kiszámolt gyök inverze lesz az $X_1$ hibahely lokátor.**"
                )
                _render_step4_locator_roots_formula_gf9(_l1, 0)
                _render_step5_hibahelyek_gf9(_l1, 0, 1)
                _c_dec_m1 = [int(x) for x in np.asarray(c).flatten()][:N]
                _render_step6_By_s_single_gf9(_l1, s1, r_fd, _c_dec_m1)
