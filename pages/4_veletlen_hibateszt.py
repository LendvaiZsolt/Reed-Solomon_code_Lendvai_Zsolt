from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nav_visibility
import rs_monte_harness as mh

st.set_page_config(page_title="Véletlen RS hibateszt", layout="wide")
st.title("Véletlen RS hibateszt (RS(7,4) és RS(8,4) GF(9))")

with st.sidebar:
    st.page_link("app.py", label="RS(7,4)")
    st.page_link("pages/2_rs84.py", label="RS(8,4) GF(9)")
    if nav_visibility.SHOW_RS84_GF16_PAGE_LINK:
        st.page_link("pages/2_rs84_gf16.py", label="RS(8,4) GF(16) - törlésre kerül")
    st.page_link("pages/3_dokumentacio.py", label="Használati útmutató")
    st.page_link("pages/4_veletlen_hibateszt.py", label="Véletlen hibateszt")
    st.divider()

st.markdown(
    "Minden futáskor **véletlen** választás: **RS(7,4)** vagy **RS(8,4) GF(9)** ág, majd ugyanilyen "
    "logika, mint a bal oldali bemenetek: véletlen üzenet, paritás / injektálási mód (ahol értelmes), "
    "0–2 hiba (RS(7,4)-nél legfeljebb 1 szándékos hiba). "
    "A **TEST_TRUE** azt jelenti, hogy a javítás után visszakapott **m₀…m₃** (információs szimbólumok) "
    "megegyezik az eredeti küldött üzenettel."
)
with st.expander("Miért lehet a kötegelt teszt lassabb, mint a felület egy frissítése?", expanded=False):
    st.markdown(
        "A Streamlit oldal egy interakcióra **egyszer** futtatja le a teljes láncot (kódolás → hiba → "
        "dekódolás / javítás). A kötegelt teszt ugyanezt **sokszor egymás után** megismétli (akár **500** "
        "független eset). **RS(8,4) GF(9)** ágon a `decode_rs84` belül **brute force**: sorban megpróbálja "
        "az összes ésszerű 1–2 szimbólumhibás magyarázatot, amíg a kódtérbe illeszkedő **ĉ**-t meg nem találja — "
        "egy fogadott szóra ez lehet fél–1 másodperc is; **500×** ez már összeadódik. "
        "A két **m** vektor összehasonlítása önmagában tényleg elhanyagolható."
    )
with st.expander("Mit jelentenek a számok egy kimeneti sorban?", expanded=False):
    st.markdown(
        "A fájl **tabulátorral elválasztott** (TSV); egy sor mezői egymás mellett külön „cellákban” vannak. "
        "Példa értelmezés **RS(8,4) GF(9)** ágon:\n\n"
        "- **eredeti m** pl. `4,8,6,2` = **m₀, m₁, m₂, m₃** GF(9) decimális címkék (**0…8**).\n"
        "- **hiba_db** pl. `2` = ennyi szándékos szimbólumhiba.\n"
        "- **hiba pozíciók** pl. `2,4` = a kódszó **j** indexei (**0…7**).\n"
        "- **fogadott r** / **javított c** = a két nyolcas int-vektor szövegesen; végül **eredmény_TEST**."
    )
st.caption(
    "Kimeneti fájl: **`exports/rs_streamlit_random_harness.tsv`** — UTF-8, **tabulátor (TSV)**, első sor = fejléc. "
    "Excelben: *Szövegből / adatok importálása* → elválasztó: tab. "
    "Ág: `RS(7,4)` vagy `RS(8,4) GF(9)`. Felülírás minden futáskor; Cloud-on a GitHub **nem** frissül automatikusan."
)

out_path = mh.default_output_path(repo_root=_ROOT)
st.code(str(out_path.relative_to(_ROOT)), language=None)

n = st.number_input("Tesztesetek száma (1…500)", min_value=1, max_value=500, value=10, step=1)
if st.button("Tesztek futtatása", type="primary"):
    n_int = int(n)
    rng = np.random.default_rng()
    with st.spinner(f"{n_int} teszt fut… (nagy szám és RS(8,4) esetek lassabbak lehetnek)"):
        rows = mh.run_random_batch(n_tests=n_int, rng=rng)
        mh.write_harness_file(out_path, rows)
    passed = sum(1 for t in rows if t[-1] == "TEST_TRUE")
    st.success(f"Kész: **{passed}** / **{len(rows)}** teszt **TEST_TRUE**. Írva: `{out_path}`.")
    _cols = mh.harness_header_line().split(mh.HARNESS_FIELD_SEP)
    _df = pd.DataFrame(rows, columns=_cols)
    st.subheader("Utolsó futás előnézete (táblázat)")
    st.dataframe(_df, use_container_width=True, hide_index=True, height=min(420, 35 + len(_df) * 36))
    with st.expander("Nyers TSV (első 15 sor, másolható)", expanded=False):
        _sep = mh.HARNESS_FIELD_SEP
        st.code("\n".join([mh.harness_header_line(), *[_sep.join(t) for t in rows[:15]]]), language=None)
