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
    st.page_link("pages/4_veletlen_hibateszt.py", label="Véletlen hibateszt")
    st.page_link("pages/3_dokumentacio.py", label="Használati útmutató")
    st.divider()

st.markdown(
    "Soronként véletlen ág: **RS(8,4) GF(9)** kétszer akkora eséllyel, mint **RS(7,4)**; az ágon belül ugyanolyan "
    "véletlen beállítások, mint a fő oldalak bal oldalán. "
    "0–2 hiba (RS(7,4)-nél legfeljebb 1 szándékos hiba). "
    "A **TEST_TRUE** azt jelenti, hogy a javított kódszó megegyezik a küldött **c** kódszóval."
)

out_path = mh.default_output_path(repo_root=_ROOT)

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
