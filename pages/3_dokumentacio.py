from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import nav_visibility
README_PATH = ROOT_DIR / 'README.md'

st.set_page_config(page_title='Dokumentáció', layout='wide')

with st.sidebar:
    st.page_link('app.py', label='RS(7,4)')
    st.page_link('pages/2_rs84.py', label='RS(8,4) GF(9)')
    if nav_visibility.SHOW_RS84_GF16_PAGE_LINK:
        st.page_link('pages/2_rs84_gf16.py', label='RS(8,4) GF(16) - törlésre kerül')
    st.page_link('pages/3_dokumentacio.py', label='Használati útmutató')
    st.divider()
    st.caption('A tartalom a projekt gyökerében lévő README.md-ből töltődik.')

if README_PATH.exists():
    st.markdown(README_PATH.read_text(encoding='utf-8'))
else:
    st.error('A README.md nem található a projekt gyökérmappájában.')
