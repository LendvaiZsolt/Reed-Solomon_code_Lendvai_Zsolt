from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
README_PATH = ROOT_DIR / 'README.md'

st.set_page_config(page_title='Dokumentáció', layout='wide')

with st.sidebar:
    st.page_link('app.py', label='app - RS(7,4) rész')
    st.page_link('pages/2_rs84.py', label='rs84 - RS(8,4) rész')
    st.page_link('pages/3_dokumentacio.py', label='Használati útmutató')
    st.divider()
    st.caption('A tartalom a projekt gyökerében lévő README.md-ből töltődik.')

if README_PATH.exists():
    st.markdown(README_PATH.read_text(encoding='utf-8'))
else:
    st.error('A README.md nem található a projekt gyökérmappájában.')
