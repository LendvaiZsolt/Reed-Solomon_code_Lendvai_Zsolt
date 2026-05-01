from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
README_PATH = ROOT_DIR / 'README.md'


def _split_readme_trailing_github_commit_url(text: str) -> tuple[str, str, str] | None:
    """Ha a README vége: Verziószám: … majd egy soros github.com commit URL, visszaadja (fej, verziósor, url)."""
    lines = text.splitlines()
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 1:
        return None
    url_line = lines[i].strip()
    if not url_line.startswith('https://github.com/') or '/commit/' not in url_line:
        return None
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j < 0:
        return None
    ver_line = lines[j].strip()
    if not ver_line.startswith('Verziószám:'):
        return None
    head = '\n'.join(lines[:j]).rstrip()
    return (head, ver_line, url_line)


st.set_page_config(page_title='Dokumentáció', layout='wide')

with st.sidebar:
    st.page_link('app.py', label='app - RS(7,4) rész')
    st.page_link('pages/2_rs84.py', label='rs84 - RS(8,4) rész')
    st.page_link('pages/3_dokumentacio.py', label='Használati útmutató')
    st.divider()
    st.caption('A tartalom a projekt gyökerében lévő README.md-ből töltődik.')

if README_PATH.exists():
    raw = README_PATH.read_text(encoding='utf-8')
    parts = _split_readme_trailing_github_commit_url(raw)
    if parts is not None:
        head, ver_line, url_line = parts
        st.markdown(head)
        st.markdown(ver_line)
        st.code(url_line, language=None)
    else:
        st.markdown(raw)
else:
    st.error('A README.md nem található a projekt gyökérmappájában.')

