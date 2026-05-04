"""Oldalsáv-navigáció: opcionálisan rejtett oldalak.

A ``pages/2_rs84_gf16.py`` kód és az oldal futása megmarad; csak a ``st.page_link``
sorok nem jelennek meg, ha a kapcsoló ``False`` (közvetlen URL / fájlnév továbbra is
megnyitható).
"""

# Állítsd ``True``-ra, ha a GF(16) RS(8,4) link újra látható legyen a sidebarban.
SHOW_RS84_GF16_PAGE_LINK = False
