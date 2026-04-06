from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / 'rs74_app.py'), run_name='__main__')
