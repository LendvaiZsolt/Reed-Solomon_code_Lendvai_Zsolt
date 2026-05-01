#!/usr/bin/env python3
"""
RS(8,4): olyan (c, r) párok, ahol d_H(c,r)=3 és c az egyetlen legközelebbi kódszó r-hez
(ML: egyedi minimum távolság). Kimenet: Markdown tábla felületre.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rs84_core as rc  # noqa: E402

N, K = rc.N, rc.K
LETTERS = rc.LETTER_ORDER


def ints_to_word(v: np.ndarray) -> str:
    return "".join(LETTERS[int(x) & 15] for x in np.asarray(v).flatten())


def ints_to_word_bold_errors(c: np.ndarray, r: np.ndarray) -> str:
    """Minden hibás szimbólum külön HTML <b>…</b> — nincs Markdown-csillag, nincs összevont blokk."""
    parts: list[str] = []
    for i in range(N):
        ch = LETTERS[int(r[i]) & 15]
        if int(c[i]) != int(r[i]):
            parts.append(f"<b>{ch}</b>")
        else:
            parts.append(ch)
    return "".join(parts)


def main() -> int:
    rng = np.random.default_rng(int(os.environ.get("COLLECT_RS84_SEED", "42")))
    out = Path(os.environ.get("COLLECT_RS84_OUT", str(ROOT / "exports" / "rs84_ml3_unique_nearest_pairs.md")))
    out.parent.mkdir(parents=True, exist_ok=True)

    F = rc.F
    add = np.zeros((16, 16), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            add[i, j] = int(F(i) + F(j)) & 15

    from itertools import product

    rows: list[list[int]] = []
    for m_tuple in product(range(16), repeat=K):
        m = F(list(m_tuple)).reshape(1, K)
        c = [int(x) for x in (m @ rc.G_BASE).flatten()]
        rows.append(c)
    CW = np.array(rows, dtype=np.uint8)
    del rows

    max_rows = int(os.environ.get("COLLECT_RS84_MAX_ROWS", "4000"))
    batch = int(os.environ.get("COLLECT_RS84_BATCH", "384"))
    t_limit = float(os.environ.get("COLLECT_RS84_SECONDS", "120"))

    seen_r: set[tuple[int, ...]] = set()
    table_rows: list[tuple[str, str, str, str]] = []

    t0 = time.perf_counter()
    batches = 0
    while len(table_rows) < max_rows and (time.perf_counter() - t0) < t_limit:
        batches += 1
        ci = rng.integers(0, CW.shape[0], size=batch, dtype=np.int64)
        c_batch = CW[ci]  # (batch, N)

        err = np.zeros((batch, N), dtype=np.uint8)
        for b in range(batch):
            pos = rng.choice(N, size=3, replace=False)
            for p in pos:
                err[b, p] = int(rng.integers(1, 16))

        r_batch = add[c_batch, err]  # (batch, N)

        dists = np.zeros((CW.shape[0], batch), dtype=np.uint8)
        for k in range(N):
            dists += (CW[:, k : k + 1] != r_batch[np.newaxis, :, k]).astype(np.uint8)

        min_d = dists.min(axis=0)
        argmin = dists.argmin(axis=0)
        cnt_min = (dists == min_d[np.newaxis, :]).sum(axis=0)

        for j in range(batch):
            if int(min_d[j]) != 3:
                continue
            if int(cnt_min[j]) != 1:
                continue
            if int(argmin[j]) != int(ci[j]):
                continue
            rt = tuple(int(x) for x in r_batch[j])
            if rt in seen_r:
                continue
            seen_r.add(rt)

            c_vec = c_batch[j]
            r_vec = r_batch[j]
            dcol = dists[:, j]
            at4 = np.where(dcol == 4)[0]
            words4 = [ints_to_word(CW[k]) for k in at4.tolist()]
            if not words4:
                col4 = "—"
            elif len(words4) <= 4:
                col4 = ", ".join(words4)
            else:
                col4 = ", ".join(words4[:4]) + ", …"

            table_rows.append(
                (
                    ints_to_word(c_vec),
                    ints_to_word_bold_errors(c_vec, r_vec),
                    ints_to_word(CW[int(argmin[j])]),
                    col4,
                )
            )

    lines = [
        "# RS(8,4) — 3 szimbólumhiba, egyedi legközelebbi kódszó",
        "",
        "Feltétel: Hamming-távolság a fogadott <b>r</b>-hez a legközelebbi kódszó egyedül az eredeti <b>c</b>, és d(c,r)=3.",
        "",
        "| kódszó c | r (hibás szimbólumok kiemelve) | legközelebbi kódszó (d=3) | d=4 távolságra: első 4 kódszó, … |",
        "| --- | --- | --- | --- |",
    ]
    for a, b, c, d in table_rows:
        lines.append(f"| {a} | {b} | {c} | {d} |")

    lines.extend(
        [
            "",
            f"*Generálva: {len(table_rows)} sor, {batches} batch (batch={batch}), időkorlát {t_limit}s, seed={os.environ.get('COLLECT_RS84_SEED', '42')}.*",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(table_rows)} rows to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
