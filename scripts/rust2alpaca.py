#!/usr/bin/env python3
"""
rust2alpaca.py
─────────────────────
Convert local Rust corpora (The‑Stack, Axum examples, etc.) into
Alpaca‑style JSONL, one object per line:

  {"instruction": "...", "input": "", "output": "..."}

Key features
------------
* Streams results → **constant RAM** (< 500 MB even on huge corpora)
* **Per‑file timeout** (skips files that hang or are corrupt)
* Skips blobs bigger than MAX_BYTES (default 25 kB) to avoid regex stalls
* NEW: `--skip-parquet`  → ignore all `.parquet` files for a fast first pass

Dependencies
------------
pip install orjson pyarrow tqdm
"""

import os, re, time, argparse, signal, multiprocessing as mp
import orjson
import pyarrow.dataset as ds
import pyarrow.json as paj
from tqdm import tqdm
from typing import List, Dict, Tuple

MAX_BYTES    = 25_000   # ignore very large code blobs (> this many bytes)
FILE_TIMEOUT = 300      # seconds before a worker gives up on one file

# ───────── regexes (pre‑compiled) ─────────
MOD_RE  = re.compile(r'(//.*\n)?\s*mod\s+([\w_]+)\s*{(?:[^{}]*|{[^}]*})*}', re.M|re.S)
TEST_RE = re.compile(r'(///[^\n]*\n)*\s*#\[test\]\s*(pub\s+)?fn\s+\w+\s*\([^)]*\)\s*{(?:[^{}]*|{[^}]*})*}', re.M|re.S)
FUNC_RE = re.compile(r'(///[^\n]*\n)+\s*(pub\s+)?fn\s+\w+\s*\([^)]*\)\s*{(?:[^{}]*|{[^}]*})*}', re.M|re.S)
USE_RE  = re.compile(r'(//.*\n)?\s*use\s+[A-Za-z0-9_:{} ,*]+;')

def extract(src: str) -> List[Tuple[str, str]]:
    """Return (doc, code) pairs from a Rust source string."""
    if len(src) > MAX_BYTES:
        return []  # skip giant generated files
    pairs=[]
    for m in MOD_RE.finditer(src):
        doc=(m.group(1) or "").strip()
        pairs.append((f"Module: {doc or f'`{m.group(2)}`'}", m.group(0).strip()))
    for m in TEST_RE.finditer(src):
        doc=" ".join(re.findall(r'///(.*)', m.group())) or "Rust test function"
        pairs.append((doc, m.group().strip()))
    for m in FUNC_RE.finditer(src):
        doc=" ".join(re.findall(r'///(.*)', m.group()))
        pairs.append((doc, m.group().strip()))
    for m in USE_RE.finditer(src):
        doc=(m.group(1) or "").strip() or "Rust use statement"
        pairs.append((doc, m.group(0).strip()))
    return pairs

def to_alpaca(instr: str, code: str) -> Dict[str, str]:
    return {"instruction": instr or "Rust snippet", "input": "", "output": code}

# ───────── time‑boxed worker helpers ─────────
def _timeout_handler(signum, frame):
    raise TimeoutError

def process_file(args):
    """Worker: parse one file under a timeout; return list of Alpaca dicts."""
    path, skip_parquet = args
    ext = os.path.splitext(path)[1].lower()
    if skip_parquet and ext == ".parquet":
        return []

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FILE_TIMEOUT)
    try:
        if ext == ".rs":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [to_alpaca(i, c) for i, c in extract(f.read())]

        if ext == ".parquet" and not skip_parquet:
            out=[]
            for batch in ds.dataset(path, format="parquet").to_batches():
                if {"lang", "content"} <= set(batch.schema.names):
                    langs    = batch["lang"].to_pylist()
                    contents = batch["content"].to_pylist()
                    for lang, content in zip(langs, contents):
                        if lang == "Rust" and content:
                            out.extend(to_alpaca(i, c) for i, c in extract(content))
            return out

        if ext in (".json", ".jsonl"):
            out=[]
            tbl = paj.read_json(path, read_options=paj.ReadOptions(use_threads=True))
            if {"lang", "content"} <= set(tbl.column_names):
                for lang, content in zip(tbl["lang"].to_pylist(),
                                         tbl["content"].to_pylist()):
                    if lang == "Rust" and content:
                        out.extend(to_alpaca(i, c) for i, c in extract(content))
            return out
    except TimeoutError:
        return []  # skip slow/corrupt file
    except Exception:
        return []  # swallow any other error
    finally:
        signal.alarm(0)

# ───────── gather & write ─────────
def gather(root: str, out_path: str, limit: int | None, skip_parquet: bool) -> int:
    allowed_exts = {".rs", ".json", ".jsonl"}
    if not skip_parquet:
        allowed_exts.add(".parquet")

    files = [os.path.join(dp, f)
             for dp, _, fs in os.walk(root)
             for f in fs
             if os.path.splitext(f)[1].lower() in allowed_exts]

    written = 0
    with mp.get_context("fork").Pool() as pool, \
         open(out_path, "wb") as fout, \
         tqdm(total=len(files), desc="files") as bar:

        for results in pool.imap_unordered(process_file,
                                           [(p, skip_parquet) for p in files],
                                           chunksize=4):
            bar.update()
            for obj in results:
                fout.write(orjson.dumps(obj))
                fout.write(b"\n")
                written += 1
                if limit and written >= limit:
                    pool.terminate(); pool.join()
                    return written
    return written

# ───────── CLI entry ─────────
def main():
    ap = argparse.ArgumentParser("Rust → Alpaca JSONL converter (streaming)")
    ap.add_argument("--input-path", required=True,
                    help="Top‑level folder with the‑stack‑*, axum_examples, etc.")
    ap.add_argument("--output-path", required=True,
                    help="Destination .jsonl file")
    ap.add_argument("--limit", type=int,
                    help="Cap total examples (quick test)")
    ap.add_argument("--skip-parquet", action="store_true",
                    help="Ignore all .parquet files for a fast first pass")
    args = ap.parse_args()

    if not os.path.isdir(args.input_path):
        ap.error("input-path must exist and be a directory")

    t0 = time.time()
    total = gather(args.input_path, args.output_path, args.limit, args.skip_parquet)
    mins = (time.time() - t0) / 60
    print(f"✅ Wrote {total:,} examples → {args.output_path} in {mins:.1f} min")

if __name__ == "__main__":
    main()
