"""
Make Apple‑Silicon‑safe stub modules so Axolotl (and its HF deps) stop
importing CUDA‑only libraries.  Creates minimal packages for:

  • bitsandbytes
  • triton, triton.language, triton.compiler, triton.compiler.compiler
  • torchao
  • hqq.core.quantize (with Quantizer & HQQLinear classes)

Run:  python patch_stubs.py

Re‑run after you update packages — it’s safe & quick.
"""
from __future__ import annotations
import site, sys, types, pathlib, textwrap

# ── find site‑packages dir for this interpreter ─────────────────────────────
SITE_DIR = next(p for p in site.getsitepackages() if "site-packages" in p)

def write_stub(dotted_name: str, body: str = "") -> None:
    """
    Create a stub module/package on disk AND register it in sys.modules
    so current & future interpreters can import it.
    """
    # Register an in‑memory module immediately
    if dotted_name in sys.modules:
        mod = sys.modules[dotted_name]
    else:
        mod = types.ModuleType(dotted_name)
        sys.modules[dotted_name] = mod

    # If we want it to behave like a *package*, give it __path__
    if "." not in dotted_name or body == "PACKAGE":
        mod.__path__ = []  # mark as pkg (empty list is fine)

    # Build the path:  site‑packages/a/b/c.py or .../a/b/__init__.py
    path = pathlib.Path(SITE_DIR, *dotted_name.split("."))
    if path.suffix != ".py":
        # It's a package; ensure dirs and use __init__.py
        (path / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
        path = path / "__init__.py"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text("" if body in ("", "PACKAGE") else body)
        print(f"✓ stubbed {dotted_name}  →  {path.relative_to(SITE_DIR)}")
    else:
        # Ensure file at least contains the desired body symbols
        if body and body not in path.read_text():
            with path.open("a") as fh:
                fh.write("\n" + body)
            print(f"✓ updated {dotted_name} with extra body")

# ── 1. simple empty-module stubs ────────────────────────────────────────────
for mod in ("bitsandbytes", "torchao"):
    write_stub(mod)

# ── 2. Triton needs to look like a *package* with sub‑modules ───────────────
write_stub("triton", "PACKAGE")
for sub in ("language", "compiler", "compiler.compiler"):
    write_stub(f"triton.{sub}")

# ── 3. hqq.core.quantize with Quantizer & HQQLinear symbols ────────────────
hqq_body = textwrap.dedent(
    """
    # Minimal stub so PEFT can import Quantizer / HQQLinear.
    class Quantizer: ...
    class HQQLinear: ...
    """
)
write_stub("hqq", "PACKAGE")
write_stub("hqq.core", "PACKAGE")
write_stub("hqq.core.quantize", hqq_body)

print("🎉  All CUDA‑only stubs installed or refreshed.")
