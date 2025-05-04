#!/usr/bin/env bash
set -euo pipefail

# → Adjust if your repo lives elsewhere
AXO=~/Documents/GitHub-public/axolotl
cd "$AXO"

echo "🔄  Cleaning conflicting wheels ..."
pip uninstall -y bitsandbytes triton torchao || true
pip uninstall -y transformers tokenizers accelerate || true

echo "⬇️  Installing HF stack required by Axolotl ..."
pip install "transformers==4.51.3" "tokenizers>=0.21.1" "accelerate>=0.25.1"

echo "🩹  Writing stub modules for CUDA‑only deps ..."
python - <<'PY'
import types, sys, site, pathlib, textwrap

STUBS = {
    # Whole packages
    "bitsandbytes": "",
    "triton": "",
    "torchao": textwrap.dedent(
        """
        def __getattr__(name):
            # any symbol access just returns a dummy func/None
            def _dummy(*args, **kwargs): return None
            return _dummy
        """
    ),
    # Common sub‑modules that may be imported directly
    "triton.language": "",
    "triton.compiler": "",
    "triton.compiler.compiler": "",
}

site_dir = next(p for p in site.getsitepackages() if "site-packages" in p)
for mod, body in STUBS.items():
    sys.modules[mod] = types.ModuleType(mod)
    path = pathlib.Path(site_dir, *mod.split("."))  # nested dirs for submods
    if path.suffix != ".py":
        path = path.with_suffix(".py")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if not path.exists():
        path.write_text(body)
        print(f"✓ stubbed {mod} -> {path}")
PY

echo "🩹  Patching Axolotl to skip Params4bit import ..."
sed -i '' 's/^from bitsandbytes.nn import Params4bit/# macOS: skip Params4bit/' \
    src/axolotl/utils/models.py || true

echo "✅  Environment ready for Axolotl on Apple Silicon!"
