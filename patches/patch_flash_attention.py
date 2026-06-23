#!/usr/bin/env python3
"""
patch_flash_attention.py — fix transformers flash_attention s_aux=None crash
Fixes: line "s_aux=s_aux.to(query.dtype)" crashes when s_aux is None (Qwen3 never passes s_aux).
"""
import sys, py_compile
from pathlib import Path

def find_file(venv_root):
    p = Path(venv_root) / "lib"
    candidates = list(p.rglob("transformers/integrations/flash_attention.py"))
    if not candidates:
        print("[patch] ERROR: flash_attention.py not found under", venv_root); sys.exit(1)
    return candidates[0]

def apply(path):
    src = path.read_text(encoding="utf-8")
    old = "s_aux=s_aux.to(query.dtype)"
    new = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None"
    if new in src:
        print(f"[patch] flash_attention.py: already patched, skipping"); return
    if old not in src:
        print(f"[patch] WARNING: target string not found in {path} — transformers version may differ"); return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[patch] flash_attention.py: patched OK -> {path}")

if __name__ == "__main__":
    import site
    venv = Path(sys.prefix)
    apply(find_file(venv))
