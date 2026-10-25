#!/usr/bin/env python3
"""生成 manifest.json：记录每个文件的角色与哈希。修改包内容后运行一次。"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = {"SKILL.md", "estimate.py", "reference.md"}
DOCS = {"README.md", "INSTALL.md", "LICENSE", ".gitignore", "OPTIMIZATION-PLAN.md", "PROPOSAL-v3.md", "env.sh", "install.py", "make_manifest.py", "rollout_all.py"}
SELF = {"manifest.json", "SHA256SUMS", ".DS_Store"}      # 自身与校验和文件不列入，避免循环依赖
SKIP_DIRS = {"state", "__pycache__", "backups", ".git"}


def role(rel: str) -> str:
    if rel in RUNTIME:
        return "runtime"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("studies/"):
        return "study"
    if rel.startswith("state/"):
        return "state"
    if rel in DOCS:
        return "doc"
    return "other"


def main() -> int:
    version = "unknown"
    for line in (ROOT / "estimate.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("VERSION ="):
            version = line.split("=", 1)[1].strip().strip('"')
            break
    files = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in SELF:
            continue
        files.append({"path": rel, "role": role(rel), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                      "bytes": p.stat().st_size})
    manifest = {"name": "completion-progress", "version": version, "files": files}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"manifest.json: {len(files)} 项，version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
