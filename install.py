#!/usr/bin/env python3
"""把 completion-progress 安装/更新到各 agent 的 skills 目录（唯一来源 = 本目录）。

用法：
  python3 install.py --list                      # 列出常见 agent 目录是否存在
  python3 install.py --target ~/.claude/skills/completion-progress
  python3 install.py --target DIR --full         # 连 tests/studies/文档一起复制
  python3 install.py --all                       # 安装到所有检测到的目录
  python3 install.py --check DIR                 # 校验已安装文件（按 manifest.json 哈希）
  python3 install.py --uninstall DIR             # 只删运行时文件，不动 state/
  python3 install.py --print-env                 # 打印共享状态的环境变量设置
  python3 install.py --print-rule                # 打印可粘贴到 AGENTS.md/CLAUDE.md/GEMINI.md 的全局规则块
  python3 install.py --all --inject-rules        # 一键安装技能到所有已存在 agent 并注入/更新全局规则块

约定：运行时只需要 SKILL.md、estimate.py、reference.md；tests/studies/文档默认不装。
state/ 永远不被安装脚本删除或覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ["SKILL.md", "estimate.py", "reference.md"]
EXTRA = ["README.md", "INSTALL.md", "LICENSE", ".gitignore", "OPTIMIZATION-PLAN.md", "PROPOSAL-v3.md",
         "manifest.json", "SHA256SUMS", "env.sh", "install.py", "make_manifest.py", "rollout_all.py"]
EXTRA_DIRS = ["tests", "studies"]
CANDIDATES = {
    "claude": "~/.claude/skills/completion-progress",
    "codex": "~/.codex/skills/completion-progress",
    "gemini": "~/.gemini/skills/completion-progress",
    "cursor": "~/.cursor/skills/completion-progress",
    "agents": "~/.agents/skills/completion-progress",
    "gemini-config": "~/.gemini/config/skills/completion-progress",
    "junie": "~/.junie/skills/completion-progress",
    "copilot": "~/.copilot/skills/completion-progress",
    "trae": "~/.trae/skills/completion-progress",
    "minimax": "~/.minimax/skills/completion-progress",
    "openclaw": "~/.openclaw/skills/completion-progress",
    "hermes": "~/.hermes/skills/completion-progress",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest() -> dict:
    m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    return {f["path"]: f for f in m["files"]}


def install(target: Path, full: bool) -> int:
    files = list(RUNTIME) + (EXTRA if full else [])
    dirs = EXTRA_DIRS if full else []
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in files:
        src = ROOT / name
        if not src.exists():
            continue
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    for d in dirs:
        for src in sorted((ROOT / d).rglob("*")):
            if src.is_file() and "__pycache__" not in src.parts:
                dst = target / src.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    print(f"已安装到 {target}（{copied} 个文件，{'完整' if full else '运行时'}）")
    return 0


def check(target: Path) -> int:
    """校验已存在的文件；缺失只对运行时三件套算失败，其余可选文件列出即可。"""
    man = manifest()
    bad, miss_rt, miss_opt = [], [], []
    for rel, info in man.items():
        if info.get("role") == "state":
            continue
        p = target / rel
        if not p.exists():
            (miss_rt if rel in RUNTIME else miss_opt).append(rel)
        elif sha(p) != info["sha256"]:
            bad.append(rel)
    if not bad and not miss_rt:
        extra = f"，另有 {len(miss_opt)} 个可选文件未安装" if miss_opt else ""
        print(f"{target}: 运行时文件与 manifest 一致{extra}")
        return 0
    if miss_rt:
        print("缺运行时文件：" + ", ".join(miss_rt))
    if bad:
        print("内容不同：" + ", ".join(bad))
    return 1


def uninstall(target: Path) -> int:
    removed = 0
    for name in RUNTIME:
        p = target / name
        if p.exists():
            p.unlink()
            removed += 1
    print(f"已从 {target} 删除 {removed} 个运行时文件（state/ 与额外文件未动）")
    return 0


def list_targets() -> int:
    print(f"唯一来源：{ROOT}")
    for name, path in CANDIDATES.items():
        p = Path(path).expanduser()
        mark = "存在" if p.exists() else "未安装"
        print(f"  {name:>7}  {str(p):<45} {mark}")
    return 0


def print_env() -> int:
    home = Path.home()
    state = ROOT / "state"
    shown = f'"{state.as_posix()}"'
    try:
        rel = state.relative_to(home)
        # 沙箱里 ROOT 常挂在 ~/mnt/... 下，用 $HOME 拼会指错；只在看起来是真实家目录时用 $HOME
        if rel.parts and rel.parts[0] != "mnt":
            shown = f'"$HOME/{rel.as_posix()}"'
    except ValueError:
        pass
    print("# 多个 agent 共用同一份历史（放进各自的 shell 配置或宿主环境变量）")
    print(f"export COMPLETION_PROGRESS_STATE={shown}")
    return 0


def print_rule() -> int:
    import rollout_all
    print(rollout_all.render_block(None))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[], help="安装目录，可重复")
    ap.add_argument("--all", action="store_true", help="安装到所有已存在的候选目录")
    ap.add_argument("--full", action="store_true", help="连 tests/studies/文档一起复制")
    ap.add_argument("--inject-rules", action="store_true", help="同时向本机已存在的 agent 全局规则文件注入/更新触发规则块")
    ap.add_argument("--check")
    ap.add_argument("--uninstall")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument("--print-rule", action="store_true", help="打印全局规则块模板")
    a = ap.parse_args()
    if a.list:
        return list_targets()
    if a.print_env:
        return print_env()
    if a.print_rule:
        return print_rule()
    if a.check:
        return check(Path(a.check).expanduser())
    if a.uninstall:
        return uninstall(Path(a.uninstall).expanduser())
    if a.inject_rules and a.all and not a.target and not a.full:
        import rollout_all
        return rollout_all.main([])
    targets = [Path(t).expanduser() for t in a.target]
    if a.all:
        targets += [Path(p).expanduser() for p in CANDIDATES.values() if Path(p).expanduser().parent.exists()]
    if not targets:
        if a.inject_rules:
            import rollout_all
            return rollout_all.main([])
        ap.print_help()
        return 2
    rc = 0
    for t in targets:
        rc |= install(t, a.full)
    if a.inject_rules:
        import rollout_all
        rc |= rollout_all.main([])
    return rc


if __name__ == "__main__":
    sys.exit(main())
