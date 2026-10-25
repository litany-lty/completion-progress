#!/usr/bin/env python3
"""把 completion-progress 3.1 全局同步到本机所有已安装的 agent（技能目录 + 全局规则块）。

用法：
  python3 rollout_all.py                  # 同步技能目录 + 更新/注入已存在规则文件中的规则块（自动备份）
  python3 rollout_all.py --create-rules   # 若某已安装 agent 还没有全局规则文件，一并创建
  python3 rollout_all.py --dry-run        # 仅预览将修改的目标，不落盘
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path

H = Path.home()
SRC = Path(__file__).resolve().parent
RUNTIME = ["SKILL.md", "estimate.py", "reference.md"]
SKILL_DIRS = [
    ".claude", ".codex", ".gemini", ".gemini/config", ".agents",
    ".junie", ".copilot", ".trae", ".minimax", ".openclaw", ".hermes",
]
# (规则文件相对家目录路径, 对应的技能目录相对家目录路径或 None)
RULE_TARGETS = [
    (".codex/AGENTS.md", ".codex/skills/completion-progress"),
    (".gemini/GEMINI.md", ".gemini/skills/completion-progress"),
    (".claude/CLAUDE.md", ".claude/skills/completion-progress"),
    (".config/opencode/AGENTS.md", None),
    (".dsh/AGENTS.md", None),
    (".aider/CONVENTIONS.md", None),
    (".minimax/AGENTS.md", ".minimax/skills/completion-progress"),
    (".trae/user_rules.md", ".trae/skills/completion-progress"),
    (".hermes/AGENTS.md", ".hermes/skills/completion-progress"),
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def render_block(skill_rel: str | None) -> str:
    copy_line = (
        f"- 本机技能副本：{(H / skill_rel).as_posix()}"
        if skill_rel and (H / skill_rel).exists()
        else "- 本机未提供独立 skills 副本时，直接读取上方规则全文并调用下方脚本路径"
    )
    return f"""<!-- completion-progress:begin -->
## 进度与预计（completion-progress 3.1）
预计超过 3 分钟的实质工作启用；只观察，不新增任务、不改业务文件、不写进交付物。
- 规则全文：{(SRC / 'SKILL.md').as_posix()}
{copy_line}
- 脚本：python3 {(SRC / 'estimate.py').as_posix()} plan|progress|status|deliver|abandon|audit
- 精度要点：每步真正完成时立即 --done；等待用 --block；明显超时加 --cause；--model 写真实型号（不写“当前会话”）
- 共享历史：默认 ~/.local/state/completion-progress（同机 agent 共用）；宿主沙箱可用 --state 指定
- 只贴脚本返回的 line，且仅当 should_report 为真
<!-- completion-progress:end -->"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create-rules", action="store_true", help="为已存在宿主目录但尚无规则文件的 agent 创建规则文件")
    ap.add_argument("--no-inject", action="store_true", help="只更新已有 completion-progress 标记的规则文件，不向无标记文件追加")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的操作，不实际写入")
    a = ap.parse_args(argv)

    bk = SRC / "backups" / ("rollout-" + time.strftime("%Y%m%d-%H%M%S"))
    src_hash = {f: sha(SRC / f) for f in RUNTIME}
    report = []

    for d in SKILL_DIRS:
        t = H / d / "skills/completion-progress"
        parent = H / d
        if not t.exists() and not parent.exists():
            continue
        if a.dry_run:
            report.append(f"DRY  skill {t}")
            continue
        bk.mkdir(parents=True, exist_ok=True)
        if t.exists():
            shutil.copytree(t, bk / (d.strip(".").replace("/", "_") + "_skills_completion-progress"), symlinks=True)
            shutil.rmtree(t)
        t.mkdir(parents=True, exist_ok=True)
        for f in RUNTIME:
            shutil.copy2(SRC / f, t / f)
        ok = sorted(p.name for p in t.iterdir()) == sorted(RUNTIME) and all(sha(t / f) == src_hash[f] for f in RUNTIME)
        report.append(f"{'OK  ' if ok else 'FAIL'} skill {t}")

    for r_rel, s_rel in RULE_TARGETS:
        p = H / r_rel
        if not p.exists():
            if not (a.create_rules and p.parent.exists()):
                continue
            text = ""
        else:
            text = p.read_text(encoding="utf-8")
        m = re.search(r"<!-- completion-progress:begin -->.*?<!-- completion-progress:end -->", text, re.S)
        if not m and a.no_inject:
            continue
        new_block = render_block(s_rel)
        if m:
            cm = re.search(r"^- (本机技能副本：.*|本机未提供.*)$", m.group(0), re.M)
            if cm and s_rel and (H / s_rel).exists():
                new_block = render_block(s_rel)
            updated = text[:m.start()] + new_block + text[m.end():]
        else:
            sep = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
            updated = text + sep + new_block + "\n"
        if a.dry_run:
            report.append(f"DRY  rule  {p} ({'update' if m else 'inject'})")
            continue
        bk.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copy2(p, bk / r_rel.strip(".").replace("/", "_"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(updated, encoding="utf-8")
        ok = p.read_text(encoding="utf-8").count("completion-progress 3.1") == 1
        report.append(f"{'OK  ' if ok else 'FAIL'} rule  {p}")

    print("\n".join(report) if report else "未检测到可更新的目标目录")
    if not a.dry_run and bk.exists():
        print(f"备份：{bk}")
    return 1 if any(x.startswith("FAIL") for x in report) else 0


if __name__ == "__main__":
    sys.exit(main())
