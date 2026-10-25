#!/usr/bin/env python3
"""真实历史回测（只读）：按时间顺序一步前瞻，比较 当时实际给出的预测 / v3（3.1 归一后，参数化区间）/ v3+经验区间（3.1 默认）/ 影子 v4。

用法：python3 studies/backtest_real.py [状态目录]   （默认 $COMPLETION_PROGRESS_STATE 或 ~/.local/state/completion-progress）
只评有效计时样本（w>0）；分“全部”和“≥60 秒”两组；指标：对数 MAE、中位偏差（正=实际更慢）、名义 60% 区间覆盖。
v4 的切换门槛见 reference.md §1.15：影子期新样本上 log_mae ≤ v3−0.05、|bias| ≤ 0.10、覆盖 0.50–0.70、n ≥ 100。
"""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("est", HERE / "estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)


def score(pairs):
    if not pairs:
        return "n=0"
    e = [math.log(a / p) for a, p, _ in pairs]
    cov = sum(lo <= a <= hi for a, _, (lo, hi) in pairs) / len(pairs)
    return (f"n={len(pairs):3d} log_mae={statistics.fmean(abs(x) for x in e):.3f} "
            f"median_abs={statistics.median(abs(x) for x in e):.3f} bias={statistics.median(e):+.3f} cover60={cov:.0%}")


def main() -> int:
    d = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else est.state_dir(None)
    steps, tasks, info = est.load_history(d / "history.jsonl", time.time())
    clean = [r for r in steps if r["w"] > 0]
    kappa = est.load_kappa(d)[0]
    print(f"history: {info['step_rows']} steps ({len(steps) - len(clean)} zero-weight), {info['task_rows']} tasks, kappa={kappa}")
    res = {k: {"all": [], "60": []} for k in ("issued", "v3", "v3cal", "v4")}
    t0 = time.time()
    for i, r in enumerate(clean):
        past = [p for p in clean if p["completed_at"] <= r["started_at"]]
        if len(past) < 30:
            continue
        a = r["actual_sec"]
        grp = ("all", "60") if a >= 60 else ("all",)
        st = dict(type=r["type"], u=r["u"], q=r["q"], d=r["d"], base_sec=r.get("base_sec"))
        model = est.Model(past[-600:], kappa, extra=[dict(st, model=r["model"])])
        counts = sum(1 for p in past if est.tkey(p) == est.tkey(r))
        fc = est.step_forecast(model, st, r["model"], counts, est.base_stats(past, model.s2),
                               est.tau2_from([t for t in tasks if t["completed_at"] <= r["started_at"]], model.s2))
        tau2 = est.tau2_from([t for t in tasks if t["completed_at"] <= r["started_at"]], model.s2)
        fcal = est.apply_calib(fc, est.calib_width(past), tau2) if fc else None
        f4 = est.ShadowV4(past, kappa).forecast(st, r["model"])
        if not (fc and f4 and r.get("pred_sec") and r.get("range_sec")):
            continue                                   # 三者都有才同批比较
        for g in grp:
            res["issued"][g].append((a, r["pred_sec"], r["range_sec"]))
            res["v3"][g].append((a, fc["pred_sec"], fc["range_sec"]))
            res["v3cal"][g].append((a, fcal["pred_sec"], fcal["range_sec"]))
            res["v4"][g].append((a, f4["pred_v4"], f4["range_v4"]))
    for g, label in (("all", "全部有效样本"), ("60", "≥60 秒")):
        print(f"\n[{label}]")
        for k, name in (("issued", "当时给出(3.0)"), ("v3", "v3 (3.1 归一)"), ("v3cal", "v3+经验区间"), ("v4", "影子 v4")):
            print(f"  {name:14} {score(res[k][g])}")
    print(f"\n用时 {time.time() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
