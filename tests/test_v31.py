#!/usr/bin/env python3
"""3.1 定向测试：归一、伪迹降权、拆分计时、等待扣除、原因码、证据截断、重划基线不丢行、放弃与收口、影子 v4、性能。
只在临时目录里写；运行: python3 tests/test_v31.py
"""
from __future__ import annotations

import importlib.util, json, math, os, random, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("est", HERE / "estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)
PY = sys.executable
FAILS = []
T0 = 1_790_000_000.0


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def run(args, state, at, env_extra=None):
    env = dict(os.environ, COMPLETION_PROGRESS_STATE=state, TZ="Asia/Shanghai")
    for k in ("COMPLETION_PROGRESS_MODEL", "COMPLETION_PROGRESS_ESTIMATOR"):
        env.pop(k, None)
    env.update(env_extra or {})
    p = subprocess.run([PY, str(HERE / "estimate.py")] + args + ["--at", str(at)], capture_output=True, text=True, env=env)
    try:
        return p.returncode, json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
    except (ValueError, IndexError):
        return p.returncode, {"raw": p.stdout, "err": p.stderr}


def history(state):
    f = Path(state) / "history.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


def seed_history(state, n=80, seed=1):
    """带 base 的合成历史：gemini 实际 ≈ base×2，gpt-6.1 ≈ base×0.5。"""
    rng = random.Random(seed)
    rows, t = [], T0 - 30 * 86400
    for i in range(n):
        fam = "gemini-4/high" if i % 2 else "gpt-6.1-sol/xhigh"
        base = rng.choice([180, 300, 600])
        actual = base * (2.0 if i % 2 else 0.5) * math.exp(rng.gauss(0, 0.3))
        rows.append(dict(v=3, kind="step", id=f"h/{i}/s1", session="h", task=str(i), type=rng.choice(["implement", "verify"]),
                         u="step", d="D3", q=1.0, model=fam, actual_sec=actual, base_sec=float(base),
                         started_at=t, completed_at=t + actual, timing="auto"))
        t += actual + 60
    Path(state).mkdir(parents=True, exist_ok=True)
    with open(Path(state) / "history.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(est.canonical(r) + "\n")


def main() -> int:
    # 1 归一
    check("型号族：gpt-6.1 不被 gpt-6 吞掉", est.norm_model("gpt-6.1-sol/xhigh") == "gpt-6.1" and est.norm_model("GPT-6/当前配置") == "gpt-6")
    check("型号族：长句取首个模型、占位写法归 unknown",
          est.norm_model("current-root;gpt-6.1-sol/xhigh") == "gpt-6.1" and est.norm_model("当前会话/未另行查询") == "unknown"
          and est.norm_model("Codex/session-config") == "unknown" and est.norm_model("Gemini 4 Argon/High") == "gemini")
    check("型号族：未知但明确的写法保留首段", est.norm_model("A/high") == "a")
    check("类型：自由写法归类", est.norm_type("runtime-check") == "verify" and est.norm_type("code-investigation") == "research"
          and est.norm_type("backend_seed") == "implement" and est.norm_type("handoff") == "docs" and est.norm_type("调研") == "research")
    check("类型：受控词表原样、长串未知归 other", est.norm_type("deliver") == "deliver" and est.norm_type("proto-data-copy-teachers-x") == "other")
    # 2 伪迹降权
    w = est.sample_weight
    check("权重：微型自动计时 0、显式不降", w(dict(actual_sec=10)) == 0 and w(dict(actual_sec=10, timing="explicit")) == 1)
    check("权重：疑似空闲 0、写明原因则保留", w(dict(actual_sec=4000, pred_sec=300, range_sec=[200, 500])) == 0
          and w(dict(actual_sec=4000, pred_sec=300, range_sec=[200, 500], cause="bug")) == 1)
    check("权重：wait 0、scope 半权、split 半权", w(dict(actual_sec=600, cause="wait")) == 0
          and w(dict(actual_sec=600, cause="scope")) == 0.5 and w(dict(actual_sec=600, timing="split")) == 0.5)

    with tempfile.TemporaryDirectory() as tmp:
        st = str(Path(tmp) / "s")
        seed_history(st)
        steps = '[{"name":"甲","type":"code","base_min":10},{"name":"乙","type":"test","base_min":5},{"name":"丙","type":"docs","base_min":3}]'
        t0 = time.time()
        rc, out = run(["plan", "--session", "a", "--task", "T", "--model", "gemini-4/high", "--steps", steps, "--verbose"], st, T0)
        check("plan 成功且 < 2 秒（含进程启动）", rc == 0 and time.time() - t0 < 2.0, f"{time.time() - t0:.2f}s")
        state = json.loads(next((Path(st) / "tasks").glob("*.json")).read_text(encoding="utf-8"))
        s1 = state["steps"][0]
        check("plan：类型归一并保留原文 label", s1["type"] == "implement" and s1.get("label") == "code")
        check("plan：型号存族名与原文", state["model"] == "gemini" and state["model_raw"] == "gemini-4/high")
        check("影子 v4：已给出预测且学到 gemini ≈ base×2", s1.get("pred_v4") and 1.5 < s1["pred_v4"] / 600 < 2.7, str(s1.get("pred_v4")))
        check("每步记录预测时刻", all(x.get("forecast_at") == T0 for x in state["steps"]))
        # 3 等待扣除：甲开工 0→等待 600 秒→解除→再做 300 秒
        run(["progress", "--session", "a", "--task", "T", "--block", "确认方案"], st, T0 + 300)
        run(["progress", "--session", "a", "--task", "T", "--block"], st, T0 + 900)
        rc, out = run(["progress", "--session", "a", "--task", "T", "--done", "s1", "--evidence", "x" * 100, "--cause", "bug"], st, T0 + 1200)
        h = [r for r in history(st) if r.get("session") == "a"]
        r1 = h[-1]
        check("等待期不计入步骤用时（1200−600=600）", abs(r1["actual_sec"] - 600) < 1, str(r1["actual_sec"]))
        check("样本带 paused_sec、cause、timing、pred_v4", r1.get("paused_sec") == 600 and r1.get("cause") == "bug"
              and r1.get("timing") == "auto" and r1.get("pred_v4"))
        check("共享历史证据截断到 40 字符", len(r1.get("evidence", "")) <= est.EVIDENCE_MAX)
        check("样本存族名与原文型号", r1["model"] == "gemini" and r1.get("model_raw") == "gemini-4/high")
        # 4 一次完成两步：按预测比例拆分，不再各计整段
        rc, out = run(["progress", "--session", "a", "--task", "T", "--done", "s2,s3"], st, T0 + 2100)
        h = [r for r in history(st) if r.get("session") == "a" and r["kind"] == "step"]
        r2, r3 = h[-2], h[-1]
        check("拆分计时：两步合计 = 900 秒、标 split", abs(r2["actual_sec"] + r3["actual_sec"] - 900) < 2
              and r2["timing"] == r3["timing"] == "split", f"{r2['actual_sec']:.0f}+{r3['actual_sec']:.0f}")
        check("拆分按预测比例（乙 > 丙）", r2["actual_sec"] > r3["actual_sec"])
        rc, out = run(["deliver", "--session", "a", "--task", "T"], st, T0 + 2200)
        trow = history(st)[-1]
        check("任务汇总扣除等待并标 delivered", trow["kind"] == "task" and trow.get("status") == "delivered"
              and abs(trow["actual_sec"] - 1600) < 2 and trow.get("paused_sec") == 600, str(trow.get("actual_sec")))
        # 5 重划基线：已开工步骤保留原预测时刻 → 样本仍带预测，不被丢弃
        run(["plan", "--session", "b", "--task", "R", "--model", "claude/high", "--steps", '[{"name":"甲","type":"implement","base_min":5}]'], st, T0 + 3000)
        run(["plan", "--session", "b", "--task", "R", "--model", "claude/high", "--steps",
             '[{"name":"甲","type":"implement","base_min":5},{"name":"乙","type":"verify","base_min":5}]'], st, T0 + 3100)
        run(["progress", "--session", "b", "--task", "R", "--done", "s1"], st, T0 + 3400)
        steps_l, _, info = est.load_history(Path(st) / "history.jsonl", T0 + 1e6)
        rr = [r for r in steps_l if r["id"] == "b/R/s1"]
        check("重划基线后完成的步骤仍有效且保留预测", rr and rr[0].get("pred_sec") and info["invalid_rows"] == 0, str(info))
        # 6 放弃与自动收口
        rc, out = run(["abandon", "--session", "b", "--task", "R"], st, T0 + 3500)
        ab = history(st)[-1]
        check("abandon：写放弃汇总（独立 id）", rc == 0 and ab.get("status") == "abandoned" and ab["id"].endswith("#abandoned"))
        run(["plan", "--session", "c", "--task", "S", "--model", "claude/high", "--steps", '[{"name":"甲","type":"implement","base_min":5}]'], st, T0 + 4000)
        rc, out = run(["plan", "--session", "d", "--task", "U", "--model", "claude/high", "--steps", '[{"name":"甲","type":"implement","base_min":5}]'],
                      st, T0 + 4000 + est.STALE_SEC + 10)
        check("超过 24 小时无事件的任务被自动放弃", any(w.startswith("auto_abandoned:1") for w in out.get("warn", [])), str(out.get("warn")))
        _, tasks_l, _ = est.load_history(Path(st) / "history.jsonl", T0 + 1e6)
        check("放弃任务不进任务级偏移", est.tau2_from([t for t in tasks_l if t.get("status") == "abandoned"], 0.5) == 0.25)
        rc, out = run(["plan", "--session", "e", "--task", "V", "--model", "claude/high", "--steps", '[{"name":"甲","type":"implement","base_min":5}]'],
                      st, T0 + 4000 + est.STALE_SEC + est.GC_SEC + 100)
        check("交付/放弃满 30 天的状态文件被清理", any(w.startswith("gc_task_files:") for w in out.get("warn", [])), str(out.get("warn")))
        # 7 估计器开关
        rc, out = run(["plan", "--session", "f", "--task", "W", "--model", "gemini-4/high", "--steps", '[{"name":"甲","type":"implement","base_min":10}]', "--verbose"],
                      st, T0 + 9e6, {"COMPLETION_PROGRESS_ESTIMATOR": "v4"})
        check("COMPLETION_PROGRESS_ESTIMATOR=v4 时由 v4 出数", rc == 0 and out["steps"][0]["src"] == "v4", str(out.get("steps")))
        rc, out = run(["plan", "--session", "g", "--task", "X", "--steps", '[{"name":"甲","type":"implement","base_min":10}]'],
                      st, T0 + 9e6, {"COMPLETION_PROGRESS_MODEL": "claude-opus-5/high"})
        state = json.loads(est.task_path(Path(st), "g", "X").read_text(encoding="utf-8"))
        check("--model 缺省读环境变量", rc == 0 and state["model"] == "claude")
        rc, out = run(["plan", "--session", "h2", "--task", "Y", "--steps", '[{"name":"甲","type":"implement","base_min":10}]'], st, T0 + 9e6)
        check("型号未知时提示", any(w.startswith("model_unknown") for w in out.get("warn", [])))
        # 8 audit 含数据质量与影子对照，且很快
        t0 = time.time()
        rc, out = run(["audit", "--compact"], st, T0 + 9e6)
        check("audit < 5 秒且含 data_quality / base_bias_by_family / shadow", rc == 0 and time.time() - t0 < 5
              and "data_quality" in out and "gemini" in out.get("base_bias_by_family", {}) and "shadow" in out,
              f"{time.time() - t0:.1f}s keys={list(out)[:3]}")
        # 9 中短任务过程汇报、秒级交付显示与 τ² 离群过滤
        rc, p_out = run(["plan", "--session", "short", "--task", "push", "--model", "claude",
                         "--steps", '[{"name":"检查","type":"verify","base_min":1},{"name":"推送","type":"deliver","base_min":1}]'], st, T0 + 9.5e6)
        rc, d1_out = run(["progress", "--session", "short", "--task", "push", "--done", "s1"], st, T0 + 9.5e6 + 18)
        check("中短任务完成步骤（18s）仍主动汇报进度", d1_out["should_report"] is True and "已完成 1/2 步" in d1_out["line"])
        run(["progress", "--session", "short", "--task", "push", "--done", "s2"], st, T0 + 9.5e6 + 32)
        rc, del_out = run(["deliver", "--session", "short", "--task", "push"], st, T0 + 9.5e6 + 32)
        check("秒级完成步骤在 deliver 中显示具体秒数而非刚开始",
              "18 秒" in del_out["line"] and "14 秒" in del_out["line"] and "刚开始" not in del_out["line"], del_out["line"])
        noisy_tasks = [dict(resid_k=3, resid_mean=0.1), dict(resid_k=3, resid_mean=-0.1), dict(resid_k=3, resid_mean=0.05),
                       dict(resid_k=2, resid_mean=-4.2)]
        check("τ² 过滤 |resid_mean|>1.5 的微步污染任务", est.tau2_from(noisy_tasks, 0.6) < 0.25)
    print(f"\n{len(FAILS)} FAILED: {FAILS}" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
