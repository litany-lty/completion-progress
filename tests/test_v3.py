#!/usr/bin/env python3
"""v3 验证：合成数据参数恢复、序贯覆盖率、四维不同仍可估、冷启动、θ 收敛、退出码、文案。
只在临时目录里写；运行: python3 tests/test_v3.py
"""
from __future__ import annotations
import importlib.util, json, math, os, random, re, statistics, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("est", HERE / "estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)
PY = sys.executable
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def run(args, state, at=None, env_extra=None):
    env = dict(os.environ, COMPLETION_PROGRESS_STATE=state, TZ="Asia/Shanghai")
    env.pop("COMPLETION_PROGRESS_TZ", None)
    if env_extra is not None:
        for k, v in env_extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    timed = args[0] in ("plan", "progress", "status", "deliver", "audit")
    cmd = [PY, str(HERE / "estimate.py")] + args + (["--verbose"] if timed else []) \
        + (["--at", str(at)] if at is not None and timed else [])
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        out = json.loads(p.stdout) if p.stdout.strip() else {}
    except json.JSONDecodeError:
        out = {"_raw": p.stdout}
    return p.returncode, out, p.stderr


def synth(n, seed, models=("A/high", "B/medium"), types=("research|file", "impl|step", "test|case")):
    """ln T = μ_t + β_t ln Q + γ_t (d−3) + δ_m + 任务偏移 + 噪声"""
    rng = random.Random(seed)
    mu = {"research|file": math.log(40), "impl|step": math.log(600), "test|case": math.log(15)}
    beta = {"research|file": 0.9, "impl|step": 1.0, "test|case": 0.7}
    gam = {"research|file": 0.3, "impl|step": 0.5, "test|case": 0.2}
    delta = {"A/high": 0.0, "B/medium": -0.4}
    rows, t = [], 1_700_000_000.0
    for task in range(n // 3):
        theta = rng.gauss(0, 0.2)
        m = rng.choice(models)
        for s in range(3):
            tk = types[s]
            typ, u = tk.split("|")
            q = 1.0 if typ == "impl" else float(rng.choice([5, 10, 20, 40, 80]))
            d = rng.choice(est.LEVELS)
            y = mu[tk] + beta[tk] * math.log(q) + gam[tk] * (est.LEVELS.index(d) - 2) + delta[m] + theta + rng.gauss(0, 0.3)
            T = math.exp(y)
            rows.append(dict(v=3, kind="step", id=f"s/T{task}/s{s}", session="s", task=f"T{task}", type=typ, u=u, d=d, q=q,
                             model=m, actual_sec=T, started_at=t, completed_at=t + T))
            t += T + 10
    return rows, dict(mu=mu, beta=beta, gam=gam, delta=delta)


def main():
    # 1 参数恢复
    rows, truth = synth(300, 1)
    m = est.Model(rows, 1.0)
    b_imp = m.coef("beta0", 1.0) + m.coef("beta:test|case", 0.0)
    d_B = m.coef("mod:B/medium", 0.0) - m.coef("mod:A/high", 0.0)
    g_imp = m.coef("gam0", 0.0) + m.coef("gam:impl|step", 0.0)
    check("参数恢复 β_test≈0.7", abs(b_imp - 0.7) < 0.1, f"{b_imp:.3f}")
    check("参数恢复 δ_B−δ_A≈−0.4", abs(d_B + 0.4) < 0.1, f"{d_B:.3f}")
    check("参数恢复 γ_impl≈0.5", abs(g_imp - 0.5) < 0.1, f"{g_imp:.3f}")
    check("σ 估计≈sqrt(0.3²+0.2²)=0.36", abs(math.sqrt(m.s2) - 0.36) < 0.06, f"{math.sqrt(m.s2):.3f}")

    # 2 序贯回测覆盖率（只用更早样本）
    cov = n = 0
    errs = []
    for r in rows[60:300:3]:
        past = [p for p in rows if p["completed_at"] <= r["started_at"]]
        mm = est.Model(past, 1.0, extra=[r])
        mu_, var = mm.predict(r)
        half = est.t80(mm.nu) * math.sqrt(var)
        y = math.log(r["actual_sec"])
        cov += mu_ - half <= y <= mu_ + half
        errs.append(abs(mu_ - y))
        n += 1
    check("序贯 P20–P80 覆盖率在 50–70%", 0.5 <= cov / n <= 0.7, f"{cov / n:.0%} n={n}")
    check("序贯对数 MAE < 0.35", sum(errs) / n < 0.35, f"{sum(errs) / n:.3f}")

    # 3 小样本：1 条样本后同类型预测贴近该样本、区间宽
    one = [rows[0]]
    m1 = est.Model(one, 1.0, extra=[dict(rows[0], q=rows[0]["q"] * 2)])
    mu1, var1 = m1.predict(rows[0])
    check("1 条样本：预测≈样本", abs(mu1 - math.log(rows[0]["actual_sec"])) < 0.2, f"{mu1 - math.log(rows[0]['actual_sec']):.3f}")
    check("1 条样本：区间宽（sd>0.3）", math.sqrt(var1) > 0.3, f"{math.sqrt(var1):.3f}")

    # 4 CLI：零历史 + base 立即可估；q/d/model 变化仍可估
    with tempfile.TemporaryDirectory() as tmp:
        T0 = 1760702400.0
        steps = '[{"name":"调研","type":"research","q":12,"u":"file","d":"D2","base_min":6},{"name":"实现","type":"impl","d":"D4","base_min":15}]'
        rc, out, err = run(["plan", "--session", "c", "--task", "T1", "--model", "A/high", "--steps", steps], tmp, T0)
        check("零历史 + base：plan 立即给数", rc == 0 and out["status"] == "seed" and out["rem_min"][1] > 0, str(out.get("rem_min")))
        check("plan 文案含进度分析与时间栏", "【进度分析" in out["line"] and "已用：" in out["line"] and "预计：" in out["line"])
        rc, out, _ = run(["plan", "--session", "c", "--task", "T0", "--model", "A/high", "--steps", '[{"name":"调研","type":"research"}]'], tmp, T0)
        check("零历史无 base：诚实说完成这步后预估", out["status"] == "none" and "完成这步后预估" in out["line"])
        # 写 300 条合成历史，再 plan 一个 q/d/model 都没出现过的组合
        hist = Path(tmp) / "history.jsonl"
        with open(hist, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(est.canonical(r) + "\n")
        rc, out, _ = run(["plan", "--session", "c", "--task", "T2", "--model", "C/low", "--verbose",
                          "--steps", '[{"name":"调研","type":"research","q":33,"u":"file","d":"D4"}]'], tmp, T0 + 1e9)
        s = out["steps"][0]
        check("q=33/D4/新模型 C 仍可估（2.0 会 insufficient）", out["status"] == "estimated" and s["src"] == "model", f"{s['pred_min']} {s['range_min']}")
        exp = math.exp(truth["mu"]["research|file"] + 0.9 * math.log(33) + 0.3) / 60
        check("新模型预测落在 A/B 之间附近", 0.5 * exp < s["pred_min"] < 1.5 * exp, f"pred={s['pred_min']} truth(A)={exp:.1f}")
        # 5 θ 收敛：τ²=σ²/2 时 k 步修正 k/(k+2)
        rc, out, _ = run(["plan", "--session", "c", "--task", "T3", "--model", "A/high", "--verbose",
                          "--steps", '[{"name":"a","type":"impl","d":"D3"},{"name":"b","type":"impl","d":"D3"},{"name":"c","type":"impl","d":"D3"},{"name":"d","type":"impl","d":"D3"}]'], tmp, T0 + 2e9)
        p = out["steps"][0]["pred_min"] * 60
        t = T0 + 2e9
        facts = []
        for i, sid in enumerate(("s1", "s2", "s3")):
            t += p * 1.5
            rc, out, _ = run(["progress", "--session", "c", "--task", "T3", "--done", sid, "--actual", str(p * 1.5), "--verbose"], tmp, t)
            facts.append(out["theta"]["factor"])
        expect = [1.5 ** (k / (k + 1)) for k in (1, 2, 3)]   # τ² = σ²/2 = 步骤内方差 → k/(k+1)
        check("θ 修正按 k/(k+1) 收敛", all(abs(a - b) < 0.03 for a, b in zip(facts, expect)), f"{[round(x, 3) for x in facts]} vs {[round(x, 3) for x in expect]}")
        # 6 批量速率 & 不可分超时
        rc, out, _ = run(["plan", "--session", "c", "--task", "T4", "--model", "A/high", "--verbose",
                          "--steps", '[{"name":"跑","type":"test","q":40,"u":"case","d":"D3"}]'], tmp, T0 + 3e9)
        p = out["steps"][0]["pred_min"] * 60
        rc, out, _ = run(["progress", "--session", "c", "--task", "T4", "--running", "s1", "--units", "20"], tmp, T0 + 3e9 + p)
        check("批量 20/40 用了整段预计：剩余≈再一段", 0.6 * p < out["rem_min"][1] * 60 < 1.4 * p and out["step_method"] == "rate", f"{out['rem_min']} p={p / 60:.1f}")
        rc, out, _ = run(["plan", "--session", "c", "--task", "T4b", "--model", "A/high", "--verbose",
                          "--steps", '[{"name":"合并","type":"impl","d":"D3"}]'], tmp, T0 + 3e9)
        p = out["steps"][0]["pred_min"] * 60
        rc, out, _ = run(["status", "--session", "c", "--task", "T4b"], tmp, T0 + 3e9 + 5 * p)
        check("不可分超出 5× 预计：不显示 0，标重估", out["rem_min"][1] > 0 and "重估" in out["line"] and out["step_method"] == "overrun", out["line"][-70:])
        # 7 汇报节奏：刚报过 2 分钟内无变化不再报；--ask 总是给
        rc, out, _ = run(["plan", "--session", "c", "--task", "T5", "--model", "A/high",
                          "--steps", '[{"name":"a","type":"impl","d":"D3","base_min":30},{"name":"b","type":"impl","d":"D3","base_min":30}]'], tmp, T0 + 4e9)
        rc, out, _ = run(["progress", "--session", "c", "--task", "T5"], tmp, T0 + 4e9 + 60)
        check("1 分钟后无事件：不主动报", out["should_report"] is False)
        rc, out, _ = run(["status", "--session", "c", "--task", "T5"], tmp, T0 + 4e9 + 60)
        check("用户问：总是给", out["should_report"] is True and "【进度汇报" in out["line"])
        # 8 deliver 拒绝未完成；完成后写任务行
        rc, out, err = run(["deliver", "--session", "c", "--task", "T5"], tmp, T0 + 4e9 + 100)
        check("未完成 deliver 退出码 2", rc == 2 and "unfinished" in err)
        run(["progress", "--session", "c", "--task", "T5", "--done", "s1,s2"], tmp, T0 + 4e9 + 3600)
        rc, out, _ = run(["deliver", "--session", "c", "--task", "T5"], tmp, T0 + 4e9 + 3700)
        check("deliver 文案含各步用时与开工预计", rc == 0 and "各步：" in out["line"] and "开工时预计" in out["line"] and "用时：" in out["line"])
        lines = hist.read_text(encoding="utf-8").splitlines()
        check("history 含 task 行", any('"kind":"task"' in l for l in lines))
        # 9 只读目录：plan 仍返回 0 并给文案
        ro = Path(tmp) / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)
        rc, out, _ = run(["plan", "--session", "c", "--task", "T6", "--model", "A/high", "--steps", steps], str(ro), T0)
        os.chmod(ro, 0o700)
        check("状态目录只读：仍给文案，warn 标记", rc == 0 and "【进度分析" in out["line"] and any("not_persisted" in w for w in out["warn"]), str(out.get("warn")))
        # 10 audit 跑通并选 κ
        rc, out, _ = run(["audit", "--compact"], tmp, T0 + 5e9)
        check("audit：κ 由 CV 选出，回测覆盖在 45–75%", rc == 0 and out["kappa_src"] == "cv" and 0.45 <= (out["backtest"]["coverage_p20_p80"] or 0) <= 0.75,
              f"kappa={out.get('kappa')} bt={out.get('backtest')}")
        check("audit：模型速度表 B 慢于 A 约 e^-0.4", abs(out["model_speed"]["b"]["factor"] / out["model_speed"]["a"]["factor"] - math.exp(-0.4)) < 0.08,
              str({k: v["factor"] for k, v in out["model_speed"].items()}))
        # 11 并行槽位：a(10) ∥ b(20) → c(5)
        par = '[{"id":"a","name":"抓取","type":"fetch","d":"D3","base_min":10,"dep":[]},{"id":"b","name":"解析","type":"parse","d":"D3","base_min":20,"dep":[]},{"id":"c","name":"汇总","type":"merge","d":"D3","base_min":5,"dep":["a","b"]}]'
        rc, o2, _ = run(["plan", "--session", "c", "--task", "P2", "--model", "A/high", "--slots", "2", "--steps", par], tmp, T0)
        rc, o1, _ = run(["plan", "--session", "c", "--task", "P1", "--model", "A/high", "--slots", "1", "--steps", par], tmp, T0)
        check("并行 2 槽 ≈25、单槽 ≈35 分钟（P50，含右偏）", 24 <= o2["rem_min"][1] <= 31 and 34 <= o1["rem_min"][1] <= 42 and o1["rem_min"][1] > o2["rem_min"][1] + 7,
              f"slots2={o2['rem_min'][1]} slots1={o1['rem_min'][1]}")
        # 12 文案：时长与时刻同源取整
        rc, out, _ = run(["plan", "--session", "c", "--task", "Z1", "--model", "A/high",
                          "--steps", '[{"name":"a","type":"impl","d":"D3","base_min":33},{"name":"b","type":"impl","d":"D3","base_min":47}]'], tmp, T0)
        m = re.search(r"还剩约 (\d+)–(\d+) 分钟，(\d\d):(\d\d)–(\d\d):(\d\d)", out["line"])
        if m:
            lo_d, hi_d = int(m.group(1)), int(m.group(2))
            base_min = int(time.localtime(T0).tm_hour) * 60 + int(time.localtime(T0).tm_min)
            lo_c = (int(m.group(3)) * 60 + int(m.group(4))) - base_min
            hi_c = (int(m.group(5)) * 60 + int(m.group(6))) - base_min
            check("文案时长与时刻同源取整", abs(lo_c - lo_d) <= 2 and abs(hi_c - hi_d) <= 2, f"时长 {lo_d}–{hi_d} vs 时刻 {lo_c}–{hi_c}")
        else:
            check("文案时长与时刻同源取整", "至少" in out["line"], out["line"])
        # 13 错误原因进 stdout，搭车 2>/dev/null 仍可见
        p = subprocess.run([PY, str(HERE / "estimate.py"), "progress", "--session", "c", "--task", "NOPE",
                            "--state", str(tmp), "--at", str(T0)], capture_output=True, text=True,
                           env=dict(os.environ, TZ="Asia/Shanghai"))
        try:
            body = json.loads(p.stdout)
        except json.JSONDecodeError:
            body = {}
        check("任务不存在：stdout 有 ok=false 与 error，退出码 2", p.returncode == 2 and body.get("ok") is False and bool(body.get("error")), p.stdout.strip()[:80])
        # 14 无法估计的步骤：按其他步骤中位数计权重，不按 60 秒；文案注明不含
        rc, out, _ = run(["plan", "--session", "c", "--task", "Z2", "--model", "A/high",
                          "--steps", '[{"name":"甲","type":"impl","d":"D3","base_min":30},{"name":"乙","type":"mystery-unseen"}]'], tmp, T0)
        s_unknown = [s for s in out["steps"] if s["src"] == "none"]
        known = [s["pred_min"] for s in out["steps"] if s["src"] != "none"]
        check("无估计步骤：权重取其他步骤中位数、ETA 注明不含",
              s_unknown and known and abs(s_unknown[0]["pred_min"] - statistics.median(known)) < 0.1 and "不含「乙」" in out["line"], out["line"][-50:])
        # 15 汇报节奏四组合
        st = dict(unit_min=5, blocked=None)
        now_ = 1_000_000.0
        st["last_report"] = dict(t=now_ - 60, rem=1200.0)
        check("变化大：冷却期内也报", est.decide_report(st, now_, 1200.0 - 60 + 400, "status")[0] is True)
        check("无变化 + 冷却期内：不报", est.decide_report(st, now_, 1140.0, "status")[0] is False)
        check("尾部 <2 分钟：不报", est.decide_report(dict(st, last_report=dict(t=now_ - 600, rem=300.0)), now_, 90.0, "status")[0] is False)
        check("block 事件：总是报", est.decide_report(st, now_, 1140.0, "block")[0] is True)
        # 16 TZ 完全未声明且系统为 UTC：不报绝对时刻
        rc, out, _ = run(["plan", "--session", "c", "--task", "Z3", "--model", "A/high",
                          "--steps", '[{"name":"甲","type":"impl","d":"D3","base_min":30}]'], tmp, T0,
                         env_extra={"TZ": None, "COMPLETION_PROGRESS_TZ": None})
        check("TZ 不明：只报时长不报时刻", not re.search(r"\d\d:\d\d", out["line"]), out["line"][-40:])
        # 17 默认输出瘦身（不加 --verbose）
        p = subprocess.run([PY, str(HERE / "estimate.py"), "status", "--session", "c", "--task", "Z3",
                            "--state", str(tmp), "--at", str(T0)], capture_output=True, text=True,
                           env=dict(os.environ, TZ="Asia/Shanghai"))
        keys = set(json.loads(p.stdout)) if p.stdout.strip() else set()
        check("默认输出只含 5 个字段", keys == {"ok", "ev", "should_report", "line", "warn"}, str(sorted(keys)))
        check("默认输出 ≤250 字节", len(p.stdout) <= 250, f"{len(p.stdout)}B")
        # 18 一次调用完成：完成上一步 + 批量进度
        rc, out, _ = run(["plan", "--session", "c", "--task", "Z4", "--model", "A/high",
                          "--steps", '[{"name":"甲","type":"impl","d":"D3","base_min":10},{"name":"乙","type":"test","q":40,"u":"case","d":"D3","base_min":20}]'], tmp, T0)
        rc, out, _ = run(["progress", "--session", "c", "--task", "Z4", "--done", "s1", "--running", "s2", "--units", "10"], tmp, T0 + 600)
        check("--done 与 --running --units 一次调用生效", rc == 0 and out["steps"][0]["status"] == "done" and out["steps"][1].get("status") == "running", str([s["status"] for s in out["steps"]]))
        # 19 并行槽位：a(10) ∥ b(20) → c(5)
        par = '[{"id":"a","name":"抓取","type":"fetch","d":"D3","base_min":10,"dep":[]},{"id":"b","name":"解析","type":"parse","d":"D3","base_min":20,"dep":[]},{"id":"c","name":"汇总","type":"merge","d":"D3","base_min":5,"dep":["a","b"]}]'
        rc, o2, _ = run(["plan", "--session", "c", "--task", "P2", "--model", "A/high", "--slots", "2", "--steps", par], tmp, T0)
        rc, o1, _ = run(["plan", "--session", "c", "--task", "P1", "--model", "A/high", "--slots", "1", "--steps", par], tmp, T0)
        check("并行 2 槽 ≈25、单槽 ≈35 分钟（P50，含右偏）", 24 <= o2["rem_min"][1] <= 31 and 34 <= o1["rem_min"][1] <= 42 and o1["rem_min"][1] > o2["rem_min"][1] + 7,
              f"slots2={o2['rem_min'][1]} slots1={o1['rem_min'][1]}")
        # 20 Huber：5% 十倍慢样本不拉偏 β 与 σ
        noisy = [dict(r) for r in rows]
        rng = random.Random(9)
        for i in rng.sample(range(len(noisy)), len(noisy) // 20):
            noisy[i]["actual_sec"] *= 10
        mh = est.Model(noisy, 1.0)
        b_h = mh.coef("beta0", 1.0) + mh.coef("beta:test|case", 0.0)
        check("Huber：β 不被离群拉偏", abs(b_h - 0.7) < 0.12, f"{b_h:.3f}")
        check("Huber：σ 不被离群吹大", math.sqrt(mh.s2) < 0.55, f"{math.sqrt(mh.s2):.3f}")
        check("Huber：确实降权了离群样本", mh.downweighted >= 8, str(mh.downweighted))
        # 21 开工估计按类型校准
        brows = []
        for i in range(12):
            brows.append(dict(v=3, kind="step", id=f"b/A/s{i}", session="b", task=f"A{i}", type="research", u="file", d="D3",
                              q=10, model="M/h", actual_sec=200, base_sec=100, started_at=i, completed_at=i + 200))
            brows.append(dict(v=3, kind="step", id=f"b/B/s{i}", session="b", task=f"B{i}", type="impl", u="step", d="D3",
                              q=1, model="M/h", actual_sec=100, base_sec=100, started_at=i, completed_at=i + 100))
        bs = est.base_stats(brows, 0.16)
        check("开工估计按类型校准：调研偏 2×、实现不偏",
              abs(bs["by_type"]["research|file"]["bias"] - math.log(2)) < 0.05 and abs(bs["by_type"]["impl|step"]["bias"]) < 0.05,
              f"{bs['by_type']['research|file']['bias']:.2f}/{bs['by_type']['impl|step']['bias']:.2f}")
        # 22 τ² 经验贝叶斯：任务少时贴近先验 σ²/2，不被 3 个任务带跑
        t3 = est.tau2_from([dict(resid_k=5, resid_mean=x) for x in (0.8, -0.8, 0.6)], 1.0)
        check("τ² 收缩：3 个任务时介于先验与经验值之间", 0.05 <= t3 <= 0.9 and abs(t3 - 0.5) < 0.45, f"{t3:.2f}")
        # 23 complexity drift 监测：D5 近期系统性变慢（判定标准变松）时 flag
        drift = [dict(r) for r in sorted(rows, key=lambda r: r["completed_at"])]
        d5 = [r for r in drift if r["d"] == "D5"]
        for r in d5[int(len(d5) * 0.7):]:
            r["actual_sec"] *= 2.2
        dd = Path(tmp) / "driftdir"
        dd.mkdir()
        with open(dd / "history.jsonl", "w", encoding="utf-8") as f:
            for r in drift:
                f.write(est.canonical(r) + "\n")
        rc, out, _ = run(["audit", "--compact", "--state", str(dd)], tmp, T0 + 6e9)
        d5r = out.get("complexity_residuals", {}).get("D5", {})
        check("复杂度漂移监测：D5 近期变慢被标记", d5r.get("flag") is True and (d5r.get("drift") or 0) > 0.3, str(d5r))
        others = [v for k, v in out.get("complexity_residuals", {}).items() if k != "D5" and v.get("flag")]
        check("其他档不误报", not others, str(others))
        # 24 单位升级：剩余越过 90 分钟时当次文案提示改用半小时
        rc, out, _ = run(["plan", "--session", "c", "--task", "U1", "--model", "A/high", "--steps",
                          '[{"name":"甲","type":"impl","d":"D3","base_min":20},{"name":"乙","type":"impl","d":"D3","base_min":20},'
                          '{"name":"丙","type":"impl","d":"D3","base_min":20}]'], tmp, T0)
        rc, out, _ = run(["progress", "--session", "c", "--task", "U1", "--done", "s1", "--actual", "6000"], tmp, T0 + 6000)
        check("单位升级：文案提示改按半小时估", "改按半小时估" in out["line"], out["line"][-40:])
        # 25 type×model 交互：只对证据达标的组合启用（≥8 条且 |均值| > max(0.2, 2·标准误)）
        rng2 = random.Random(11)
        rows_ix = []
        t = 1_760_000_000.0
        for i in range(60):
            for typ, u, base_s, q in (("research", "file", 200.0, 10.0), ("impl", "step", 600.0, 1.0)):
                mm = "A/high" if i % 2 == 0 else "B/medium"
                extra = 0.6 if (typ == "impl" and mm == "B/medium") else 0.0
                T = base_s * math.exp(extra + rng2.gauss(0, 0.3))
                rows_ix.append(dict(v=3, kind="step", id=f"x/T{i}/{typ}", session="x", task=f"T{i}", type=typ, u=u,
                                    d="D3", q=q, model=mm, actual_sec=T, started_at=t, completed_at=t + T))
                t += T + 5
            if i < 6:
                T = 100.0 * math.exp(rng2.gauss(0, 0.3))
                rows_ix.append(dict(v=3, kind="step", id=f"x/T{i}/rare", session="x", task=f"T{i}", type="rare", u="step",
                                    d="D3", q=1, model="B/medium", actual_sec=T, started_at=t, completed_at=t + T))
                t += T + 5
        di = Path(tmp) / "ixdir"
        di.mkdir()
        with open(di / "history.jsonl", "w", encoding="utf-8") as f:
            for r in rows_ix:
                f.write(est.canonical(r) + "\n")
        rc, out, _ = run(["audit", "--compact", "--state", str(di)], tmp, T0 + 7e9)
        enabled = out["interactions"]["enabled"]
        check("交互：impl×B 所在单元被整组启用，s² 降幅 ≥10%", "implement|step|b" in enabled
              and out["interactions"]["test"]["s2_ratio"] < 0.9, str(out["interactions"]["test"]))
        check("交互：样本 <20 的 rare 组合不参与判定", "rare|step|b" not in out["interactions"]["candidates"])
        rc, o_ix, _ = run(["plan", "--session", "x", "--task", "P", "--model", "B/medium", "--state", str(di),
                           "--steps", '[{"name":"实现","type":"impl","d":"D3"}]'], tmp, T0 + 7e9)
        rc, o_ix_a, _ = run(["plan", "--session", "x", "--task", "PA", "--model", "A/high", "--state", str(di),
                             "--steps", '[{"name":"实现","type":"impl","d":"D3"}]'], tmp, T0 + 7e9)
        os.remove(di / "model.json")
        rc, o_no, _ = run(["plan", "--session", "x2", "--task", "P", "--model", "B/medium", "--state", str(di),
                           "--steps", '[{"name":"实现","type":"impl","d":"D3"}]'], tmp, T0 + 7e9)
        rc, o_no_a, _ = run(["plan", "--session", "x2", "--task", "PA", "--model", "A/high", "--state", str(di),
                             "--steps", '[{"name":"实现","type":"impl","d":"D3"}]'], tmp, T0 + 7e9)
        # 真值：impl/A≈10 分钟、impl/B≈e^0.6×10≈18 分钟；交互应让两者都更接近
        truth_a, truth_b = 10.0, 10.0 * math.exp(0.6)
        err_ix = abs(o_ix_a["steps"][0]["pred_min"] - truth_a) + abs(o_ix["steps"][0]["pred_min"] - truth_b)
        err_no = abs(o_no_a["steps"][0]["pred_min"] - truth_a) + abs(o_no["steps"][0]["pred_min"] - truth_b)
        check("交互：启用后 impl 两种型号的误差都变小", err_ix < err_no * 0.8,
              f"ix={err_ix:.1f} no={err_no:.1f} (A {o_ix_a['steps'][0]['pred_min']}/{o_no_a['steps'][0]['pred_min']}, "
              f"B {o_ix['steps'][0]['pred_min']}/{o_no['steps'][0]['pred_min']})")
        # 负对照：型号间无真实差异时不应启用交互
        rng3 = random.Random(5)
        flat = []
        t = 1_760_000_000.0
        for i in range(60):
            for typ, u, base_s, q in (("research", "file", 200.0, 10.0), ("impl", "step", 600.0, 1.0)):
                mm = "A/high" if i % 2 == 0 else "B/medium"
                T = base_s * math.exp(rng3.gauss(0, 0.3))
                flat.append(dict(v=3, kind="step", id=f"y/T{i}/{typ}", session="y", task=f"T{i}", type=typ, u=u,
                                 d="D3", q=q, model=mm, actual_sec=T, started_at=t, completed_at=t + T))
                t += T + 5
        df = Path(tmp) / "flatdir"
        df.mkdir()
        with open(df / "history.jsonl", "w", encoding="utf-8") as f:
            for r in flat:
                f.write(est.canonical(r) + "\n")
        rc, out, _ = run(["audit", "--compact", "--state", str(df)], tmp, T0 + 8e9)
        check("交互负对照：无真实差异时不启用", out["interactions"]["enabled"] == [], str(out["interactions"]["test"]))
        # 26 英文输出：标记、单位、无中文残留；默认仍中文
        rc, oen, _ = run(["plan", "--session", "en", "--task", "T", "--model", "A/high", "--lang", "en", "--tz", "Asia/Shanghai",
                          "--steps", '[{"name":"调研","type":"impl","d":"D3","base_min":12},{"name":"实现","type":"impl","d":"D3","base_min":20}]'], tmp, T0)
        rc, oen2, _ = run(["progress", "--session", "en", "--task", "T", "--lang", "en", "--tz", "Asia/Shanghai",
                           "--at", str(T0 + 600), "--done", "s1"], tmp, T0 + 600)
        rc, ozh, _ = run(["plan", "--session", "en", "--task", "T2", "--model", "A/high",
                          "--steps", '[{"name":"调研","type":"impl","d":"D3","base_min":12}]'], tmp, T0)
        check("英文：plan 用 [Plan …] 且无中文标签", oen["line"].startswith("> **[Plan:") and "【" not in oen["line"]
              and "正在" not in oen["line"] and "remaining" in oen["line"], oen["line"][:60])
        check("英文：progress 用 [Progress …] 与 elapsed/remaining", oen2["line"].startswith("> **[Progress:")
              and "elapsed" in oen2["line"] and "remaining" in oen2["line"] and "预计" not in oen2["line"], oen2["line"][:60])
        check("默认仍中文", "【进度分析" in ozh["line"] and "[Plan" not in ozh["line"])
        # 27 输入与状态边界：坏 JSON、重复完成、重复交付
        p = subprocess.run([PY, str(HERE / "estimate.py"), "plan", "--session", "c", "--task", "BAD", "--model", "A/high",
                            "--state", str(tmp), "--steps", "{not json"], capture_output=True, text=True,
                           env=dict(os.environ, TZ="Asia/Shanghai"))
        check("坏 JSON：退出码 2 且 stdout 有 error", p.returncode == 2 and '"ok": false' in p.stdout, p.stdout[:60])
        run(["plan", "--session", "c", "--task", "D1x", "--model", "A/high",
             "--steps", '[{"name":"a","type":"impl","d":"D3","base_min":5}]'], tmp, T0)
        run(["progress", "--session", "c", "--task", "D1x", "--done", "s1"], tmp, T0 + 300)
        rc, out, err = run(["progress", "--session", "c", "--task", "D1x", "--done", "s1"], tmp, T0 + 400)
        check("重复完成：拒绝且退出码 2", rc == 2 and "already done" in err, err[:60])
        rc, out, _ = run(["deliver", "--session", "c", "--task", "D1x"], tmp, T0 + 500)
        rc2, out2, err2 = run(["deliver", "--session", "c", "--task", "D1x"], tmp, T0 + 600)
        task_rows = [l for l in (Path(tmp) / "history.jsonl").read_text(encoding="utf-8").splitlines()
                     if '"kind":"task"' in l and '"id":"c/D1x"' in l]
        check("重复交付：拒绝且不重复写任务汇总", rc2 == 2 and "already delivered" in err2 and len(task_rows) == 1,
              f"rc={rc2} rows={len(task_rows)}")
        # 28 v2 → v3 迁移：难度映射与 model 提取
        v2 = Path(tmp) / "v2.jsonl"
        v2.write_text(json.dumps(dict(v=2, id="s1/T/s1", session="s1", task="T", type="research", env="m/h|bash|c1|warm",
                                      q=10, u="page", d="D2", actual_sec=100, started_at=1, completed_at=101,
                                      method="unestimated")) + "\n", encoding="utf-8")
        rc, out, _ = run(["migrate", str(v2), str(Path(tmp) / "v3.jsonl")], tmp, T0)
        row = json.loads((Path(tmp) / "v3.jsonl").read_text(encoding="utf-8").splitlines()[0])
        check("迁移：D2→D3、model 取 env 第一段、v=3", rc == 0 and row["d"] == "D3" and row["model"] == "m/h" and row["v"] == 3, str(row)[:80])
        # 29 外推：数量远超历史范围时加宽区间并给出 warn
        ex_dir = Path(tmp) / "exdir"
        ex_dir.mkdir()
        erng = random.Random(21)
        with open(ex_dir / "history.jsonl", "w", encoding="utf-8") as f:
            t = 1.76e9
            for i in range(30):
                T = 20.0 * 10 ** 0.9 * math.exp(erng.gauss(0, 0.25))
                f.write(est.canonical(dict(v=3, kind="step", id=f"e/T{i}/x", session="e", task=f"T{i}", type="batch",
                                           u="item", d="D3", q=10.0, model="A/high", actual_sec=T,
                                           started_at=t, completed_at=t + T)) + "\n")
                t += T + 1
        rc, o_in, _ = run(["plan", "--session", "e", "--task", "IN", "--model", "A/high", "--state", str(ex_dir),
                           "--steps", '[{"name":"批","type":"batch","q":10,"u":"item","d":"D3"}]'], tmp, T0)
        rc, o_out, _ = run(["plan", "--session", "e", "--task", "OUT", "--model", "A/high", "--state", str(ex_dir),
                            "--steps", '[{"name":"批","type":"batch","q":100,"u":"item","d":"D3"}]'], tmp, T0)
        w_in = o_in["steps"][0]["range_min"][1] / o_in["steps"][0]["range_min"][0]
        w_out = o_out["steps"][0]["range_min"][1] / o_out["steps"][0]["range_min"][0]
        check("外推：区间变宽且给出 warn", o_out["steps"][0]["extrap"] > 0 and w_out > w_in
              and any("extrapolated_steps" in w for w in o_out["warn"]), f"宽比 {w_in:.2f}→{w_out:.2f} extrap={o_out['steps'][0]['extrap']}")
        # 30 曲率诊断：固定开销被标记、纯幂律不标
        cv_dir = Path(tmp) / "cvdir"
        cv_dir.mkdir()
        cr = random.Random(7)
        with open(cv_dir / "history.jsonl", "w", encoding="utf-8") as f:
            t = 1.76e9
            for i in range(90):
                q = cr.choice([1, 2, 5, 10, 20, 40])
                T = (120 + 20 * q) * math.exp(cr.gauss(0, 0.25))
                f.write(est.canonical(dict(v=3, kind="step", id=f"c/T{i}/x", session="c", task=f"T{i}", type="batch",
                                           u="item", d="D3", q=float(q), model="A/high", actual_sec=T,
                                           started_at=t, completed_at=t + T)) + "\n")
                t += T + 1
        rc, oc, _ = run(["audit", "--compact", "--state", str(cv_dir)], tmp, T0 + 5e8)
        curv = oc.get("curvature", {}).get("batch|item", {})
        check("曲率诊断：固定开销数据被标记", curv.get("flag") is True and curv.get("corr", 0) > 0.4, str(curv))
        # 31 半小时时区
        est.resolve_tz("Asia/Kolkata")
        check("半小时时区标签", "5.5" in est.tz_label(), est.tz_label())
        est.resolve_tz("Asia/Shanghai")
        # 32 base 方差极小时权重仍有上限
        tiny = est.step_forecast(est.Model(rows, 1.0), dict(type="impl", u="step", q=1.0, d="D3", base_sec=600.0),
                                 "A/high", 100, dict(bias=0.0, var=1e-8, by_type={}, src="history"), 0.05)
        check("base 历史方差极小也不独断", abs(math.log(tiny["pred_sec"]) - math.log(600.0)) > 0.001,
              f"pred={tiny['pred_sec']:.0f}")
        # 33 非法 --actual / --units
        rc, _, err = run(["progress", "--session", "c", "--task", "D1x", "--actual", "-5"], tmp, T0 + 800)
        rc2, _, err2 = run(["progress", "--session", "c", "--task", "D1x", "--running", "s1", "--units", "-2"], tmp, T0 + 800)
        check("非法 --actual/--units 被拒绝", rc == 2 and rc2 == 2 and "positive" in err and "nonnegative" in err2,
              f"{err[:40]} / {err2[:40]}")
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
