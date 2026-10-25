#!/usr/bin/env python3
"""建模校准研究：在已知真值的合成数据上检验预测区间、偏差与稳健性。

运行：python3 studies/calibration.py
不读写真实状态；只用 estimate.py 的模型函数。输出为报告表格。
"""
from __future__ import annotations
import importlib.util
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("est", HERE / "estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)

TYPES = [("research", "file", 40.0, 0.9, 0.3), ("impl", "step", 600.0, 1.0, 0.5), ("test", "case", 15.0, 0.7, 0.2)]
MODELS = ["A/high", "B/medium"]


def gen_rows(n_tasks, seed, noise_sd=0.35, beta=None, gamma=None, delta=None, heavy_df=None, covariate_range=(1, 5)):
    """按真实模型生成步骤样本；可注入误设。"""
    rng = random.Random(seed)
    beta = beta or {f"{t}|{u}": b for t, u, _, b, _ in TYPES}
    gamma = gamma or {f"{t}|{u}": g for t, u, _, _, g in TYPES}
    delta = delta if delta is not None else {"A/high": 0.0, "B/medium": -0.25}
    base = {f"{t}|{u}": m for t, u, m, _, _ in TYPES}
    rows, t = [], 1.70e9
    for k in range(n_tasks):
        theta = rng.gauss(0, 0.15)
        m = rng.choice(MODELS)
        for typ, u, _, _, _ in TYPES:
            key = f"{typ}|{u}"
            q = float(rng.randint(*covariate_range))
            d = rng.choice(est.LEVELS)
            mu = math.log(base[key]) + beta[key] * math.log(q) + gamma[key] * (est.LEVELS.index(d) - 2) + delta[m] + theta
            if heavy_df:
                z = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(heavy_df / 2, 2 / heavy_df))
                y = mu + noise_sd * z
            else:
                y = mu + rng.gauss(0, noise_sd)
            T = math.exp(y)
            rows.append(dict(v=3, kind="step", id=f"s/T{k}/{typ}", session="s", task=f"T{k}", type=typ, u=u, d=d,
                             q=q, model=m, actual_sec=T, started_at=t, completed_at=t + T))
            t += T + 5
    rows.sort(key=lambda r: (r["completed_at"], r["id"]))
    return rows


def coverage(model, rows, limit=None):
    errs, hit, n = [], 0, 0
    for r in rows:
        mu, var = model.predict(r)
        half = est.t80(model.nu) * math.sqrt(var)
        y = math.log(r["actual_sec"])
        errs.append(mu - y)
        hit += mu - half <= y <= mu + half
        n += 1
    return dict(n=n, coverage=hit / n, bias=statistics.fmean(errs), log_mae=statistics.fmean(abs(e) for e in errs))


def e1_coverage_by_n():
    print("\nE1 模型假设内的区间覆盖（目标 ≈60%）")
    print(f"{'n':>6} {'覆盖':>7} {'偏差':>8} {'对数MAE':>8}")
    rows = gen_rows(120, 1, covariate_range=(1, 8))
    for n in (10, 30, 60, 120, 240):
        m = est.Model(rows[:n], 1.0)
        c = coverage(m, rows[:n])
        print(f"{n:>6} {c['coverage']:>6.0%} {c['bias']:>+8.3f} {c['log_mae']:>8.3f}")


def e2_fixed_cost_misspec():
    """真实过程 T = (a + b·Q)·ε，用 ln 线性模型拟合：看覆盖与预测偏差随 Q 的变化。"""
    print("\nE2 误设：真实为固定开销 a+bQ（对数模型看不到 a）")
    rng = random.Random(7)
    a, b = 120.0, 20.0
    rows, t = [], 1.70e9
    for k in range(90):
        q = rng.choice([1, 2, 5, 10, 20, 40])
        T = (a + b * q) * math.exp(rng.gauss(0, 0.25))
        rows.append(dict(v=3, kind="step", id=f"s/T{k}/x", session="s", task=f"T{k}", type="batch", u="item", d="D3",
                         q=float(q), model="A/high", actual_sec=T, started_at=t, completed_at=t + T))
        t += T + 1
    m = est.Model(rows, 1.0)
    print(f"  β 估计={m.coef('beta0', 1.0) + m.coef('beta:batch|item', 0.0):.2f}（对数模型的“平均”弹性，真值介于 0~1）")
    for q in (1, 5, 20, 40, 80):
        r = dict(type="batch", u="item", q=float(q), d="D3", model="A/high")
        mu, var = m.predict(r)
        half = est.t80(m.nu) * math.sqrt(var)
        truth = a + b * q
        print(f"  q={q:>3}: 预测={math.exp(mu):7.0f}s  真值={truth:7.0f}s  误差={math.exp(mu) / truth - 1:+6.0%}  "
              f"区间宽×{math.exp(half):.2f}")


def e3_heavy_tail():
    print("\nE3 厚尾（t df=3）+ 10% 十倍离群：Huber 与普通最小二乘（同一设计矩阵，仅权重不同）")
    rows = gen_rows(80, 3, noise_sd=0.35, heavy_df=3)
    rng = random.Random(9)
    for i in rng.sample(range(len(rows)), len(rows) // 10):
        rows[i]["actual_sec"] *= 10

    class LS(est.Model):
        """不做 IRLS：直接用权重 1 解一次。"""
        def __init__(self, rows_, kappa, extra=(), interact=()):
            self.interact = frozenset(interact)
            cols, seen = [], set()
            for r in list(rows_) + list(extra):
                for c in self._f(r):
                    if c not in seen:
                        seen.add(c)
                        cols.append(c)
            self.cols, self.idx, self.kappa, self.n = cols, {c: i for i, c in enumerate(cols)}, kappa, len(rows_)
            self._solve(rows_, [1.0] * len(rows_))
            self.downweighted = 0
            self.coef_id = "ls"
    mh = est.Model(rows, 1.0)
    m0 = LS(rows, 1.0)
    for name, m in (("普通LS", m0), ("Huber", mh)):
        b = m.coef("beta0", 1.0) + m.coef("beta:impl|step", 0.0)
        cov = coverage(m, rows)["coverage"]
        print(f"  {name:>6}: β_impl={b:.3f}（真 1.0） σ={math.sqrt(m.s2):.3f}（真约 0.36） 覆盖={cov:.0%} 降权={m.downweighted}")


def e4_confounded_q():
    """某类型只在窄 q 范围内出现：检验对同类型新 q 的外推可靠性。"""
    print("\nE4 混杂：research 只见过 q∈{1,2}，impl 只见过 q∈{20,40}")
    rng = random.Random(11)
    rows, t = [], 1.70e9
    for k in range(120):
        typ, u, qs, beta = ("research", "file", (1, 2), 0.6) if k % 2 == 0 else ("impl", "step", (20, 40), 1.0)
        q = float(rng.choice(qs))
        T = 50.0 * q ** beta * math.exp(rng.gauss(0, 0.3))
        rows.append(dict(v=3, kind="step", id=f"s/T{k}/x", session="s", task=f"T{k}", type=typ, u=u, d="D3",
                         q=q, model="A/high", actual_sec=T, started_at=t, completed_at=t + T))
        t += T + 1
    m = est.Model(rows, 1.0)
    for typ, u, beta, q in (("research", "file", 0.6, 8.0), ("impl", "step", 1.0, 80.0)):
        mu, var = m.predict(dict(type=typ, u=u, q=q, d="D3", model="A/high"))
        truth = 50.0 * q ** beta
        print(f"  {typ} q={q:>4g}: 预测={math.exp(mu):8.0f}s 真值={truth:8.0f}s 误差={math.exp(mu)/truth-1:+7.0%}"
              f"（该类型 β 估计={m.coef('beta0',1.0)+m.coef(f'beta:{typ}|{u}',0.0):.2f}，真 {beta}）")


def e5_cold_start_bias():
    print("\nE5 冷启动：agent 开工估计系统性偏乐观 2×（历史 <10 条）")
    rows = gen_rows(3, 13)  # 9 条样本，带 base 的很少
    for r in rows:
        r["base_sec"] = r["actual_sec"] / 2.0      # 故意低估一半
    m = est.Model(rows, 1.0)
    bst = est.base_stats(rows, m.s2)
    st = dict(type="research", u="file", q=4.0, d="D3", base_sec=100.0)   # 真值应约 40*4^0.9≈300
    fc = est.step_forecast(m, st, "A/high", 3, bst, 0.05)
    lo, hi = fc["range_sec"]
    print(f"  base={100}s（同偏差 2×）→ 预测={fc['pred_sec']:.0f}s 区间=[{lo:.0f},{hi:.0f}]  真值≈{40*4**0.9:.0f}s")
    print(f"  区间是否覆盖真值：{'是' if lo <= 40 * 4 ** 0.9 <= hi else '否'}；base 统计={bst['src']} n={bst['n']}")


def e6_interaction():
    print("\nE6 型号×类型交互：假阳率与检出率（各 40 次）")
    fp = tp = 0
    for seed in range(40):
        rows = gen_rows(60, 1000 + seed, noise_sd=0.3)
        cands = {(t, m) for t, m in ((f"{t}|{u}", mm) for t, u, *_ in TYPES for mm in MODELS)}
        base = est.Model(rows, 1.0)
        probe = est.Model(rows, 1.0, interact=cands)
        fp += probe.s2 / base.s2 < 0.9
    for seed in range(40):
        rows = gen_rows(60, 2000 + seed, noise_sd=0.3, delta=None)
        for r in rows:
            if r["type"] == "impl" and r["model"] == "B/medium":
                r["actual_sec"] *= math.exp(0.6)
        cands = {(t, m) for t, m in ((f"{t}|{u}", mm) for t, u, *_ in TYPES for mm in MODELS)}
        base = est.Model(rows, 1.0)
        probe = est.Model(rows, 1.0, interact=cands)
        tp += probe.s2 / base.s2 < 0.9
    print(f"  无交互时误启用：{fp}/40 = {fp / 40:.0%}；有 0.6 对数效应时启用：{tp}/40 = {tp / 40:.0%}")


def e7_conditional_remaining():
    print("\nE7 进行中步骤的条件剩余（单步 lognormal，真值已知）")
    for ratio in (0.3, 0.6, 1.0, 1.8):
        med_ln, sd = math.log(600.0), 0.5
        e = 600.0 * ratio
        st = dict(status="running", ln=med_ln, sd_mc=sd, q=1.0, units=0.0, started_at=0.0, pred_sec=600.0, type="impl", u="step")
        state = dict(theta=dict(m=0.0, v=0.0), steps=[st], slots=1, unit_min=5)
        st["id"] = "s1"; st["dep"] = []
        ln_med, s, method, elapsed = est.running_remaining(st, e, 0.0, 0.0)
        pred = math.exp(ln_med)
        # 经验条件中位数
        rng = random.Random(5)
        vals = []
        for _ in range(200000):
            T = math.exp(med_ln + sd * rng.gauss(0, 1))
            if T > e:
                vals.append(T - e)
        vals.sort()
        emp = vals[len(vals) // 2]
        n_surv = len(vals) / 200000
        print(f"  已用={ratio:>4.1f}×中位: 预测剩余={pred:6.0f}s 经验剩余中位={emp:6.0f}s 偏差={pred / emp - 1:+6.0%} "
              f"未完成比例={n_surv:.0%} 方法={method}")


def e8_batch_rate():
    print("\nE8 批量速率：真实含 30% 准备开销（固定 300s + 每单元 10s）")
    rng = random.Random(17)
    q, setup, per = 40, 300.0, 10.0
    T_total = setup + per * q
    for k in (5, 10, 20, 30):
        e = setup + per * k
        st = dict(id="s", status="running", ln=math.log(T_total), sd_mc=0.15, q=float(q), units=float(k),
                  started_at=0.0, pred_sec=T_total, setup_sec=setup, dep=[], type="test", u="case")
        ln_med, s, method, elapsed = est.running_remaining(st, e, 0.0, 0.0)
        print(f"  完成 {k:>2}/{q}：预测剩余={math.exp(ln_med):6.0f}s 真剩余={per*(q-k):6.0f}s 方法={method}")


def e9_mc_schedule():
    print("\nE9 整体排程与区间（MC）：固定时长下 P50 应等于点估计")
    def mk(ids, durs, dep, slots):
        state = dict(theta=dict(m=0.0, v=0.0), slots=slots, unit_min=1, steps=[])
        for i, d in zip(ids, durs):
            state["steps"].append(dict(id=i, name=i, status="pending", ln=math.log(d), sd_mc=0.0, dep=dep[i],
                                       type="x", u="s", q=1.0, pred_sec=d, range_sec=[d, d]))
        return state
    s2 = mk(["a", "b", "c"], [600, 1200, 300], {"a": [], "b": [], "c": ["a", "b"]}, 2)
    tot, _ = est.simulate(s2, 0.0, draws=400)
    print(f"  并行 a(10)∥b(20)+c(5) 两槽：P50={tot[1]/60:.1f} 分钟（期望 25）")
    s1 = mk(["a", "b", "c"], [600, 1200, 300], {"a": [], "b": [], "c": ["a", "b"]}, 1)
    tot, _ = est.simulate(s1, 0.0, draws=400)
    print(f"  同任务单槽：P50={tot[1]/60:.1f} 分钟（期望 35）")


def e10_interval_coverage_mc():
    print("\nE10 整体区间覆盖（40 个任务；用独立抽样检验，而不是与单次抽样比中位数）")
    hits, width, med_err = 0, [], []
    for seed in range(40):
        rng = random.Random(seed)
        state = dict(theta=dict(m=0.0, v=0.0), slots=1, unit_min=5, steps=[])
        mus, sds = [], []
        for i in range(4):
            mu = math.log(rng.uniform(300, 1200))
            mus.append(mu)
            sds.append(0.4)
            state["steps"].append(dict(id=f"s{i}", name=f"s{i}", status="pending", ln=mu, sd_mc=0.4,
                                       dep=[f"s{i-1}"] if i else [], type="x", u="s", q=1.0,
                                       pred_sec=math.exp(mu), range_sec=[math.exp(mu - 1), math.exp(mu + 1)]))
        state["steps"][0].update(status="running", started_at=0.0, units=0.0)
        tot, _ = est.simulate(state, 0.0, draws=800)
        # 独立抽样 400 次得到"真实"完成时间分布
        draws = []
        r2 = random.Random(seed + 999)
        for _ in range(400):
            draws.append(sum(math.exp(mu + sd * r2.gauss(0, 1)) for mu, sd in zip(mus, sds)))
        draws.sort()
        true_med = draws[len(draws) // 2]
        hits += sum(1 for t in draws if tot[0] <= t <= tot[2]) / len(draws)
        width.append((tot[2] - tot[0]) / max(tot[1], 1e-9))
        med_err.append(tot[1] / true_med - 1)
    print(f"  平均覆盖={hits/40:.0%}（目标 ≈60%）；平均相对区间宽={statistics.fmean(width):.1%}；"
          f"P50 相对中位误差中位={statistics.median(med_err):+.1%}")


def e11_extrapolation():
    """固定开销数据：q=80/160 超出历史上限 40，检验外推罚项与 agent 估计加权。"""
    print("\nE11 外推（历史 q∈{1..40}，目标 q=80/160；agent 估计 ±25%）")
    rng = random.Random(7)
    a, b = 120.0, 20.0
    rows, t = [], 1.7e9
    for k in range(90):
        q = rng.choice([1, 2, 5, 10, 20, 40])
        T = (a + b * q) * math.exp(rng.gauss(0, 0.25))
        rows.append(dict(v=3, kind="step", id=f"s/T{k}/x", session="s", task=f"T{k}", type="batch", u="item", d="D3",
                         q=float(q), model="A/high", actual_sec=T, started_at=t, completed_at=t + T))
        t += T + 1
    m = est.Model(rows, 1.0)
    bst = est.base_stats(rows, m.s2)
    qrange = {"batch|item": (min(math.log(r["q"]) for r in rows), max(math.log(r["q"]) for r in rows))}
    for q in (40, 80, 160):
        truth = a + b * q
        r0 = random.Random(q)
        errs, cov = [], 0
        for _ in range(300):
            base = truth * math.exp(r0.gauss(0, 0.25))
            fc = est.step_forecast(m, dict(type="batch", u="item", q=float(q), d="D3", base_sec=base),
                                   "A/high", 90, bst, 0.02, qrange=qrange)
            errs.append(fc["pred_sec"] / truth - 1)
            cov += fc["range_sec"][0] <= truth <= fc["range_sec"][1]
        print(f"  q={q:>3}（超出 {fc['extrap']:.2f} ln 单位）: 平均误差={statistics.fmean(errs):+6.0%} "
              f"区间覆盖真值={cov/300:.0%}")


def e12_curvature():
    print("\nE12 曲率诊断：残差与 (ln q)² 的相关系数（残差对 ln q 正交，必须用平方项）")
    rng = random.Random(7)
    a, b = 120.0, 20.0
    rows = []
    t = 1.7e9
    for k in range(90):
        q = rng.choice([1, 2, 5, 10, 20, 40])
        T = (a + b * q) * math.exp(rng.gauss(0, 0.25))
        rows.append(dict(v=3, kind="step", id=f"s/T{k}/x", session="s", task=f"T{k}", type="batch", u="item", d="D3",
                         q=float(q), model="A/high", actual_sec=T, started_at=t, completed_at=t + T))
        t += T + 1
    def corr(rows):
        m = est.Model(rows, 1.0)
        xs = [math.log(r["q"]) for r in rows]
        mx = statistics.fmean(xs)
        res = [math.log(r["actual_sec"]) - m.mean(m._f(r)) for r in rows]
        sq = [(x - mx) ** 2 for x in xs]
        mres, msq = statistics.fmean(res), statistics.fmean(sq)
        num = sum((s - msq) * (e - mres) for s, e in zip(sq, res))
        den = math.sqrt(sum((s - msq) ** 2 for s in sq) * sum((e - mres) ** 2 for e in res))
        return num / den
    pure = []
    t2 = 1.7e9
    rng2 = random.Random(3)
    for k in range(90):
        q = float(rng2.randint(1, 40))
        T = 20 * q * math.exp(rng2.gauss(0, 0.25))
        pure.append(dict(v=3, kind="step", id=f"p/T{k}/x", session="p", task=f"T{k}", type="batch", u="item", d="D3",
                         q=q, model="A/high", actual_sec=T, started_at=t2, completed_at=t2 + T))
        t2 += T + 1
    print(f"  固定开销数据 r={corr(rows):+.2f}（应 >0.4 被标记）；纯幂律 r={corr(pure):+.2f}（应接近 0）")


def e13_edges():
    print("\nE13 数值与输入边界")
    m = est.Model([], 1.0)
    fc = est.step_forecast(m, dict(type="x", u="s", q=1e12, d="D3", base_sec=10.0), "A/high", 0,
                           dict(bias=0, var=0.32, by_type={}), 0.1)
    print(f"  q=1e12 无历史：预测={fc['pred_sec']:.0f}s 区间上界={fc['range_sec'][1]:.2e}（不崩溃）")
    est.resolve_tz("Asia/Kolkata")
    print(f"  半小时时区 Asia/Kolkata → {est.tz_label()}（应为 UTC+5.5）")


def e14_theta_and_invariants():
    print("\nE14 任务级偏移的方向与幅度、资源互斥、进度不变量、100 步性能")
    # θ 修正：任务真实慢 1.5×，完成 k 步后剩余预测应乘 ≈1.5^(k/(k+1))
    import copy
    base = math.log(600.0)
    for k, expect in ((1, 1.5 ** 0.5), (2, 1.5 ** (2 / 3)), (3, 1.5 ** 0.75)):
        state = dict(theta=dict(m=0.0, v=0.3, k=0, sum_r=0.0), tau2=0.3, s2=0.6, slots=1, unit_min=5,
                     steps=[dict(id=f"s{i}", name=f"s{i}", status="pending", ln=base, sd_mc=0.3, dep=[], type="x",
                                 u="s", q=1.0, pred_sec=600.0, range_sec=[400, 900]) for i in range(6)])
        state["steps"][0]["status"] = "running"
        state["steps"][0]["started_at"] = 0.0
        state["steps"][0]["units"] = 0.0
        now = 0.0
        for i in range(k):
            now += 900.0     # 真实 1.5×
            st = state["steps"][i]
            st.update(status="done", actual_sec=900.0, finished_at=now)
            resid = math.log(900.0) - st["ln"]
            th = state["theta"]
            th["k"] += 1
            th["sum_r"] += resid
            rbar = th["sum_r"] / th["k"]
            s2w = max(0.25 * state["s2"], state["s2"] - state["tau2"])
            th["m"] = rbar * state["tau2"] / (state["tau2"] + s2w / th["k"])
            th["v"] = 1 / (1 / state["tau2"] + th["k"] / s2w)
            state["steps"][i + 1].update(status="running", started_at=now, units=0.0)
        tot, per = est.simulate(state, now, draws=400)
        nxt_id = state["steps"][k + 1]["id"]
        print(f"  完成 {k} 步后【下一步】剩余≈{per[nxt_id]['rem_sec']/60:5.1f} 分钟"
              f"（理论 {600*expect/60:5.1f}）")
    # 资源互斥：两个 10 分钟步骤共用同一 res，即使 4 槽也应串行
    state = dict(theta=dict(m=0.0, v=0.0), slots=4, unit_min=5, steps=[
        dict(id="a", name="a", status="pending", ln=math.log(600), sd_mc=0.0, dep=[], type="x", u="s", q=1.0,
             pred_sec=600, range_sec=[600, 600], res="browser"),
        dict(id="b", name="b", status="pending", ln=math.log(600), sd_mc=0.0, dep=[], type="x", u="s", q=1.0,
             pred_sec=600, range_sec=[600, 600], res="browser")])
    state["steps"][0].update(status="running", started_at=0.0, units=0.0)
    tot, _ = est.simulate(state, 0.0, draws=200)
    print(f"  共享 res 的两步、4 槽：P50={tot[1]/60:.0f} 分钟（应为 20，不能并行）")
    # 进度不变量：未交付封顶 99；完成全部但无交付=99
    state2 = dict(theta=dict(m=0.0, v=0.0), steps=[
        dict(id="a", status="done", pred_sec=100, q=1.0, type="x", u="s"),
        dict(id="b", status="done", pred_sec=100, q=1.0, type="x", u="s")])
    print(f"  全部完成未交付时 progress_percent={est.progress_percent(state2)}（应为 99）")
    # 100 步性能
    import time
    state = dict(theta=dict(m=0.0, v=0.0), slots=2, unit_min=30, steps=[
        dict(id=f"s{i}", name=f"s{i}", status="pending", ln=math.log(300), sd_mc=0.3,
             dep=[f"s{i-1}"] if i else [], type="x", u="s", q=1.0, pred_sec=300, range_sec=[200, 500]) for i in range(100)])
    state["steps"][0].update(status="running", started_at=0.0, units=0.0)
    t0 = time.time()
    est.simulate(state, 0.0, draws=2000)
    print(f"  100 步 2000 次蒙特卡洛：{time.time()-t0:.2f}s")


if __name__ == "__main__":
    e1_coverage_by_n()
    e2_fixed_cost_misspec()
    e3_heavy_tail()
    e4_confounded_q()
    e5_cold_start_bias()
    e6_interaction()
    e7_conditional_remaining()
    e8_batch_rate()
    e9_mc_schedule()
    e10_interval_coverage_mc()
    e11_extrapolation()
    e12_curvature()
    e13_edges()
    e14_theta_and_invariants()
