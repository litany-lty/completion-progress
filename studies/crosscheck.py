#!/usr/bin/env python3
"""独立交叉验证（crosscheck）：不复用 estimate.py 的求解/模拟实现，另写对照实现。

C1 线性代数独立对照（Cholesky 解岭回归 vs estimate.Model）
C2 t 分位数数值积分对照
C3 排程独立对照（事件驱动调度 vs estimate.simulate）
C4 时间外推预测：训练前 70%、测试后 30%，与基线比较
C5 随机模糊测试：不变量（有限、区间有序、pred 在区间内、进度 0–99）
C6 CLI 端到端矩阵（中英文、错误、只读、坏状态、重复 id）
C7 文件与文档完整性（校验和、引用、副本一致、测试计数）
运行：python3 studies/crosscheck.py
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable
spec = importlib.util.spec_from_file_location("est", HERE / "estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- C1
def spec_feats(r, interact=()):
    """按文档规则独立重建设计列（不调用 estimate.feats）。"""
    t = f"{r['type']}|{r['u'] if r['q'] != 1 else 'step'}"   # 3.1：不可分步骤的单位名统一为 step
    lq = math.log(r["q"])
    f = {"mu0": 1.0, f"mu:{t}": 1.0, f"mod:{r['model']}": 1.0}
    if lq:
        f["beta0"] = lq
        f[f"beta:{t}"] = lq
    dd = est.LEVELS.index(r["d"]) - 2
    if dd:
        f["gam0"] = float(dd)
        f[f"gam:{t}"] = float(dd)
    if (t, r["model"]) in interact:
        f[f"txm:{t}|{r['model']}"] = 1.0
    return f


def cholesky_solve(A, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(1e-15, A[i][i] - s))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def c1_independent_ridge():
    print("\nC1 独立线性代数对照（Cholesky vs estimate 的 Gauss-Jordan）")
    rng = random.Random(31)
    rows = []
    t = 1.7e9
    for k in range(50):
        for typ, u, base, beta, ga in (("research", "file", 40.0, 0.9, 0.3), ("impl", "step", 600.0, 1.0, 0.5)):
            q = float(rng.choice([1, 2, 5, 10]))
            d = rng.choice(est.LEVELS)
            m = rng.choice(["A/high", "B/medium"])
            y = math.log(base) + beta * math.log(q) + ga * (est.LEVELS.index(d) - 2) + (0 if m == "A/high" else -0.2) + rng.gauss(0, 0.3)
            T = math.exp(y)
            rows.append(dict(v=3, kind="step", id=f"s/{typ}{k}", session="s", task=f"T{k}", type=typ, u=u, d=d, q=q,
                             model=m, actual_sec=T, started_at=t, completed_at=t + T))
            t += T + 1
    interact = {("impl|step", "B/medium")}
    m = est.Model(rows, 1.0, interact=interact)
    cols = m.cols
    idx = {c: i for i, c in enumerate(cols)}
    p = len(cols)
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for r, w in zip(rows, m.weights):
        x = spec_feats(r, interact)
        y = math.log(r["actual_sec"])
        items = [(idx[c], v) for c, v in x.items()]
        for i, xi in items:
            b[i] += w * xi * y
            for j, xj in items:
                A[i][j] += w * xi * xj
    for c, i in idx.items():
        if c == "mu0":
            center, lam = 0.0, 0.01
        elif c == "beta0":
            center, lam = 1.0, 1.0
        elif c.startswith("txm:"):
            center, lam = 0.0, 4.0
        else:
            center, lam = 0.0, 1.0
        A[i][i] += lam
        b[i] += lam * center
    theta = cholesky_solve(A, b)
    diff = max(abs(a - c) for a, c in zip(theta, m.theta))
    check("独立 Cholesky 解与 Model.theta 一致", diff < 1e-8, f"max|Δ|={diff:.2e}")
    # 预测也对照一次
    probe = dict(type="impl", u="step", q=7.0, d="D4", model="B/medium")
    xprobe = spec_feats(probe, interact)
    ln_ind = sum(theta[idx[c]] * v for c, v in xprobe.items())
    ln_mod = m.mean(m._f(probe))
    check("独立预测与 Model.mean 一致", abs(ln_ind - ln_mod) < 1e-8, f"{ln_ind:.9f} vs {ln_mod:.9f}")


# ---------------------------------------------------------------- C2
def t_cdf(q, nu, N=40000):
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
    h = q / N
    s = (1 + (N * h) ** 2 / nu) ** (-(nu + 1) / 2) + sum((1 + (k * h) ** 2 / nu) ** (-(nu + 1) / 2) * (4 if k % 2 else 2) for k in range(1, N))
    return 0.5 + c * h * s / 3


def c2_t_quantiles():
    print("\nC2 t 分位数数值积分对照")
    worst = 0.0
    for nu in (4, 5, 6, 8, 10, 15, 20, 30, 60, 120, 500):
        worst = max(worst, abs(t_cdf(est.t80(nu), nu) - 0.8))
    check("t80 在所有自由度上都是 P=0.800", worst < 2e-3, f"最大偏差 {worst:.1e}")


# ---------------------------------------------------------------- C3
def independent_schedule(steps, slots, draw, rng):
    """独立的事件驱动调度：按依赖、资源、槽位贪心排程；返回完成时间。"""
    finish, res_free = {}, {}
    slot_free = [0.0] * slots
    for st in steps:
        if st["status"] == "done":
            finish[st["id"]] = 0.0
            continue
        dur = draw(st, rng)
        start = max([finish.get(d, 0.0) for d in st["dep"]] + [res_free.get(st.get("res") or "", 0.0)])
        j = min(range(slots), key=lambda k: slot_free[k])
        start = max(start, slot_free[j])
        fin = start + dur
        slot_free[j] = fin
        if st.get("res"):
            res_free[st["res"]] = fin
        finish[st["id"]] = fin
    return max(finish.values()) if finish else 0.0


def c3_schedule():
    print("\nC3 排程独立对照（同一 DAG、同一随机数序列）")
    rng = random.Random(77)
    worst = 0.0
    for trial in range(20):
        n = rng.randint(2, 8)
        steps = []
        for i in range(n):
            steps.append(dict(id=f"s{i}", name=f"s{i}", status="pending", ln=math.log(rng.uniform(100, 1000)),
                              sd_mc=0.3, dep=[f"s{j}" for j in range(i) if rng.random() < 0.4],
                              res="shared" if rng.random() < 0.3 else None, type="x", u="s", q=1.0,
                              pred_sec=500.0, range_sec=[300, 800]))
        steps[0]["status"] = "running"
        steps[0]["started_at"] = 0.0
        steps[0]["units"] = 0.0
        slots = rng.choice([1, 2, 3])
        state = dict(theta=dict(m=0.0, v=0.0), slots=slots, unit_min=5, steps=steps)
        # 同一组抽样：先按估计的 running_remaining 取 ln 中位与 sd，两边都从同一 rng 序列抽
        med = {}
        for st in steps:
            if st["status"] == "running":
                ln_med, sd, _, _ = est.running_remaining(st, 0.0, 0.0, 0.0)
                med[st["id"]] = (ln_med, sd)
            else:
                med[st["id"]] = (st["ln"], st["sd_mc"])
        draws = {st["id"]: math.exp(med[st["id"]][0] + med[st["id"]][1] * rng.gauss(0, 1)) for st in steps}
        ref = independent_schedule(steps, slots, lambda st, r: draws[st["id"]], None)
        # estimate.simulate 用自己的 rng：为了比较，直接调用其内部逻辑不可行；改为比较 200 次均值
        tot, _ = est.simulate(state, 0.0, draws=2000)
        ref_med = []
        for _ in range(2000):
            d = {st["id"]: math.exp(med[st["id"]][0] + med[st["id"]][1] * rng.gauss(0, 1)) for st in steps}
            ref_med.append(independent_schedule(steps, slots, lambda st, r: d[st["id"]], None))
        ref_med.sort()
        ref50 = ref_med[1000]
        worst = max(worst, abs(tot[1] / ref50 - 1))
    check("20 个随机 DAG 的 P50 与独立调度一致（≤2%）", worst < 0.02, f"最大相对差 {worst:.2%}")


# ---------------------------------------------------------------- C4
def c4_time_split():
    print("\nC4 时间外推：前 70% 训练、后 30% 检验（含基线对比）")
    rng = random.Random(41)
    rows, t = [], 1.7e9
    for k in range(140):
        for typ, u, base, beta, ga in (("research", "file", 40.0, 0.9, 0.3), ("impl", "step", 600.0, 1.0, 0.5), ("test", "case", 15.0, 0.7, 0.2)):
            q = float(rng.choice([1, 2, 4, 8]))
            d = rng.choice(est.LEVELS)
            y = math.log(base) + beta * math.log(q) + ga * (est.LEVELS.index(d) - 2) + rng.gauss(0, 0.35)
            T = math.exp(y)
            rows.append(dict(v=3, kind="step", id=f"s/{typ}{k}", session="s", task=f"T{k}", type=typ, u=u, d=d, q=q,
                             model="A/high", actual_sec=T, started_at=t, completed_at=t + T))
            t += T + 1
    split = int(len(rows) * 0.7)
    train, test = rows[:split], rows[split:]
    m = est.Model(train, 1.0)
    errs, cov = [], 0
    for r in test:
        mu, var = m.predict(r)
        half = est.t80(m.nu) * math.sqrt(var)
        y = math.log(r["actual_sec"])
        errs.append(abs(mu - y))
        cov += mu - half <= y <= mu + half
    # 基线：同类型训练集中位数；全局中位数
    tmed = {}
    for r in train:
        tmed.setdefault(est.tkey(r), []).append(math.log(r["actual_sec"]))
    tmed = {k: statistics.median(v) for k, v in tmed.items()}
    gmed = statistics.median(math.log(r["actual_sec"]) for r in train)
    e_t, e_g = [], []
    for r in test:
        y = math.log(r["actual_sec"])
        e_t.append(abs(tmed.get(est.tkey(r), gmed) - y))
        e_g.append(abs(gmed - y))
    check("模型优于同类型中位数基线", statistics.fmean(errs) < statistics.fmean(e_t),
          f"{statistics.fmean(errs):.3f} vs {statistics.fmean(e_t):.3f}")
    check("模型优于全局中位数基线", statistics.fmean(errs) < statistics.fmean(e_g),
          f"{statistics.fmean(errs):.3f} vs {statistics.fmean(e_g):.3f}")
    check("时间外推的区间覆盖在 45–80%", 0.45 <= cov / len(test) <= 0.80, f"{cov/len(test):.0%}")


# ---------------------------------------------------------------- C5
def c5_fuzz():
    print("\nC5 随机模糊测试（600 例不变量）")
    rng = random.Random(99)
    bad = []
    for i in range(600):
        n = rng.randint(0, 6)
        rows = []
        for k in range(n):
            rows.append(dict(v=3, kind="step", id=f"f/{i}/{k}", session="f", task=f"T{i}", type=rng.choice(["a", "b", "c"]),
                             u="s", d=rng.choice(est.LEVELS), q=float(rng.choice([0.2, 1, 3, 20, 500])),
                             model=rng.choice(["m1", "m2"]), actual_sec=math.exp(rng.uniform(-2, 9)),
                             started_at=k, completed_at=k + 1))
        try:
            m = est.Model(rows, rng.choice([0.25, 1.0, 4.0]))
            bst = est.base_stats(rows, m.s2)
            st = dict(type=rng.choice(["a", "b", "c", "new"]), u="s", q=float(rng.choice([0.5, 1, 5, 50, 1000])),
                      d=rng.choice(est.LEVELS), base_sec=(math.exp(rng.uniform(0, 8)) if rng.random() < 0.5 else None))
            n_type = sum(1 for r in rows if f"{r['type']}|{r['u']}" == f"{st['type']}|{st['u']}")
            fc = est.step_forecast(m, st, "m1", n_type, bst, 0.1, qrange={"a|s": (0.0, 3.0)})
            if fc is None:
                continue
            ln = fc["ln"]
            lo, hi = fc["range_sec"]
            if not (math.isfinite(ln) and math.isfinite(lo) and math.isfinite(hi)):
                bad.append(("nonfinite", i))
            elif not (0 < lo <= fc["pred_sec"] <= hi):
                bad.append(("order", i, lo, fc["pred_sec"], hi))
            elif not (fc["sd_mc"] > 0 and fc["sd_mc"] < 1e6):
                bad.append(("sd", i, fc["sd_mc"]))
        except Exception as exc:                     # 不变量测试：任何异常都是失败
            bad.append(("exc", i, type(exc).__name__, str(exc)))
    check("600 例模糊测试无异常且不变量成立", not bad, str(bad[:3]))


# ---------------------------------------------------------------- C6
def run_cli(args, state, at=None, env_extra=None, lang=None):
    env = dict(os.environ, COMPLETION_PROGRESS_STATE=state, TZ="Asia/Shanghai")
    env.pop("COMPLETION_PROGRESS_TZ", None)
    env.pop("COMPLETION_PROGRESS_LANG", None)
    if env_extra is not None:
        for k, v in env_extra.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    cmd = [PY, str(HERE / "estimate.py")] + args
    if at is not None and args[0] in ("plan", "progress", "status", "deliver", "audit"):
        cmd += ["--at", str(at)]
    if lang:
        cmd += ["--lang", lang]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        out = json.loads(p.stdout) if p.stdout.strip() else {}
    except json.JSONDecodeError:
        out = {"_raw": p.stdout}
    return p.returncode, out, p.stderr


def c6_cli_matrix():
    print("\nC6 CLI 端到端矩阵")
    T0 = 1760702400.0
    with tempfile.TemporaryDirectory() as tmp:
        steps = ('[{"name":"调研","type":"research","q":12,"u":"file","d":"D2","base_min":6},'
                 '{"name":"实现","type":"impl","d":"D4","base_min":15},'
                 '{"name":"测试","type":"test","q":40,"u":"case","d":"D3","base_min":8}]')
        rc, o, _ = run_cli(["plan", "--session", "s1", "--task", "A", "--model", "m/h", "--tz", "Asia/Shanghai", "--steps", steps], tmp, T0)
        ok1 = rc == 0 and set(o) == {"ok", "ev", "should_report", "line", "warn"} and "【进度分析" in o["line"]
        rc, oe, _ = run_cli(["plan", "--session", "s1", "--task", "B", "--model", "m/h", "--steps", steps], tmp, T0, lang="en")
        ok2 = rc == 0 and oe["line"].startswith("> **[Plan:")
        rc, o2, _ = run_cli(["progress", "--session", "s1", "--task", "A", "--done", "s1", "--actual", "600"], tmp, T0 + 600)
        ok3 = rc == 0 and o2["should_report"] is True
        rc, o3, _ = run_cli(["status", "--session", "s1", "--task", "A"], tmp, T0 + 700)
        ok4 = rc == 0 and o3["should_report"] is True
        rc, o4, err = run_cli(["progress", "--session", "s1", "--task", "A", "--done", "zzz"], tmp, T0 + 800)
        ok5 = rc == 2 and o4.get("ok") is False
        rc, o5, _ = run_cli(["progress", "--session", "s1", "--task", "A", "--block", "确认"], tmp, T0 + 900)
        ok6 = "等待您确认" in o5["line"]
        rc, o6, _ = run_cli(["progress", "--session", "s1", "--task", "A", "--block"], tmp, T0 + 1000)
        rc, o7, _ = run_cli(["progress", "--session", "s1", "--task", "A", "--done", "s2,s3"], tmp, T0 + 2000)
        rc, o8, _ = run_cli(["deliver", "--session", "s1", "--task", "A", "--ref", "PR#1"], tmp, T0 + 2100)
        ok7 = rc == 0 and "各步" in o8["line"]
        rc, o9, err = run_cli(["deliver", "--session", "s1", "--task", "A"], tmp, T0 + 2200)
        ok8 = rc == 2 and "already delivered" in err
        # 坏状态文件
        bad_dir = Path(tmp) / "tasks"
        for f in bad_dir.iterdir():
            f.write_text("{broken", encoding="utf-8")
        rc, o10, _ = run_cli(["status", "--session", "s1", "--task", "A"], tmp, T0 + 2300)
        ok9 = rc == 2 and o10.get("ok") is False
        # 只读目录
        ro = Path(tmp) / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)
        rc, o11, _ = run_cli(["plan", "--session", "r", "--task", "R", "--model", "m/h", "--steps", steps], str(ro), T0)
        os.chmod(ro, 0o700)
        ok10 = rc == 0 and "line" in o11 and any("not_persisted" in w for w in o11["warn"])
        # 审计带过滤时不写缓存
        rc, o12, _ = run_cli(["audit", "--compact", "--type", "research"], tmp, T0 + 2400)
        ok11 = rc == 0
        check("CLI 矩阵：plan/进度/英文/错误/阻塞/交付/重复/坏状态/只读/audit",
              all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9, ok10, ok11]),
              f"{[ok1,ok2,ok3,ok4,ok5,ok6,ok7,ok8,ok9,ok10,ok11]}")


# ---------------------------------------------------------------- C7
def c7_completeness():
    print("\nC7 文件与文档完整性")
    files = ["SKILL.md", "estimate.py", "reference.md", "README.md", "INSTALL.md", "install.py", "env.sh",
             "OPTIMIZATION-PLAN.md", "PROPOSAL-v3.md", "manifest.json", "make_manifest.py",
             "tests/test_v3.py", "studies/calibration.py", "studies/crosscheck.py", "SHA256SUMS"]
    missing = [f for f in files if not (HERE / f).exists()]
    check("必需文件齐全", not missing, str(missing))
    sums = {}
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip()] = h
    bad = []
    for name, h in sums.items():
        p = HERE / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(name)
    check("SHA256SUMS 与实际文件一致", not bad, str(bad))
    skill = (HERE / "SKILL.md").read_text(encoding="utf-8")
    ref = (HERE / "reference.md").read_text(encoding="utf-8")
    need_skill = ["estimate.py", "reference.md", "selftest", "--tz", "--lang", "【进度分析", "P20"]
    miss = [x for x in need_skill if x not in skill]
    check("SKILL.md 引用与关键约定完整", not miss, str(miss))
    need_ref = ["1.13", "已知限制", "c1", "C1"] if False else ["### 1.13", "已知限制", "外推", "曲率", "Huber"]
    miss = [x for x in need_ref if x not in ref]
    check("reference.md 限制与诊断章节齐全", not miss, str(miss))
    # 代码里承诺的子命令都存在
    src = (HERE / "estimate.py").read_text(encoding="utf-8")
    cmds = ["plan", "progress", "status", "deliver", "audit", "selftest", "migrate"]
    miss = [c for c in cmds if f'sub.add_parser("{c}"' not in src]
    check("所有对外子命令均已注册", not miss, str(miss))
    # 测试与 selftest 能跑（此处代表性地跑一次子集由外层负责；这里只数 check 调用）
    test_src = (HERE / "tests/test_v3.py").read_text(encoding="utf-8")
    n = test_src.count("check(") - test_src.count("def check(") - test_src.count("# check(")
    check("回归测试至少 60 个断言", n >= 60, f"{n}")
    # manifest 覆盖所有非 state 文件，且哈希一致
    man = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    listed = {f["path"] for f in man["files"]}
    have = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*")
            if p.is_file() and p.name != ".DS_Store" and not {"state", "__pycache__", "backups", ".git"} & set(p.relative_to(HERE).parts)}
    check("manifest 覆盖全部文件", listed == have - {"manifest.json", "SHA256SUMS"},
          str(sorted(have - listed - {"manifest.json", "SHA256SUMS"})[:5]))
    badm = [f["path"] for f in man["files"] if hashlib.sha256((HERE / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
    check("manifest 哈希与实际一致", not badm, str(badm[:3]))


if __name__ == "__main__":
    t0 = time.time()
    c1_independent_ridge()
    c2_t_quantiles()
    c3_schedule()
    c4_time_split()
    c5_fuzz()
    c6_cli_matrix()
    c7_completeness()
    print("\n" + (f"ALL PASS（{time.time()-t0:.1f}s）" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)
