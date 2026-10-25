#!/usr/bin/env python3
"""completion-progress estimate.py v3 — 四维估时（类型·复杂度·任务量·执行模型）与执行中动态修正。

只用标准库。子命令：plan / progress / status / deliver / audit / migrate（`--help` 看参数）。
状态目录：--state > $COMPLETION_PROGRESS_STATE > ${XDG_STATE_HOME:-~/.local/state}/completion-progress
  tasks/<key>.json   当前任务状态，脚本独占、原子改写；agent 不手改
  history.jsonl      已完成步骤样本 + 任务汇总，只追加、冻结
  model.json         audit 选出的 κ 缓存
数学、约定值与迁移规则见 reference.md §1。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

VERSION = "3.1.0"
# 3.1：类型/型号归一后列数从数百降到几十，读 1 MiB / 4000 行仍在 0.3 秒内（v3.0 的 256 KiB 五天即触顶）
MAX_BYTES, MAX_LINES = 1024 * 1024, 4000
LEVELS = ("D1", "D2", "D3", "D4", "D5")
# 唯一的约定值：零数据时起作用，有数据后被替代（reference.md §1.4）
PRIOR = dict(kappa=1.0,        # 每个收缩系数等价 1 个伪样本；audit 在 KAPPAS 中按时间顺序检验后缓存
             alpha=2.0, beta=0.16,   # σ² 的 Inverse-Gamma 先验：σ≈0.4，即 ×/÷1.5
             w_base=0.5,       # agent 开工前估计 = 半个样本；≥10 条带 base 的样本后改用实测偏差与方差
             w_base_max=10.0,  # base 的有效样本上限，防止极小的历史方差让 agent 估计独断
             lam_global=0.01)  # 全局截距几乎不收缩
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
T80 = [(4, .941), (5, .920), (6, .906), (7, .896), (8, .889), (9, .883), (10, .879),
       (12, .873), (15, .866), (20, .860), (30, .854), (60, .848), (1e9, .842)]
Z80 = 0.8416212
ND = statistics.NormalDist()
MC_DRAWS = 2000
REPORT_GAP, REPORT_MIN_CHANGE, REPORT_REL_CHANGE, TAIL_SEC = 180, 120, 0.25, 120

# ---------------------------------------------------------------- 3.1 数据质量（reference.md §1.14）
# 受控类型词表：自由写法按关键词归类，原文另存 label。真实历史 253 种写法 → 7 类，
# 碎片化会让类型项学不出来、列数膨胀（v3.0 plan 24 秒的根因）。
CATEGORIES = ("research", "implement", "verify", "build", "docs", "deliver", "other")
_CAT_RULES = (
    ("deliver", r"deliver|交付|收尾|wrap"),
    ("docs", r"doc|handoff|report|summar|ledger|readme|manual|changelog|文档|登记|汇报|交接|说明书|台账"),
    ("research", r"research|investig|diagnos|diag|explor|locat|analy|audit|review|survey|inspect|read|调研|定位|排查|分析|梳理|复核|审查|研究"),
    ("verify", r"verif|test|check|valid|smoke|accept|\bqa\b|e2e|regress|验证|测试|验收|检查|回归|自测"),
    ("build", r"build|deploy|release|restart|reload|install|compile|package|launch|构建|部署|发布|重启|打包|启动|安装"),
    ("implement", r"impl|code|coding|edit|fix|patch|refactor|write|\bui\b|style|migrat|seed|feature|dev|change|修改|修复|实现|编码|改造|开发|迁移"),
)
# 型号族：同一模型的各种写法（大小写、宿主前缀、长句）归到一族；识别不了的占位写法归 unknown。
_FAMILY_RULES = (
    ("gpt-6.1", r"gpt[-_ ]?6\.1"), ("gpt-6", r"gpt[-_ ]?6"), ("gpt-5", r"gpt[-_ ]?5"),
    ("claude", r"claude|opus|sonnet|haiku"), ("gemini", r"gemini"), ("minimax", r"minimax"),
    ("deepseek", r"deepseek"), ("qwen", r"qwen"), ("glm", r"glm"), ("kimi", r"kimi"),
)
_PLACEHOLDER = r"current|root|session|当前|默认|主控|协调|unknown|config|配置|codex|未另行|^$"
CAUSES = ("env", "bug", "scope", "wait", "rework", "unclear", "none")
MICRO_SEC = 30                       # 短于此的自动计时多为连续 --done 摊出的伪迹
IDLE_MIN_SEC = 1800                  # 疑似把空闲算进去：> 30 分钟且 > 3× 区间上限（或 5× 点估计）
STALE_SEC, GC_SEC = 24 * 3600, 30 * 86400
EVIDENCE_MAX = 40                    # 共享历史只存位置引用；全文只留在任务状态文件
SHADOW_MIN_POOL, SHADOW_FAMILY_POOL, SHADOW_WINDOW = 20, 30, 300


class ContractError(ValueError):
    pass


# ---------------------------------------------------------------- 校验与读写
def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name}: expected a finite number, not bool/string")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        raise ContractError(f"{name}: expected {'positive' if positive else 'nonnegative'} finite value")
    return v


def real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{name}: expected a finite number")
    return float(value)


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ContractError(f"{name}: expected nonempty string (<=2048 characters)")
    return value


def choice(value: Any, allowed, name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ContractError(f"{name}: invalid category {value!r}")
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def norm_type(raw: str) -> str:
    """自由类型 → 受控类别；已是类别则原样。认不出的简短单词保留（小写），其余归 other。"""
    s = (raw or "").strip().lower()
    if s in CATEGORIES:
        return s
    for cat, pat in _CAT_RULES:
        if re.search(pat, s):
            return cat
    return s if re.fullmatch(r"[a-z][a-z0-9_]{0,11}", s) else "other"


def norm_model(raw: str | None) -> str:
    """模型写法 → 型号族（推理强度不入键：同族样本太少时拆开反而学不出）。"""
    s = (raw or "").strip().lower()
    hits = [(m.start(), i, fam) for i, (fam, pat) in enumerate(_FAMILY_RULES) for m in [re.search(pat, s)] if m]
    if hits:
        return min(hits)[2]                  # 多模型协作的长句取最先出现者（通常是主执行者）；同位置取更具体的规则
    tok = re.split(r"[/;；,，\s|]", s)[0]
    return "unknown" if re.search(_PLACEHOLDER, tok) else tok[:24]


def sample_weight(r: dict) -> float:
    """拟合权重：计时伪迹不学，范围变更降权；文件里照存，只在拟合时生效。"""
    timing, cause, a = r.get("timing"), r.get("cause"), r["actual_sec"]
    if cause == "wait":
        return 0.0
    if timing != "explicit" and a < MICRO_SEC:
        return 0.0
    if not cause or cause == "none":
        hi = (r.get("range_sec") or [None, None])[1]
        lim = 3 * hi if hi else (5 * r["pred_sec"] if r.get("pred_sec") else None)
        if lim and a > max(IDLE_MIN_SEC, lim):
            return 0.0
    w = 1.0
    if timing == "split":
        w *= 0.5
    if cause == "scope":
        w *= 0.5
    return w


def read_tail(path, max_bytes: int = MAX_BYTES, max_lines: int = MAX_LINES):
    """只读文件尾部 max_bytes，不全量扫描。"""
    info = dict(source_status="ok", file_bytes=0, truncated=False, scanned_lines=0, invalid_json=0)
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = max(0, size - max_bytes)
            stream.seek(start)
            raw = stream.read(max_bytes)
        info.update(file_bytes=size, truncated=start > 0)
    except FileNotFoundError:
        info["source_status"] = "missing"
        return [], info
    except OSError as exc:
        info.update(source_status="unreadable", error=type(exc).__name__)
        return [], info
    if start:
        raw = raw.partition(b"\n")[2]
    lines = raw.splitlines()
    if len(lines) > max_lines:
        info["truncated"] = True
        lines = lines[-max_lines:]
    rows = []
    for line in lines:
        if not line.strip():
            continue
        info["scanned_lines"] += 1
        try:
            obj = parse_json(line)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            rows.append(obj)
        except (ValueError, UnicodeError, RecursionError):
            info["invalid_json"] += 1
    return rows, info


def validate_step(r: dict) -> dict:
    out = {"v": 3, "kind": "step"}
    for k in ("id", "session", "task", "type", "u", "model"):
        out[k] = text(r.get(k), k)
    out["d"] = choice(r.get("d"), LEVELS, "d")
    out["q"] = number(r.get("q"), "q", positive=True)
    out["actual_sec"] = number(r.get("actual_sec"), "actual_sec", positive=True)
    for k in ("started_at", "completed_at"):
        out[k] = number(r.get(k), k)
    if out["completed_at"] < out["started_at"]:
        raise ContractError("completed_at precedes started_at")
    for k in ("pred_sec", "base_sec", "setup_sec", "forecast_at", "units", "paused_sec", "pred_v4"):
        if r.get(k) is not None:
            out[k] = number(r[k], k, positive=k in ("pred_sec", "base_sec", "pred_v4"))
    for key in ("range_sec", "range_v4"):
        if r.get(key) is not None:
            rs = r[key]
            if not isinstance(rs, list) or len(rs) != 2:
                raise ContractError(f"{key} needs two bounds")
            lo, hi = (number(v, key, positive=True) for v in rs)
            if lo > hi:
                raise ContractError(f"{key} must be ordered")
            out[key] = [lo, hi]
    if "pred_sec" in out and "forecast_at" in out and out["forecast_at"] > out["started_at"] + 1:
        # 3.0 重划基线时把新基线时刻当成了已开工步骤的预测时刻：实测仍有效，只丢掉这条“事后预测”
        for k in ("pred_sec", "range_sec", "pred_v4", "range_v4", "forecast_at"):
            out.pop(k, None)
    for k in ("coef_id", "evidence", "src", "label", "model_raw"):
        if r.get(k) is not None:
            out[k] = text(r[k], k)
    if r.get("timing") is not None:
        out["timing"] = choice(r["timing"], ("auto", "explicit", "split"), "timing")
    if r.get("cause") is not None:
        out["cause"] = choice(r["cause"], CAUSES, "cause")
    if isinstance(r.get("env"), dict):
        out["env"] = r["env"]
    return out


def validate_task(r: dict) -> dict:
    out = {"v": 3, "kind": "task"}
    for k in ("id", "session", "task", "model"):
        out[k] = text(r.get(k), k)
    n = r.get("n_steps")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ContractError("n_steps must be a positive integer")
    out["n_steps"] = n
    out["actual_sec"] = number(r.get("actual_sec"), "actual_sec", positive=True)
    for k in ("started_at", "completed_at"):
        out[k] = number(r.get(k), k)
    for k in ("pred_sec",):
        if r.get(k) is not None:
            out[k] = number(r[k], k, positive=True)
    if r.get("range_sec") is not None:
        out["range_sec"] = [number(v, "range_sec", positive=True) for v in r["range_sec"]]
    k = r.get("resid_k", 0)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ContractError("resid_k must be a nonnegative integer")
    out["resid_k"] = k
    out["resid_mean"] = real(r.get("resid_mean", 0.0), "resid_mean")
    if r.get("status") is not None:
        out["status"] = choice(r["status"], ("delivered", "abandoned"), "status")
    if r.get("paused_sec") is not None:
        out["paused_sec"] = number(r["paused_sec"], "paused_sec")
    return out


def load_history(path, as_of: float, max_bytes=MAX_BYTES, max_lines=MAX_LINES):
    rows, info = read_tail(path, max_bytes, max_lines)
    info.update(invalid_rows=0, legacy_rows=0, duplicate_rows=0, conflict_ids=0, future_rows=0)
    stores = {"step": {}, "task": {}}
    conflicts = set()
    for value in rows:
        if value.get("v") != 3:
            info["legacy_rows"] += 1
            continue
        try:
            row = validate_task(value) if value.get("kind") == "task" else validate_step(value)
        except ContractError:
            info["invalid_rows"] += 1
            continue
        if row["completed_at"] > as_of:          # 回测时不得偷看未来
            info["future_rows"] += 1
            continue
        store = stores[row["kind"]]
        if row["kind"] == "step":
            # 读取时派生（不改文件）：老历史同样受益于归一与伪迹降权
            if row["type"] != norm_type(row["type"]):
                row.setdefault("label", row["type"])
            row["model_raw"] = row.get("model_raw") or row["model"]
            row["type"], row["model"] = norm_type(row["type"]), norm_model(row["model"])
            row["w"] = sample_weight(row)
        else:
            row["model"] = norm_model(row["model"])
        if row["id"] in store:
            if row == store[row["id"]]:
                info["duplicate_rows"] += 1
            else:
                conflicts.add(row["id"])
        else:
            store[row["id"]] = row
    info["conflict_ids"] = len(conflicts)
    steps = sorted((r for k, r in stores["step"].items() if k not in conflicts), key=lambda r: (r["completed_at"], r["id"]))
    # 词表外且有效样本 <3 的类型并入 other：单样本类型学不出自己的水平，只会让列数膨胀
    tc: dict[str, int] = {}
    for r in steps:
        if r["w"] > 0:
            tc[r["type"]] = tc.get(r["type"], 0) + 1
    for r in steps:
        if r["type"] not in CATEGORIES and tc.get(r["type"], 0) < 3:
            r.setdefault("label", r["type"])
            r["type"] = "other"
    tasks = sorted((r for k, r in stores["task"].items() if k not in conflicts), key=lambda r: (r["completed_at"], r["id"]))
    info["step_rows"], info["task_rows"] = len(steps), len(tasks)
    return steps, tasks, info


# ---------------------------------------------------------------- 模型
def tkey(r) -> str:
    # 不可分步骤（q=1）的单位名没有含义：统一成 step，避免“文件/项/files”把同一类拆成多列
    return f"{r['type']}|{r['u'] if r['q'] != 1 else 'step'}"


def feats(r, interact=()) -> dict:
    t = tkey(r)
    lq = math.log(r["q"])
    f = {"mu0": 1.0, f"mu:{t}": 1.0, f"mod:{r['model']}": 1.0}
    if lq:
        f["beta0"] = lq
        f[f"beta:{t}"] = lq
    dd = LEVELS.index(r["d"]) - 2
    if dd:
        f["gam0"] = float(dd)
        f[f"gam:{t}"] = float(dd)
    if interact and (t, r["model"]) in interact:      # 仅对证据达标的组合启用（audit 决定并缓存）
        f[f"txm:{t}|{r['model']}"] = 1.0
    return f


def prior_of(col: str, kappa: float):
    if col in ("mu0", "r0"):
        return 0.0, PRIOR["lam_global"]
    if col == "beta0":
        return 1.0, kappa
    if col.startswith("txm:"):
        # 交互单元比主效应多一层收缩：单元样本少时自动退回"类型 + 型号"，避免过拟合
        return 0.0, max(4.0, 4 * kappa)
    return 0.0, kappa


def cholesky(A):
    """对称正定分解 A = L·Lᵀ；岭惩罚保证正定，数值失败返回 None 由高斯-约当兜底。"""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = A[i][j] - sum(Li[k] * Lj[k] for k in range(j))
            if i == j:
                if s <= 1e-12:
                    return None
                Li[i] = math.sqrt(s)
            else:
                Li[j] = s / Lj[j]
    return L


def chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def chol_inverse(L):
    """A⁻¹ = L⁻ᵀ·L⁻¹，运算量约为高斯-约当求逆的 1/4。"""
    n = len(L)
    M = [[0.0] * n for _ in range(n)]         # M = L⁻¹（下三角）
    for j in range(n):
        M[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, n):
            M[i][j] = -sum(L[i][k] * M[k][j] for k in range(j, i)) / L[i][i]
    cols = [[M[k][i] for k in range(n)] for i in range(n)]   # 第 i 列
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ci = cols[i]
        for j in range(i, n):
            cj = cols[j]
            v = sum(ci[k] * cj[k] for k in range(j, n))
            out[i][j] = out[j][i] = v
    return out


def invert(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ContractError("singular system")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                rowc = M[c]
                M[r] = [a - f * b for a, b in zip(M[r], rowc)]
    return [row[n:] for row in M]


def t80(nu: float) -> float:
    if nu <= T80[0][0]:
        return T80[0][1]
    for (n1, q1), (n2, q2) in zip(T80, T80[1:]):
        if nu <= n2:
            return q1 + (q2 - q1) * (1 / n1 - 1 / nu) / (1 / n1 - 1 / n2)
    return T80[-1][1]


class Model:
    """ln T = μ0 + μ_type + (β0 + β_type)·ln Q + (γ0 + γ_type)·(d−3) + δ_model，岭回归闭式解。"""

    def __init__(self, rows, kappa: float, extra=(), interact=(), featfn=None, target=None):
        self.interact = frozenset(interact)
        self.featfn = featfn or (lambda r: feats(r, self.interact))
        self.target = target or (lambda r: math.log(r["actual_sec"]))
        rows = [r for r in rows if r.get("w", 1.0) > 0]      # 伪迹样本（w=0）不进列也不进自由度
        cols, seen = [], set()
        for r in list(rows) + list(extra):
            for c in self._f(r):
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        self.cols, self.idx, self.kappa, self.n = cols, {c: i for i, c in enumerate(cols)}, kappa, len(rows)
        self.rows = rows
        base_w = [r.get("w", 1.0) for r in rows]
        w = list(base_w)
        # Huber IRLS（δ=1.5）：偶发十倍耗时不主导系数与 σ；无离群时与普通最小二乘一致
        for _ in range(3):
            self._solve(rows, w, full=False)
            if not rows:
                break
            resid = [self.target(r) - self.mean(self._f(r)) for r in rows]
            med = statistics.median(resid)
            mad = statistics.median([abs(x - med) for x in resid])
            scale = max(1e-6, 1.4826 * mad) if mad else math.sqrt(statistics.fmean(x * x for x in resid) or PRIOR["beta"])
            new = [bw * min(1.0, 1.5 * scale / (abs(x) + 1e-12)) for bw, x in zip(base_w, resid)]
            if max(abs(a - b) for a, b in zip(new, w)) < 1e-4:
                w = new
                break
            w = new
        self._solve(rows, w)
        self.weights = list(w)                # 最终 Huber 权重，供诊断与加权正规方程校验
        self.downweighted = sum(1 for a, b in zip(base_w, w) if b < 0.99 * a)
        self.coef_id = hashlib.sha256(canonical([VERSION, kappa, sorted(self.interact), cols, [round(t, 6) for t in self.theta]]).encode()).hexdigest()[:12]

    def _f(self, r) -> dict:
        return self.featfn(r)

    def _solve(self, rows, w, full: bool = True) -> None:
        """full=False 只解系数（Huber 中间轮用 Cholesky 求解，约为求逆的 1/3）；最终一轮才求逆算 σ² 与杠杆。"""
        p = len(self.cols)
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        for r, weight in zip(rows, w):
            x = self._f(r)
            y = self.target(r)
            items = [(self.idx[c], v) for c, v in x.items()]
            for i, xi in items:
                b[i] += weight * xi * y
                for j, xj in items:
                    A[i][j] += weight * xi * xj
        for c, i in self.idx.items():
            center, lam = prior_of(c, self.kappa)
            A[i][i] += lam
            b[i] += lam * center
        L = cholesky(A)
        if not full and L is not None:
            self.theta = chol_solve(L, b)
            return
        self.Ainv = chol_inverse(L) if L is not None else invert(A)
        self.theta = [sum(self.Ainv[i][j] * b[j] for j in range(p)) for i in range(p)]
        rss = n_eff = df = 0.0
        for r, weight in zip(rows, w):
            x = self._f(r)
            rss += weight * (self.target(r) - self.mean(x)) ** 2
            n_eff += weight
            df += weight * self.leverage(x)
        dof = max(0.0, n_eff - df)
        # Inverse-Gamma 后验均值：零数据时 = β₀/(α₀−1) = 0.16，即 σ≈0.4
        self.s2 = (2 * PRIOR["beta"] + rss) / max(1e-9, 2 * PRIOR["alpha"] + dof - 2)
        self.nu = 2 * PRIOR["alpha"] + dof

    def mean(self, x) -> float:
        # 未出现在训练列里的类型/型号按系数 0 处理：新组合不会因少一列而崩溃
        return sum(self.theta[self.idx[c]] * v for c, v in x.items() if c in self.idx)

    def leverage(self, x) -> float:
        items = [(self.idx[c], v) for c, v in x.items() if c in self.idx]
        return sum(vi * vj * self.Ainv[i][j] for i, vi in items for j, vj in items)

    def predict(self, r):
        x = self._f(r)
        return self.mean(x), self.s2 * (1 + self.leverage(x))

    def coef(self, col: str, default: float):
        return self.theta[self.idx[col]] if col in self.idx else default

    def coef_se(self, col: str) -> float:
        """系数的后验标准误（岭惩罚已计入逆矩阵）。"""
        if col not in self.idx:
            return float("inf")
        i = self.idx[col]
        return math.sqrt(max(1e-12, self.s2 * self.Ainv[i][i]))


def base_stats(steps, s2: float) -> dict:
    """agent 开工前估计的可靠性：≥10 条按类型学偏差与方差，不足用全局，再不足用先验。"""
    rb = [(tkey(r), math.log(r["actual_sec"] / r["base_sec"])) for r in steps if r.get("base_sec") and r.get("w", 1.0) > 0]
    by_type: dict[str, list[float]] = {}
    for t, x in rb:
        by_type.setdefault(t, []).append(x)
    out = dict(n=len(rb), bias=0.0, var=s2 / PRIOR["w_base"], src="prior", by_type={})
    if len(rb) >= 10:
        out.update(bias=statistics.fmean(x for _, x in rb),
                   var=max(0.01, statistics.pvariance([x for _, x in rb])), src="history")
    for t, xs in by_type.items():
        if len(xs) >= 10:
            out["by_type"][t] = dict(n=len(xs), bias=statistics.fmean(xs), var=max(0.01, statistics.pvariance(xs)))
        else:
            out["by_type"][t] = dict(n=len(xs), bias=out["bias"], var=out["var"])
    return out


def tau2_from(tasks, s2: float) -> float:
    """任务级偏移方差 τ²（σ² 的一部分）：先验 σ²/2 与任务间经验值按 n/(n+3) 加权，避免 3 个任务就跳变。
    过滤 |resid_mean| > 1.5 的历史微步/空闲污染行，避免把 τ² 顶到上限导致开工区间恒 >3×。
    """
    rows = [t for t in tasks if t["resid_k"] > 0 and t.get("status") != "abandoned" and abs(t.get("resid_mean", 0.0)) <= 1.5]
    if len(rows) < 3:
        return s2 / 2
    means = [t["resid_mean"] for t in rows]
    noise = statistics.fmean(s2 / t["resid_k"] for t in rows)
    emp = max(0.05 * s2, statistics.pvariance(means) - noise)
    w = len(rows) / (len(rows) + 3)
    return min(0.9 * s2, max(0.05 * s2, w * emp + (1 - w) * (s2 / 2)))


def step_forecast(model: Model, st: dict, plan_model: str, n_type: int, bstats: dict, tau2: float, qrange=None):
    """返回步骤预测；sd_mc 是去掉任务级共享成分后的步骤自身离散度（展示区间含全部成分）。

    qrange: {type|u: (最小 ln q, 最大 ln q)}。目标数量超出该范围时：
    - 给模型方差加 (0.3·超出量)² 的外推不确定度（幂律形式在范围外更不可信）；
    - 提高 agent 开工估计的权重（划分 1+3·超出量 倍），因为结构性知识比外推曲线可靠。
    """
    row = dict(type=st["type"], u=st["u"], q=st["q"], d=st["d"], model=plan_model)
    lq = math.log(st["q"])
    lo, hi = (qrange or {}).get(tkey(st), (lq, lq))
    excess = max(0.0, lq - hi) + max(0.0, lo - lq)
    parts = []
    if n_type > 0:
        m, var = model.predict(row)
        parts.append((m, var, "model"))
    if st.get("base_sec"):
        b = bstats["by_type"].get(tkey(st), bstats)
        var_base = max(b["var"], model.s2 / PRIOR["w_base_max"])   # 防止极小的历史方差让 base 独断
        if excess > 0:
            var_base /= 1 + 3 * excess
        parts.append((math.log(st["base_sec"]) + b["bias"], var_base, "base"))
    if not parts:
        return None
    prec = sum(1 / v for _, v, _ in parts)
    mean = sum(m / v for m, v, _ in parts) / prec
    var = 1 / prec + (0.3 * excess) ** 2      # 外推不确定度加在融合之后：两种来源都不可全信
    half = t80(model.nu) * math.sqrt(var)
    within = math.sqrt(max(0.25 * var, var - tau2))
    pred = min(30.0, max(-30.0, mean))
    return dict(ln=pred, sd_mc=within * t80(model.nu) / Z80, src="blend" if len(parts) == 2 else parts[0][2], n_type=n_type,
                extrap=round(excess, 2),
                pred_sec=math.exp(pred), range_sec=[math.exp(max(-30.0, pred - half)), math.exp(min(30.0, pred + half))])


# ---------------------------------------------------------------- 3.1 影子估计器 v4（reference.md §1.15）
def feats_v4(r) -> dict:
    """v4 只学“开工估计偏差” r = ln(actual/base)：全局 + 型号族 + 类别 + 复杂度 + 数量，全部向 0 收缩。"""
    f = {"r0": 1.0, f"rfam:{r['model']}": 1.0, f"rcat:{r['type']}": 1.0}
    dd = LEVELS.index(r["d"]) - 2
    if dd:
        f["rgam"] = float(dd)
    lq = math.log(r["q"])
    if lq:
        f["rbeta"] = lq
    return f


class ShadowV4:
    """点估计 = ln base + 偏差回归 + 留一残差中位数；区间 = 留一残差的经验 P20/P80（同型号族 ≥30 条用本族，否则全局）。

    留一残差 e/(1−h) 是岭回归的闭式交叉验证残差，不需要逐条重拟合；经验分位不假设误差独立或正态——
    v3 把模型与开工估计按精度加权、假设两路误差独立，真实数据上区间覆盖只有 43%。
    """

    def __init__(self, steps, kappa: float):
        rows = [r for r in steps if r.get("base_sec") and r.get("w", 1.0) > 0]
        self.ok = len(rows) >= SHADOW_MIN_POOL
        self.pool: dict[str, list[float]] = {}
        if not self.ok:
            return
        self.m = Model(rows, kappa, featfn=feats_v4, target=self._y)
        for r, w in zip(self.m.rows, self.m.weights):
            x = feats_v4(r)
            h = min(0.95, w * self.m.leverage(x))
            e = (self._y(r) - self.m.mean(x)) / (1 - h)
            self.pool.setdefault(r["model"], []).append(e)
            self.pool.setdefault("*", []).append(e)
        self.pool = {k: v[-SHADOW_WINDOW:] for k, v in self.pool.items()}

    @staticmethod
    def _y(r) -> float:
        return math.log(r["actual_sec"] / r["base_sec"])

    def forecast(self, st: dict, family: str):
        if not self.ok or not st.get("base_sec"):
            return None
        pool = self.pool.get(family, [])
        pool = sorted(pool if len(pool) >= SHADOW_FAMILY_POOL else self.pool["*"])
        q = lambda p: pool[min(len(pool) - 1, int(p * len(pool)))]
        row = dict(type=st["type"], q=st["q"], d=st["d"], model=family)
        c = math.log(st["base_sec"]) + self.m.mean(feats_v4(row))
        ln, lo, hi = (min(30.0, max(-30.0, c + q(p))) for p in (0.5, 0.2, 0.8))
        return dict(ln=ln, lo=lo, hi=hi, pred_v4=math.exp(ln), range_v4=[math.exp(lo), math.exp(hi)])


def estimator() -> str:
    return "v4" if os.environ.get("COMPLETION_PROGRESS_ESTIMATOR", "").strip().lower() == "v4" else "v3"


def calib_width(steps, window: int = 200):
    """经验区间宽度：最近 window 条已发出预测的对数残差，去中位数后的 P20/P80（不足 30 条返回 None）。

    这些预测都是开工前冻结的，残差是真正的样本外误差（split-conformal）。只取宽度不取中心：
    点估计的偏差由回归负责，旧版本的系统偏差不会被带进新区间；样本随使用自动滚动更新。
    """
    rs = [math.log(r["actual_sec"] / r["pred_sec"]) for r in steps if r.get("w", 1.0) > 0 and r.get("pred_sec")][-window:]
    if len(rs) < 30:
        return None
    med = statistics.median(rs)
    c = sorted(x - med for x in rs)
    return c[int(0.2 * len(c))], c[min(len(c) - 1, int(0.8 * len(c)))]


def apply_calib(fc: dict, width, tau2: float) -> dict:
    """把 v3 的参数化区间换成经验宽度；外推步骤保留更宽者（范围外更不确定）。
    sd_mc 保留融合 base 后的后验步内标准差上限，避免历史粗粒度残差把已给 base_min 的步骤全部顶成 >3×（“至少 N 分钟”）。
    """
    if not fc or not width or fc.get("ln") is None:
        return fc
    lo, hi = fc["ln"] + width[0], fc["ln"] + width[1]
    if fc.get("extrap"):
        lo = min(lo, math.log(fc["range_sec"][0]))
        hi = max(hi, math.log(fc["range_sec"][1]))
    var = ((hi - lo) / (2 * Z80)) ** 2
    sd_emp = math.sqrt(max(0.25 * var, var - tau2))
    sd_mc = max(fc["sd_mc"], sd_emp) if fc.get("extrap") else min(fc["sd_mc"], sd_emp)
    return dict(fc, range_sec=[math.exp(lo), math.exp(hi)], sd_mc=sd_mc, calib=True)


# ---------------------------------------------------------------- 状态
def state_dir(arg) -> Path:
    if arg:
        return Path(arg).expanduser()
    env = os.environ.get("COMPLETION_PROGRESS_STATE")
    if env:
        return Path(env).expanduser()
    root = os.environ.get("XDG_STATE_HOME")
    return (Path(root).expanduser() if root else Path.home() / ".local" / "state") / "completion-progress"


def task_path(d: Path, session: str, task: str) -> Path:
    return d / "tasks" / (hashlib.sha1(f"{session}\n{task}".encode()).hexdigest()[:16] + ".json")


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def append_line(path: Path, obj, warns: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(canonical(obj) + "\n")
    except OSError as exc:
        warns.append(f"history_not_written:{type(exc).__name__}")


def load_state(d: Path, session: str, task: str) -> dict:
    p = task_path(d, session, task)
    try:
        st = parse_json(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"no task state for {session}/{task}; run plan first")
    if st.get("v") != 3:
        raise ContractError("task state version mismatch")
    return st


def load_kappa(d: Path):
    """读 audit 缓存：κ 与已启用的 type×model 交互；缓存缺失/损坏则用先验。"""
    try:
        obj = parse_json((d / "model.json").read_text(encoding="utf-8"))
        kappa = float(obj["kappa"])
        interact = {(a, b) for a, b in obj.get("interact", []) if isinstance(a, str) and isinstance(b, str)}
        return kappa, obj, interact
    except (OSError, ValueError, KeyError, TypeError):
        return PRIOR["kappa"], None, set()


# ---------------------------------------------------------------- 剩余时间
def theta_of(state) -> tuple[float, float]:
    th = state["theta"]
    return th["m"], th["v"]


def running_remaining(st: dict, now: float, th_m: float, th_v: float):
    """进行中步骤的剩余分布 (ln 中位, sd, 方法, 已用)。"""
    e = max(0.0, now - st["started_at"])
    units = st.get("units", 0) or 0
    if units > 0 and st["q"] > 1 and units < st["q"]:
        # 批量：实测速率与模型速率在对数空间按 1:k 混合（reference §1.6）
        prior_rate = math.exp(st["ln"] + th_m) / st["q"]
        obs_rate = max(1e-6, (e - (st.get("setup_sec") or 0.0)) / units)
        ln_rate = (math.log(prior_rate) + units * math.log(obs_rate)) / (1 + units)
        rem = (st["q"] - units) * math.exp(ln_rate)
        return math.log(max(rem, 1e-6)), st["sd_mc"] / math.sqrt(1 + units), "rate", e
    # 不可分：对数正态条件剩余 P(T−e ≤ r | T > e)
    m, s = st["ln"] + th_m, math.sqrt(st["sd_mc"] ** 2 + th_v)
    fe = ND.cdf((math.log(e) - m) / s) if e > 0 else 0.0
    if fe > 0.95:
        # 已超出模型 P95：条件尾部会给出接近 0 的剩余，不可信。按"重新开始一段同等工作"给保守区间并标记超时
        return m, max(s, 0.5), "overrun", e
    med = math.exp(m + s * ND.inv_cdf(fe + 0.5 * (1 - fe))) - e
    lo = math.exp(m + s * ND.inv_cdf(fe + 0.2 * (1 - fe))) - e
    hi = math.exp(m + s * ND.inv_cdf(fe + 0.8 * (1 - fe))) - e
    sd = max(0.05, (math.log(max(hi, 1e-6)) - math.log(max(lo, 1e-6))) / (2 * Z80))
    return math.log(max(med, 1e-6)), sd, "conditional", e


def simulate(state: dict, now: float, draws: int = MC_DRAWS):
    """按依赖与并发槽位排程的蒙特卡洛，返回整体剩余 (P20, P50, P80) 与各步点估计。"""
    th_m, th_v = theta_of(state)
    steps = state["steps"]
    slots = state.get("slots", 1)
    per_step = {}
    running = {}
    for st in steps:
        if st.get("ln") is None:
            continue                     # 无法估计的步骤不进 ETA
        if st["status"] == "running":
            running[st["id"]] = running_remaining(st, now, th_m, th_v)
            per_step[st["id"]] = dict(rem_sec=math.exp(running[st["id"]][0]), method=running[st["id"]][2], elapsed_sec=running[st["id"]][3],
                                      range_sec=[math.exp(running[st["id"]][0] - Z80 * running[st["id"]][1]), math.exp(running[st["id"]][0] + Z80 * running[st["id"]][1])])
        elif st["status"] == "pending":
            per_step[st["id"]] = dict(rem_sec=math.exp(st["ln"] + th_m), method="plan")
    rng = random.Random(20261017)
    totals = []
    for _ in range(draws):
        th = rng.gauss(th_m, math.sqrt(th_v)) if th_v > 0 else th_m
        finish, res_free = {}, {}
        slot_free = [0.0] * max(1, slots)
        for st in steps:
            sid = st["id"]
            if st["status"] == "done" or st.get("ln") is None:
                finish[sid] = max([finish.get(dep, 0.0) for dep in st["dep"]] + [0.0]) if st.get("ln") is None else 0.0
                continue
            if sid in running:
                ln_med, sd, _, _ = running[sid]
                fin = math.exp(ln_med + sd * rng.gauss(0, 1))
                j = min(range(len(slot_free)), key=slot_free.__getitem__)
                slot_free[j] = max(slot_free[j], fin)
            else:
                dur = math.exp(st["ln"] + th + st["sd_mc"] * rng.gauss(0, 1))
                start = max([finish.get(dep, 0.0) for dep in st["dep"]] + [res_free.get(st.get("res") or "", 0.0)])
                j = min(range(len(slot_free)), key=slot_free.__getitem__)
                start = max(start, slot_free[j])
                fin = start + dur
                slot_free[j] = fin
            if st.get("res"):
                res_free[st["res"]] = fin
            finish[sid] = fin
        totals.append(max(finish.values()) if finish else 0.0)
    totals.sort()
    q = lambda p: totals[min(len(totals) - 1, int(p * len(totals)))]
    return (q(.2), q(.5), q(.8)), per_step


def progress_percent(state: dict) -> int | None:
    if any(st["pred_sec"] is None for st in state["steps"]):
        return None                      # 有步骤连相对权重都没有：只报计数，不造百分比
    total = sum(st["pred_sec"] for st in state["steps"])
    if total <= 0:
        return None
    got = 0.0
    for st in state["steps"]:
        if st["status"] == "done":
            got += st["pred_sec"]
        elif st["status"] == "running" and st["q"] > 1 and st.get("units"):
            got += st["pred_sec"] * min(1.0, st["units"] / st["q"])
    return min(99, round(100 * got / total))


# ---------------------------------------------------------------- 文案
_TZ: dict[str, Any] = {"label": None, "known": False, "off": None}
_TZ_NAMES = {8: "北京时间", 9: "东京时间", 0: "UTC", 1: "中欧时间", -5: "美东时间", -8: "美西时间"}
_TZ_NAMES_EN = {8: "Beijing time", 9: "Tokyo time", 0: "UTC", 1: "CET", -5: "US Eastern", -8: "US Pacific"}
_LANG = {"code": "zh"}


def resolve_lang(arg: str | None) -> None:
    """--lang > $COMPLETION_PROGRESS_LANG > zh。影响进度段文案，不影响 JSON 字段名与错误。"""
    name = (arg or os.environ.get("COMPLETION_PROGRESS_LANG") or "zh").strip().lower()
    _LANG["code"] = "en" if name.startswith("en") else "zh"


def en() -> bool:
    return _LANG["code"] == "en"


def resolve_tz(arg: str | None) -> None:
    """--tz > $COMPLETION_PROGRESS_TZ > $TZ > 系统本地偏移推断；都不可靠就不报绝对时刻。"""
    name = arg or os.environ.get("COMPLETION_PROGRESS_TZ") or os.environ.get("TZ")
    if name:
        os.environ["TZ"] = name
        if hasattr(time, "tzset"):
            time.tzset()
    off = round(-(time.altzone if time.localtime().tm_isdst > 0 else time.timezone) / 3600 * 2) / 2   # 保留半小时时区
    _TZ["off"] = off
    if name in ("Asia/Shanghai", "Asia/Chongqing", "Asia/Harbin", "PRC"):
        _TZ.update(label=_TZ_NAMES[8], known=True)
    elif name:
        _TZ.update(label=_TZ_NAMES.get(off, f"UTC{off:+g}"), known=True)
    elif off != 0 or time.strftime("%Z") not in ("", "UTC", "GMT"):
        _TZ.update(label=_TZ_NAMES.get(off, f"UTC{off:+g}"), known=True)   # 系统有明确非 UTC 偏移
    else:
        _TZ.update(label=None, known=False)   # 沙箱默认 UTC 且无声明：时刻不可信


def tz_label() -> str:
    if _TZ["label"] is None:
        return "local time" if en() else "本地时间"
    if en():
        return _TZ_NAMES_EN.get(_TZ["off"], f"UTC{_TZ['off']:+g}")
    return _TZ["label"]


def clock(t: float, now: float) -> str:
    lt, ln = time.localtime(t), time.localtime(now)
    s = time.strftime("%H:%M", lt)
    if (lt.tm_year, lt.tm_yday) != (ln.tm_year, ln.tm_yday):
        s = time.strftime("%m-%d ", lt) + s
    return s


def unit_for(hi_sec: float) -> int:
    m = hi_sec / 60
    return 1 if m < 30 else 5 if m < 90 else 30


def round_sec(sec: float, unit: int) -> float:
    """按展示单位取整后的秒数，时长与时刻共用。"""
    m = sec / 60
    if unit >= 30 and m >= 30:
        return round(m / 30) * 30 * 60
    if m < 1:
        return 60.0
    step = unit if m >= unit else 1
    return max(1, round(m / step) * step) * 60.0


def fmt_dur(sec: float, unit: int) -> str:
    if sec < 60:
        return "不到 1 分钟"
    r = round_sec(sec, unit)
    if unit >= 30 and r >= 1800:
        return f"{r / 3600:g} 小时"
    return f"{r / 60:.0f} 分钟"


def fmt_range(lo: float, hi: float, unit: int) -> str:
    if hi < 60:
        return "不到 1 分钟"
    a, b = fmt_dur(lo, unit), fmt_dur(hi, unit)
    if a == b:
        return "约 " + a
    if a.startswith("不到"):
        return f"不到 {b}"
    return "约 " + a.split(" ")[0] + "–" + b


def remaining_text(state: dict, now: float, totals, with_tz: bool, narrow_after: str | None) -> str:
    lo, _, hi = totals
    unit = state["unit_min"]
    if hi < 60:
        return "不到 1 分钟"
    if hi > 3 * lo and lo >= 60:
        s = f"至少 {fmt_dur(lo, unit)}"
        return s + (f"，完成「{narrow_after}」后收窄" if narrow_after else "")
    s = fmt_range(lo, hi, unit)
    if not _TZ["known"]:
        return s
    return f"{s}，{eta_text(now, round_sec(lo, unit), round_sec(hi, unit), with_tz)}"


def fmt_elapsed(sec: float, exact: bool = False) -> str:
    if sec < 30 and not exact:
        return "刚开始"
    if sec < 60:
        return f"{max(1, round(sec))} 秒"
    if sec < 3600:
        m, s = divmod(round(sec), 60)
        return f"{m} 分 {s:02d} 秒" if sec < 600 and s else f"{m} 分钟"
    h, m = divmod(round(sec / 60), 60)
    return f"{h} 小时 {m} 分钟" if m else f"{h} 小时"


def eta_text(now: float, lo: float, hi: float, with_tz: bool) -> str:
    a, b = clock(now + lo, now), clock(now + hi, now)
    s = a if a == b else f"{a}–{b}"
    if en():
        return f"{s} ({tz_label()})" if with_tz else s
    return f"{s}（{tz_label()}）完成" if with_tz else f"{s} 完成"


# ---- 英文文案（--lang en）
def fmt_dur_en(sec: float, unit: int) -> str:
    if sec < 60:
        return "under 1 min"
    r = round_sec(sec, unit)
    if unit >= 30 and r >= 1800:
        return f"{r / 3600:g} h"
    return f"{r / 60:.0f} min"


def fmt_range_en(lo: float, hi: float, unit: int) -> str:
    if hi < 60:
        return "under 1 min"
    a, b = fmt_dur_en(lo, unit), fmt_dur_en(hi, unit)
    if a == b:
        return f"~{a}"
    return f"~{a.split(' ')[0]}–{b}"


def fmt_elapsed_en(sec: float, exact: bool = False) -> str:
    if sec < 30 and not exact:
        return "just started"
    if sec < 60:
        return f"{max(1, round(sec))} s"
    if sec < 3600:
        m, s = divmod(round(sec), 60)
        return f"{m} min {s:02d} s" if sec < 600 and s else f"{m} min"
    h, m = divmod(round(sec / 60), 60)
    return f"{h} h {m} min" if m else f"{h} h"


def remaining_text_en(state: dict, now: float, totals, with_tz: bool, narrow_after: str | None) -> str:
    lo, _, hi = totals
    unit = state["unit_min"]
    if hi < 60:
        return "under 1 min"
    if hi > 3 * lo and lo >= 60:
        s = f"at least {fmt_dur_en(lo, unit)}"
        return s + (f", narrowing after {narrow_after}" if narrow_after else "")
    s = fmt_range_en(lo, hi, unit)
    if not _TZ["known"]:
        return s
    return f"{s}, due {eta_text(now, round_sec(lo, unit), round_sec(hi, unit), with_tz)}"


def append_note(line: str, note: str) -> str:
    """把附注插到进度段末尾标记之前（中英文标记结尾不同）。"""
    if line.endswith("】**"):
        return line[:-3] + note + "】**"
    if line.endswith("]"):
        return line[:-1] + note + "]"
    return line + note


def names(state: dict, status=None) -> list[str]:
    return [st["name"] for st in state["steps"] if status is None or st["status"] == status]


def plan_line(state: dict, now: float, totals, elapsed: float) -> str:
    chain = [st["name"] for st in state["steps"]]
    if len(chain) <= 4:
        chain_s = " → ".join(chain)
    else:
        chain_s = " → ".join(chain[:3]) + " → …"
    cur = names(state, "running") or names(state, "pending")
    if en():
        cur_s = ", ".join(cur[:2]) if cur else "wrapping up"
        rem = remaining_text_en(state, now, totals, True, cur_s)
        return (f"> **[Plan: {len(chain)} steps: {chain_s} | now: {cur_s} | "
                f"elapsed: {fmt_elapsed_en(elapsed)} | remaining: {rem}]**")
    cur_s = "、".join(cur[:2]) if cur else "收尾"
    rem = remaining_text(state, now, totals, True, cur_s)
    return f"> **【进度分析：共 {len(chain)} 步：{chain_s}｜正在{cur_s}｜已用：{fmt_elapsed(elapsed)} 预计：还剩{rem}】**"


def progress_line(state: dict, now: float, totals, per_step, elapsed: float, pct, change_note: str | None) -> str:
    done = names(state, "done")
    cur = [st for st in state["steps"] if st["status"] == "running"] or [st for st in state["steps"] if st["status"] == "pending"][:1]
    if en():
        done_s = f"{len(done)}/{len(state['steps'])} steps done" + (f" (last: {done[-1]})" if done else "")
        if pct is not None:
            done_s = f"~{pct}%, " + done_s
        if cur:
            st = cur[0]
            ps = per_step.get(st["id"], {})
            cur_s = f"now: {st['name']}"
            if st["status"] == "running" and st["q"] > 1 and st.get("units"):
                cur_s += f" ({st['units']:g}/{st['q']:g})"
            if ps.get("method") == "overrun":
                over = ps["elapsed_sec"] - st["pred_sec"] * math.exp(state["theta"]["m"])
                cur_s += f", over by ~{fmt_dur_en(max(60.0, over), 1)}, re-estimating"
            narrow = st["name"]
        else:
            cur_s, narrow = "finishing final checks", None
        if change_note:
            cur_s += f"; {change_note}"
        rem = remaining_text_en(state, now, totals, False, narrow)
        return (f"> **[Progress: {done_s} | {cur_s} | "
                f"elapsed: {fmt_elapsed_en(elapsed)} | remaining: {rem}]**")
    done_s = f"已完成 {len(done)}/{len(state['steps'])} 步" + (f"（最近：{done[-1]}）" if done else "")
    if pct is not None:
        done_s = f"约 {pct}%，" + done_s
    if cur:
        st = cur[0]
        ps = per_step.get(st["id"], {})
        cur_s = f"正在{st['name']}"
        if st["status"] == "running" and st["q"] > 1 and st.get("units"):
            cur_s += f"（{st['units']:g}/{st['q']:g}）"
        if ps.get("method") == "overrun":
            over = ps["elapsed_sec"] - st["pred_sec"] * math.exp(state["theta"]["m"])
            cur_s += f"，已超出预计约 {fmt_dur(max(60.0, over), 1)}，正在重估"
        narrow = st["name"]
    else:
        cur_s, narrow = "主体已完成，正在做最后确认", None
    if change_note:
        cur_s += f"；{change_note}"
    rem = remaining_text(state, now, totals, False, narrow)
    return f"> **【进度汇报：{done_s}｜{cur_s}｜已用：{fmt_elapsed(elapsed)} 预计：还剩{rem}】**"


def deliver_line(state: dict, now: float, elapsed: float) -> str:
    first = state["plan_history"][0]
    if en():
        parts = [f"{st['name']} {fmt_elapsed_en(st['actual_sec'], exact=True)}" for st in state["steps"]]
        planned = ""
        if first.get("total"):
            planned = f", planned {fmt_range_en(first['total'][0], first['total'][2], state['unit_min']).lstrip('~')}"
        when = f"delivered {clock(now, now)} ({tz_label()})" if _TZ["known"] else "delivered"
        return (f"> **[Progress: all {len(parts)}/{len(parts)} steps done, {when} | "
                f"steps: {', '.join(parts)}{planned} | total: {fmt_elapsed_en(elapsed, exact=True)}]**")
    parts = [f"{st['name']} {fmt_elapsed(st['actual_sec'], exact=True)}" for st in state["steps"]]
    planned = ""
    if first.get("total"):
        planned = f"，开工时预计 {fmt_range(first['total'][0], first['total'][2], state['unit_min']).replace('约 ', '')}"
    when = f"{clock(now, now)} 交付（{tz_label()}）" if _TZ["known"] else "已交付"
    return (f"> **【进度汇报：{len(parts)}/{len(parts)} 步全部完成，{when}｜各步：{'、'.join(parts)}{planned}"
            f"｜用时：{fmt_elapsed(elapsed, exact=True)}】**")


def decide_report(state: dict, now: float, rem_mid: float, ev: str):
    """顺序：基线/交付/阻塞 总是报 → 预计明显变化 总是报 → 步骤完成/重划范围 报（防连击）→ 周期轮询受尾部与冷却约束。"""
    last = state.get("last_report")
    if ev in ("plan", "deliver", "block"):
        return True, None
    note = None
    if last:
        expected = max(0.0, last["rem"] - (now - last["t"]))
        delta = rem_mid - expected
        if abs(delta) >= REPORT_MIN_CHANGE and abs(delta) > REPORT_REL_CHANGE * max(last["rem"], 1.0):
            if en():
                note = f"{fmt_dur_en(abs(delta), state['unit_min'])} {'later' if delta > 0 else 'earlier'} than last estimate"
            else:
                note = f"比上次预计{'晚' if delta > 0 else '早'}约 {fmt_dur(abs(delta), state['unit_min']).replace('约 ', '')}"
    if note:
        return True, note
    if ev in ("done", "rescope"):
        # 实质步骤完成或范围变化：只要还有未完步骤（rem_mid > 0）且非同秒连发，均允许汇报进度
        if rem_mid <= 0 or (last and now - last["t"] < 5):
            return False, None
        return True, None
    if rem_mid < TAIL_SEC:
        return False, None
    if last and now - last["t"] < REPORT_GAP:
        return False, None
    return False, None


# ---------------------------------------------------------------- 命令
def parse_steps(raw: str) -> list[dict]:
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    data = parse_json(raw)
    if not isinstance(data, list) or not data:
        raise ContractError("steps must be a nonempty JSON array")
    out, ids = [], set()
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            raise ContractError("each step must be an object")
        sid = text(s.get("id", f"s{i + 1}"), "step.id")
        if sid in ids:
            raise ContractError(f"duplicate step id {sid}")
        ids.add(sid)
        raw_type = text(s.get("type"), "step.type")
        st = dict(id=sid, name=text(s.get("name", sid), "step.name"), type=norm_type(raw_type),
                  u=text(s.get("u", "step"), "step.u"), q=number(s.get("q", 1), "step.q", positive=True),
                  d=choice(s.get("d", "D3"), LEVELS, "step.d"))
        if st["type"] != raw_type.strip().lower():
            st["label"] = raw_type[:60]
        base = s.get("base_sec")
        if base is None and s.get("base_min") is not None:
            base = number(s["base_min"], "base_min", positive=True) * 60
        if base is not None:
            st["base_sec"] = number(base, "base_sec", positive=True)
        dep = s.get("dep")
        if dep is None:
            dep = [out[-1]["id"]] if out else []
        if not isinstance(dep, list) or any(d not in ids or d == sid for d in dep):
            raise ContractError(f"step {sid}: dep must list earlier step ids")
        st["dep"] = dep
        if s.get("res") is not None:
            st["res"] = text(s["res"], "step.res")
        if s.get("setup_sec") is not None:
            st["setup_sec"] = number(s["setup_sec"], "setup_sec")
        out.append(st)
    return out


def cmd_plan(a) -> dict:
    now = a.at if a.at is not None else time.time()
    d = state_dir(a.state)
    warns: list[str] = []
    steps_in = parse_steps(a.steps)
    model_raw = (a.model or os.environ.get("COMPLETION_PROGRESS_MODEL") or "unknown").strip() or "unknown"
    family = norm_model(model_raw)
    if family == "unknown":
        warns.append("model_unknown:set --model or $COMPLETION_PROGRESS_MODEL")
    nonstd = sorted({s["type"] for s in steps_in if s["type"] not in CATEGORIES})
    if nonstd:
        warns.append("nonstandard_type:" + ",".join(nonstd))
    sweep_tasks(d, now, a.session, a.task, warns)
    hist_steps, hist_tasks, info = load_history(d / "history.jsonl", now, a.max_bytes, a.max_lines)
    kappa, kinfo, cached_interact = load_kappa(d)
    probe = [dict(type=s["type"], u=s["u"], q=s["q"], d=s["d"], model=family) for s in steps_in]
    # 缓存里的组合只有仍存在于历史中才启用，避免过期组合凭空加列
    have = {(tkey(r), r["model"]) for r in hist_steps}
    interact = {c for c in cached_interact if c in have}
    model = Model(hist_steps, kappa, extra=probe, interact=interact)
    bstats = base_stats(hist_steps, model.s2)
    tau2 = tau2_from(hist_tasks, model.s2)
    shadow = ShadowV4(hist_steps, kappa)
    width = calib_width(hist_steps)
    use_v4 = estimator() == "v4"
    counts: dict[str, int] = {}
    qlogs: dict[str, list[float]] = {}
    for r in hist_steps:
        if r.get("w", 1.0) <= 0:
            continue
        counts[tkey(r)] = counts.get(tkey(r), 0) + 1
        qlogs.setdefault(tkey(r), []).append(math.log(r["q"]))
    qrange = {k: (min(v), max(v)) for k, v in qlogs.items()}
    previous = None
    try:
        previous = load_state(d, a.session, a.task)
    except ContractError:
        pass
    steps, unestimable = [], []
    for s in steps_in:
        prev = next((p for p in previous["steps"] if p["id"] == s["id"]), None) if previous else None
        if prev and prev["status"] in ("done", "running"):
            keep = dict(prev)
            keep.update({k: s[k] for k in ("name", "dep", "res") if k in s})
            steps.append(keep)
            continue
        fc = step_forecast(model, s, family, counts.get(tkey(s), 0), bstats, tau2, qrange=qrange)
        fc = apply_calib(fc, width, tau2)
        f4 = shadow.forecast(s, family)
        if use_v4 and f4:
            var = ((f4["hi"] - f4["lo"]) / (2 * Z80)) ** 2
            fc = dict(ln=f4["ln"], sd_mc=math.sqrt(max(0.25 * var, var - tau2)), src="v4", n_type=counts.get(tkey(s), 0),
                      extrap=0.0, pred_sec=f4["pred_v4"], range_sec=f4["range_v4"])
        if fc is None:
            unestimable.append(s["name"])
            fc = dict(ln=None, sd_mc=None, src="none", n_type=0, pred_sec=None, range_sec=None)
        if f4:
            fc = dict(fc, pred_v4=f4["pred_v4"], range_v4=f4["range_v4"])
        steps.append(dict(s, status="pending", started_at=None, finished_at=None, actual_sec=None, units=0.0,
                          forecast_at=now, paused_sec=0.0, **fc))
    # 无法估计的步骤：进度权重取其他步骤预测的中位数；ETA 排除它（文案注明），不按 60 秒冒充
    known = [st["pred_sec"] for st in steps if st["pred_sec"]]
    for st in steps:
        if st["src"] == "none":
            st["pred_sec"] = statistics.median(known) if known else None
            st["excluded_from_eta"] = True
    theta = previous["theta"] if previous else dict(m=0.0, v=tau2, k=0, sum_r=0.0)
    if previous:
        # τ²/σ² 可能在这次 plan 里变了：保持后验均值 m，但按新参数重算后验方差
        s2w = max(0.25 * model.s2, model.s2 - tau2)
        theta["v"] = 1 / (1 / tau2 + theta["k"] / s2w)
    state = dict(v=3, session=a.session, task=a.task, goal=a.goal, model=family, model_raw=model_raw[:200], slots=max(1, a.slots),
                 created_at=previous["created_at"] if previous else now, scope_v=(previous["scope_v"] + 1) if previous else 1,
                 coef_id=model.coef_id, kappa=kappa, s2=model.s2, nu=model.nu, tau2=tau2, base=bstats, steps=steps, theta=theta,
                 plan_history=(previous["plan_history"] if previous else []), last_report=previous.get("last_report") if previous else None,
                 last_event_at=now, blocked=None, blocked_at=None, paused_total=(previous or {}).get("paused_total", 0.0))
    # 首步自动开始（串行）；并行用 --start
    start_ids = [x for x in (a.start or "").split(",") if x]
    if not start_ids and not any(st["status"] == "running" for st in steps):
        first = next((st for st in steps if st["status"] == "pending"), None)
        if first:
            start_ids = [first["id"]]
    for sid in start_ids:
        st = next((x for x in steps if x["id"] == sid), None)
        if st is None:
            raise ContractError(f"--start: unknown step {sid}")
        if st["status"] == "pending":
            st.update(status="running", started_at=now)
    totals, per_step = simulate(state, now)
    state["unit_min"] = max(unit_for(totals[2] + (now - state["created_at"])), previous["unit_min"] if previous else 1)
    state["plan_history"].append(dict(v=state["scope_v"], t=now, coef_id=model.coef_id, total=list(totals),
                                      pred=[(st["id"], st["pred_sec"]) for st in steps]))
    elapsed = now - state["created_at"]
    status = "none" if unestimable and len(unestimable) == len(steps) else "seed" if all(st["src"] in ("base", "none") for st in steps) \
        else "initial" if sum(1 for st in steps if st["n_type"] >= 5) < len(steps) else "estimated"
    line = plan_line(state, now, totals, elapsed)
    if unestimable:
        if len(unestimable) == len(steps):
            line = f"> **【进度分析：共 {len(steps)} 步：{' → '.join(names(state))}｜正在{names(state)[0]}｜已用：{fmt_elapsed(elapsed)} 预计：完成这步后预估】**"
        else:
            note = f"（不含「{'」「'.join(unestimable)}」）" if not en() else f" (excluding {', '.join(unestimable)})"
            line = append_note(line, note)
            warns.append("unestimable_steps:" + ",".join(unestimable))
    state["last_report"] = dict(t=now, rem=totals[1])
    try:
        write_atomic(task_path(d, a.session, a.task), state)
    except OSError as exc:
        warns.append(f"state_not_persisted:{type(exc).__name__}")
    if kinfo is None and len(hist_steps) >= 20:
        warns.append("run_audit_to_select_kappa")
    extrapolated = [st["name"] for st in steps if st.get("extrap")]
    if extrapolated:
        warns.append("extrapolated_steps:" + ",".join(extrapolated))
    out = dict(ok=True, ev="plan", should_report=True, line=line, warn=warns)
    if a.verbose:
        out.update(status=status, scope_v=state["scope_v"], n_hist=len(hist_steps), rem_min=[round(x / 60, 1) for x in totals],
                   eta=[clock(now + totals[0], now), clock(now + totals[2], now)], elapsed_min=round(elapsed / 60, 1))
        out["steps"] = [dict(id=st["id"], name=st["name"], status=st["status"],
                             pred_min=None if st["pred_sec"] is None else round(st["pred_sec"] / 60, 1),
                             range_min=None if not st["range_sec"] else [round(x / 60, 1) for x in st["range_sec"]],
                             src=st["src"], n_type=st["n_type"], extrap=st.get("extrap", 0)) for st in steps]
        out["diag"] = dict(coef_id=model.coef_id, kappa=kappa, sigma=round(math.sqrt(model.s2), 3), tau=round(math.sqrt(tau2), 3),
                           base=bstats, interact=sorted(f"{t}|{m}" for t, m in interact), history=info)
    return out


def unpause(state: dict, now: float) -> None:
    """解除等待：等待期从进行中步骤与任务的计时里扣除（只扣各步开工后的那段）。"""
    t0 = state.get("blocked_at")
    if t0 is None:
        return
    for st in state["steps"]:
        if st["status"] == "running" and st.get("started_at") is not None:
            st["paused_sec"] = st.get("paused_sec", 0.0) + max(0.0, now - max(t0, st["started_at"]))
    state["paused_total"] = state.get("paused_total", 0.0) + max(0.0, now - t0)
    state["blocked_at"] = None


def _finish_step(state: dict, st: dict, now: float, a, warns: list, d: Path, actual=None, timing=None) -> None:
    if st["status"] == "pending":
        st.update(status="running", started_at=state["last_event_at"])
    t0 = state.get("blocked_at")
    paused = st.get("paused_sec", 0.0) + (max(0.0, now - max(t0, st["started_at"])) if t0 is not None else 0.0)
    if a.actual is not None:
        actual, timing = a.actual, "explicit"
    elif actual is None:
        actual, timing = max(1.0, now - st["started_at"] - paused), timing or "auto"
    units = a.units if a.units is not None else st["q"]
    st.update(status="done", finished_at=now, actual_sec=actual, units=units, timing=timing)
    if a.evidence:
        st["evidence"] = a.evidence
    cause = getattr(a, "cause", None)
    if cause:
        st["cause"] = cause
    if a.setup is not None:
        st["setup_sec"] = a.setup
    # 任务级偏移后验（reference §1.5）：步骤内方差 = σ² − τ²；无法估计的步骤、计时伪迹不参与修正
    th = state["theta"]
    probe = dict(actual_sec=actual, timing=timing, cause=cause, range_sec=st.get("range_sec"), pred_sec=st.get("pred_sec"))
    if st.get("ln") is not None and sample_weight(probe) > 0:
        resid = math.log(actual) - st["ln"]
        th["k"] += 1
        th["sum_r"] += resid
        rbar = th["sum_r"] / th["k"]
        tau2 = state["tau2"]
        s2w = max(0.25 * state["s2"], state["s2"] - tau2)
        th["m"] = rbar * tau2 / (tau2 + s2w / th["k"])
        th["v"] = 1 / (1 / tau2 + th["k"] / s2w)
    sample = dict(v=3, kind="step", id=f"{state['session']}/{state['task']}/{st['id']}", session=state["session"], task=state["task"],
                  type=st["type"], u=st["u"], d=st["d"], q=st["q"], model=state["model"], actual_sec=actual, units=units,
                  started_at=st["started_at"], completed_at=now, src=st["src"], coef_id=state["coef_id"],
                  forecast_at=st.get("forecast_at") or state["plan_history"][0]["t"], timing=timing)
    if st.get("pred_sec") is not None:
        sample.update(pred_sec=st["pred_sec"], range_sec=st["range_sec"])
    if st.get("pred_v4") is not None:
        sample.update(pred_v4=st["pred_v4"], range_v4=st["range_v4"])
    for k in ("base_sec", "setup_sec", "cause", "label"):
        if st.get(k) is not None:
            sample[k] = st[k]
    if paused > 0:
        sample["paused_sec"] = paused
    if state.get("model_raw") and state["model_raw"] != state["model"]:
        sample["model_raw"] = state["model_raw"]
    if st.get("evidence"):                   # 共享历史只留位置引用，全文留在本任务状态文件
        ev = st["evidence"]
        sample["evidence"] = ev if len(ev) <= EVIDENCE_MAX else ev[:EVIDENCE_MAX - 1] + "…"
    append_line(d / "history.jsonl", sample, warns)


def cmd_progress(a) -> dict:
    now = a.at if a.at is not None else time.time()
    d = state_dir(a.state)
    warns: list[str] = []
    state = load_state(d, a.session, a.task)
    if a.actual is not None and not (isinstance(a.actual, (int, float)) and math.isfinite(a.actual) and a.actual > 0):
        raise ContractError("--actual must be a positive number of seconds")
    if a.units is not None and not (isinstance(a.units, (int, float)) and math.isfinite(a.units) and a.units >= 0):
        raise ContractError("--units must be a nonnegative number")
    by_id = {st["id"]: st for st in state["steps"]}
    ev = "status"
    for sid in [x for x in (a.start or "").split(",") if x]:
        st = by_id.get(sid)
        if st is None:
            raise ContractError(f"unknown step {sid}; re-plan to change scope")
        if st["status"] == "pending":
            st.update(status="running", started_at=now)
            ev = "start"
    if a.done:
        ids = [x for x in a.done.split(",") if x]
        for sid in ids:
            st = by_id.get(sid)
            if st is None:
                raise ContractError(f"unknown step {sid}; re-plan to change scope")
            if st["status"] == "done":
                raise ContractError(f"step {sid} already done")
        alloc: dict[str, float] = {}
        if len(ids) > 1 and a.actual is None:
            # 一次完成多步：总时长按开工预测比例拆分（标 split、拟合半权），不再各自重复计整段
            group = [by_id[x] for x in ids]
            run_starts = [st["started_at"] for st in group if st["status"] == "running" and st.get("started_at") is not None]
            t0 = min(run_starts) if run_starts else state["last_event_at"]
            bt = state.get("blocked_at")
            paused = max([st.get("paused_sec", 0.0) for st in group if st["status"] == "running"] + [0.0]) \
                + (max(0.0, now - max(bt, t0)) if bt is not None else 0.0)
            total = max(float(len(ids)), now - t0 - paused)
            wts = [st.get("pred_sec") or 1.0 for st in group]
            cursor = t0
            for st, wt in zip(group, wts):
                alloc[st["id"]] = max(1.0, total * wt / sum(wts))
                st.update(status="running", started_at=cursor, paused_sec=0.0)
                cursor += alloc[st["id"]]
            if bt is not None:                   # 已扣入拆分总时长：记入任务等待，计时起点移到现在，避免重复扣
                state["paused_total"] = state.get("paused_total", 0.0) + max(0.0, now - bt)
                state["blocked_at"] = now
        for sid in ids:
            _finish_step(state, by_id[sid], now, a, warns, d, actual=alloc.get(sid), timing="split" if alloc else None)
        ev = "done"
        if state["slots"] == 1 and not any(st["status"] == "running" for st in state["steps"]) and not a.no_auto_start:
            done_ids = {st["id"] for st in state["steps"] if st["status"] == "done"}
            nxt = next((st for st in state["steps"] if st["status"] == "pending" and all(x in done_ids for x in st["dep"])), None)
            if nxt:
                nxt.update(status="running", started_at=now)
    if a.running:
        st = by_id.get(a.running)
        if st is None:
            raise ContractError(f"unknown step {a.running}")
        if st["status"] == "pending":
            st.update(status="running", started_at=state["last_event_at"])
        if a.units is not None:
            if a.units >= st["q"]:
                raise ContractError("units reached q: use --done")
            st["units"] = a.units
        if a.setup is not None:
            st["setup_sec"] = a.setup
        ev = "running" if ev == "status" else ev
    if a.block is not None:
        if a.block:
            if state.get("blocked_at") is None:
                state["blocked_at"] = now
            state["blocked"] = a.block
            ev = "block"
        else:
            unpause(state, now)          # 等待期不计入步骤与任务用时
            state["blocked"] = None
    if ev != "status":
        state["last_event_at"] = now
    totals, per_step = simulate(state, now)
    pct = progress_percent(state)
    elapsed = now - state["created_at"]
    need_unit = unit_for(totals[2] + elapsed)
    bump_note = None
    if need_unit > state["unit_min"]:
        bump_note = ("（改按半小时估）" if need_unit == 30 else "（改用 5 分钟精度）") if not en() \
            else (" (switching to half-hour precision)" if need_unit == 30 else " (switching to 5-minute precision)")
        state["unit_min"] = need_unit
    should, note = decide_report(state, now, totals[1], ev)
    if a.ask:
        should = True
    line = progress_line(state, now, totals, per_step, elapsed, pct, note)
    if bump_note:
        line = append_note(line, bump_note)
    all_done = all(st["status"] == "done" for st in state["steps"])
    if all_done:
        if en():
            line = (f"> **[Progress: {len(state['steps'])}/{len(state['steps'])} steps done | finishing final checks | "
                    f"elapsed: {fmt_elapsed_en(elapsed)} | remaining: under {fmt_dur_en(max(60.0, totals[2]), 1)}]**")
        else:
            line = f"> **【进度汇报：{len(state['steps'])}/{len(state['steps'])} 步已完成｜主体已完成，正在做最后确认｜已用：{fmt_elapsed(elapsed)} 预计：还剩不到 {fmt_dur(max(60.0, totals[2]), 1)}】**"
    if state.get("blocked"):
        if en():
            line = (f"> **[Progress: {'~' + str(pct) + '%, ' if pct is not None else ''}waiting for your confirmation: "
                    f"{state['blocked']} | elapsed: {fmt_elapsed_en(elapsed)} | "
                    f"remaining after confirmation: {fmt_range_en(totals[0], totals[2], state['unit_min'])}]**")
        else:
            line = (f"> **【进度汇报：{'约 ' + str(pct) + '%，' if pct is not None else ''}等待您确认{state['blocked']}｜"
                    f"已用：{fmt_elapsed(elapsed)} 预计：确认后还剩{fmt_range(totals[0], totals[2], state['unit_min'])}】**")
    if should and not a.ask:
        state["last_report"] = dict(t=now, rem=totals[1])
    try:
        write_atomic(task_path(d, a.session, a.task), state)
    except OSError as exc:
        warns.append(f"state_not_persisted:{type(exc).__name__}")
    cur = [st for st in state["steps"] if st["status"] == "running"]
    out = dict(ok=True, ev=ev, should_report=should, line=line, warn=warns)
    if a.verbose:
        out.update(pct=pct, step=[st["name"] for st in cur], rem_min=[round(x / 60, 1) for x in totals],
                   eta=[clock(now + totals[0], now), clock(now + totals[2], now)], elapsed_min=round(elapsed / 60, 1))
        if cur:
            ps = per_step.get(cur[0]["id"], {})
            if "range_sec" in ps:
                out["step_rem_min"] = [round(x / 60, 1) for x in ps["range_sec"]]
                out["step_method"] = ps["method"]
        out["theta"] = dict(factor=round(math.exp(state["theta"]["m"]), 3), k=state["theta"]["k"])
        out["steps"] = [dict(id=st["id"], name=st["name"], status=st["status"],
                             pred_min=None if st["pred_sec"] is None else round(st["pred_sec"] / 60, 1),
                             actual_min=None if st["actual_sec"] is None else round(st["actual_sec"] / 60, 1),
                             rem_min=None if st["id"] not in per_step else round(per_step[st["id"]]["rem_sec"] / 60, 1)) for st in state["steps"]]
    return out


def task_row(state: dict, now: float, status: str) -> dict:
    """任务汇总行；用时扣除等待期。放弃的任务另起 id（#abandoned），之后若仍交付不会与正式汇总冲突。"""
    th = state["theta"]
    first = state["plan_history"][0]
    end = now if status == "delivered" else max(state["created_at"], state.get("last_event_at") or now)
    paused = state.get("paused_total", 0.0)
    sid = f"{state['session']}/{state['task']}" + ("" if status == "delivered" else "#abandoned")
    row = dict(v=3, kind="task", id=sid, session=state["session"], task=state["task"], model=state["model"],
               n_steps=len(state["steps"]), actual_sec=max(1.0, end - state["created_at"] - paused), started_at=state["created_at"],
               completed_at=end, resid_k=th["k"], resid_mean=(th["sum_r"] / th["k"]) if th["k"] else 0.0, status=status)
    if paused > 0:
        row["paused_sec"] = paused
    if first.get("total"):
        row["pred_sec"] = first["total"][1]
        row["range_sec"] = [first["total"][0], first["total"][2]]
    return row


def sweep_tasks(d: Path, now: float, session: str, task: str, warns: list) -> None:
    """plan 时顺带收口：超过 24 小时无事件且未交付的任务记为放弃（不进 τ²）；交付/放弃满 30 天的状态文件删除。"""
    tdir = d / "tasks"
    if not tdir.is_dir():
        return
    me = task_path(d, session, task).name
    n_ab = n_gc = 0
    for p in tdir.glob("*.json"):
        if p.name == me:
            continue
        try:
            st = parse_json(p.read_text(encoding="utf-8"))
            if st.get("v") != 3:
                continue
            closed = st.get("delivered_at") or st.get("abandoned_at")
            if closed:
                if now - closed > GC_SEC:
                    p.unlink()
                    n_gc += 1
                continue
            last = st.get("last_event_at") or st.get("created_at") or now
            if now - last > STALE_SEC and st.get("plan_history"):
                append_line(d / "history.jsonl", task_row(st, now, "abandoned"), warns)
                st["abandoned_at"] = now
                write_atomic(p, st)
                n_ab += 1
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if n_ab:
        warns.append(f"auto_abandoned:{n_ab}")
    if n_gc:
        warns.append(f"gc_task_files:{n_gc}")


def cmd_abandon(a) -> dict:
    now = a.at if a.at is not None else time.time()
    d = state_dir(a.state)
    warns: list[str] = []
    state = load_state(d, a.session, a.task)
    if state.get("delivered_at") is not None or state.get("abandoned_at") is not None:
        raise ContractError("task already closed")
    unpause(state, now)
    state["last_event_at"] = now
    append_line(d / "history.jsonl", task_row(state, now, "abandoned"), warns)
    state["abandoned_at"] = now
    write_atomic(task_path(d, a.session, a.task), state)
    return dict(ok=True, ev="abandon", should_report=False, line="", warn=warns)


def cmd_deliver(a) -> dict:
    now = a.at if a.at is not None else time.time()
    d = state_dir(a.state)
    warns: list[str] = []
    state = load_state(d, a.session, a.task)
    if state.get("delivered_at") is not None:
        raise ContractError("task already delivered; start a new task id instead of delivering twice")
    pending = [st["name"] for st in state["steps"] if st["status"] != "done"]
    if pending:
        raise ContractError("unfinished steps: " + "、".join(pending) + "; use --done or re-plan the scope")
    elapsed = now - state["created_at"]
    first = state["plan_history"][0]
    unpause(state, now)
    row = task_row(state, now, "delivered")
    append_line(d / "history.jsonl", row, warns)
    state["delivered_at"] = now
    if a.ref:
        state["delivery_ref"] = a.ref
    try:
        write_atomic(task_path(d, a.session, a.task), state)
    except OSError as exc:
        warns.append(f"state_not_persisted:{type(exc).__name__}")
    out = dict(ok=True, ev="deliver", should_report=True, line=deliver_line(state, now, elapsed), warn=warns)
    if getattr(a, "verbose", False):
        out.update(pct=100, elapsed_min=round(elapsed / 60, 1),
                   planned_min=[round(x / 60, 1) for x in first["total"]] if first.get("total") else None)
    return out


def _fit_before(rows, t: float, kappa: float, extra, limit: int = 600, interact=()):
    """回测只用该步开工前已完成、最近 limit 条样本；远期样本对当前速度参考价值低。"""
    past = [r for r in rows if r["completed_at"] <= t]
    return Model(past[-limit:], kappa, extra=extra, interact=interact)
    return Model([r for r in rows if r["completed_at"] <= t], kappa, extra=extra)


def _score(rows, pk: str, rk: str) -> dict:
    """对数误差：均值绝对值、中位偏差（正 = 实际比预计慢）、名义 60% 区间覆盖率。"""
    e = [math.log(r["actual_sec"] / r[pk]) for r in rows]
    return dict(n=len(rows), log_mae=round(statistics.fmean(abs(x) for x in e), 3), bias=round(statistics.median(e), 3),
                coverage=round(sum(r[rk][0] <= r["actual_sec"] <= r[rk][1] for r in rows) / len(rows), 2))


def cmd_audit(a) -> dict:
    now = a.at if a.at is not None else time.time()
    d = state_dir(a.state)
    steps, tasks, info = load_history(d / "history.jsonl", now, a.max_bytes, a.max_lines)
    if a.type:
        steps = [r for r in steps if r["type"] == norm_type(a.type)]
    if a.model:
        steps = [r for r in steps if r["model"] == norm_model(a.model)]
    clean = [r for r in steps if r.get("w", 1.0) > 0]       # 回测只评有效计时的样本
    out: dict[str, Any] = dict(ok=True, ev="audit", n_steps=len(steps), n_tasks=len(tasks), history=info)
    if steps:
        span = steps[-1]["completed_at"] - steps[0]["completed_at"]
        out["window_span_days"] = round(span / 86400, 1)
    if not steps:
        out["note"] = "无样本"
        print(json.dumps(out, ensure_ascii=False))
        return out
    kappa = PRIOR["kappa"]
    cv = None
    if len(steps) >= 20 and not a.type and not a.model:
        evals = clean[-a.cv_rows:]
        cv = {}
        for k in KAPPAS:
            err, cnt = 0.0, 0
            for r in evals:
                m = _fit_before(steps, r["started_at"], k, [r])
                if m.n == 0:
                    continue
                err += abs(m.predict(r)[0] - math.log(r["actual_sec"]))
                cnt += 1
            cv[str(k)] = round(err / cnt, 4) if cnt else None
        kappa = float(min((k for k in KAPPAS if cv[str(k)] is not None), key=lambda k: cv[str(k)]))
    # type×model 交互：样本 ≥20 的组合作候选，整组加列（单元效应 4 倍先验收缩）；
    # 门控用自由度已修正的 s² 比：加列后噪声方差降 ≥10% 才启用（异质 0.75、同质 0.99，分界清楚）。
    # 不逐组合筛选：2×2 设计下交互与主效应不可区分；也不用一步前瞻 CV——30 行量级波动 ±10%。
    counts_combo: dict[tuple[str, str], int] = {}
    for r in steps:
        counts_combo[(tkey(r), r["model"])] = counts_combo.get((tkey(r), r["model"]), 0) + 1
    candidates = {c for c, n in counts_combo.items() if n >= 20}
    interact = []
    combo_test = None
    if len(candidates) >= 2:
        base_model = Model(steps, kappa)
        probe = Model(steps, kappa, interact=candidates)
        coefs = {f"{t}|{m}": probe.coef(f"txm:{t}|{m}", 0.0) for t, m in candidates}
        ratio = probe.s2 / base_model.s2 if base_model.s2 > 0 else 1.0
        combo_test = dict(n_cells=len(candidates), s2_ratio=round(ratio, 3),
                          cells={k: round(v, 2) for k, v in coefs.items()}, enabled=ratio < 0.9)
        if ratio < 0.9:
            interact = sorted(candidates)
    model = Model(steps, kappa, interact=interact)
    if not a.type and not a.model:
        try:
            write_atomic(d / "model.json", dict(kappa=kappa, interact=[list(x) for x in interact], n=len(steps),
                                                chosen_at=now, cv_log_mae=cv, interaction_test=combo_test,
                                                candidates=sorted(f"{t}|{m}" for t, m in candidates)))
        except OSError as exc:
            out["warn"] = [f"model_not_written:{type(exc).__name__}"]
    bst = base_stats(steps, model.s2)
    if not a.full:
        # 默认只列有独立校准（≥10 条）的类型；其余共用全局偏差，逐条列出无信息
        bst = dict(bst, by_type={k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                 for k, v in bst["by_type"].items() if v["n"] >= 10})
        bst.update({k: round(bst[k], 3) for k in ("bias", "var")})
    out.update(kappa=kappa, kappa_src="cv" if cv else "prior", cv_log_mae=cv, sigma=round(math.sqrt(model.s2), 3),
               tau=round(math.sqrt(tau2_from(tasks, model.s2)), 3), base=bst, coef_id=model.coef_id,
               interactions=dict(enabled=[f"{t}|{m}" for t, m in interact],
                                 test=combo_test if a.full or not combo_test else {k: v for k, v in combo_test.items() if k != "cells"},
                                 factors={f"{t}|{m}": round(math.exp(model.coef(f"txm:{t}|{m}", 0.0)), 2) for t, m in interact},
                                 candidates=sorted(f"{t}|{m}" for t, m in candidates)))
    models = sorted({r["model"] for r in steps})
    out["model_speed"] = {m: dict(factor=round(math.exp(model.coef(f"mod:{m}", 0.0)), 3), n=sum(1 for r in steps if r["model"] == m)) for m in models}
    types = sorted({tkey(r) for r in steps})
    tn = {t: sum(1 for r in steps if tkey(r) == t) for t in types}
    out["types"] = {t: dict(n=tn[t],
                            beta=round(model.coef("beta0", 1.0) + model.coef(f"beta:{t}", 0.0), 3),
                            per_level=round(math.exp(model.coef("gam0", 0.0) + model.coef(f"gam:{t}", 0.0)), 3),
                            unit_sec_d3=round(math.exp(model.coef("mu0", 0.0) + model.coef(f"mu:{t}", 0.0)), 1))
                    for t in types if a.full or tn[t] >= 5}
    # 复杂度自判漂移：同档前后两段残差差（模型已吸收静态水平，漂移才是判定标准变化）
    levels = {}
    for lv in LEVELS:
        rs = [math.log(r["actual_sec"]) - model.mean(model._f(r)) for r in steps if r["d"] == lv]
        if len(rs) >= 10:
            half = len(rs) // 2
            early, late = statistics.fmean(rs[:half]), statistics.fmean(rs[half:])
            levels[lv] = dict(n=len(rs), mean_resid=round(statistics.fmean(rs), 2),
                              drift=round(late - early, 2), flag=len(rs[half:]) >= 5 and abs(late - early) > 0.3)
        elif rs:
            levels[lv] = dict(n=len(rs), mean_resid=round(statistics.fmean(rs), 2), drift=None, flag=False)
    out["complexity_residuals"] = levels
    out["huber_downweighted"] = model.downweighted
    if not a.full:
        out["history"] = {k: info[k] for k in ("file_bytes", "truncated", "invalid_rows", "conflict_ids", "step_rows", "task_rows") if k in info}
    # 幂律误设诊断：残差对 ln q 正交是拟合的必然结果，所以曲率用 (ln q)² 测。
    # 固定开销会让大 q 端被低估（正相关）；此时优先用开工估计或把批量拆小——只提示，不自动改模型。
    curvature = {}
    for t in types:
        rs = [r for r in steps if tkey(r) == t]
        if len(rs) < 20:
            continue
        xs = [math.log(r["q"]) for r in rs]
        if max(xs) - min(xs) < 0.7:
            continue
        mx = statistics.fmean(xs)
        res = [math.log(r["actual_sec"]) - model.mean(model._f(r)) for r in rs]
        sq = [(x - mx) ** 2 for x in xs]
        mres, msq = statistics.fmean(res), statistics.fmean(sq)
        num = sum((s - msq) * (e - mres) for s, e in zip(sq, res))
        den = math.sqrt(sum((s - msq) ** 2 for s in sq) * sum((e - mres) ** 2 for e in res))
        corr = num / den if den else 0.0
        tstat = corr * math.sqrt((len(rs) - 2) / max(1e-9, 1 - corr * corr))
        curvature[t] = dict(n=len(rs), corr=round(corr, 2), t=round(tstat, 1),
                            flag=abs(corr) > 0.4 and abs(tstat) > 2.5)
    out["curvature"] = curvature
    # 回测：最近 N 条，一步前瞻，只用该步开工前已完成的样本
    bt = clean[-a.cv_rows:]
    errs, cov, n_bt, bias = [], 0, 0, 0.0
    for r in bt:
        m = _fit_before(steps, r["started_at"], kappa, [r])
        if m.n == 0 or sum(1 for p in steps if p["completed_at"] <= r["started_at"] and tkey(p) == tkey(r)) == 0:
            continue
        mu, var = m.predict(r)
        half = t80(m.nu) * math.sqrt(var)
        y = math.log(r["actual_sec"])
        errs.append(abs(mu - y))
        bias += mu - y
        cov += mu - half <= y <= mu + half
        n_bt += 1
    out["backtest"] = dict(n=n_bt, log_mae=round(statistics.fmean(errs), 3) if errs else None,
                           bias=round(bias / n_bt, 3) if n_bt else None, coverage_p20_p80=round(cov / n_bt, 2) if n_bt else None)
    issued = [r for r in bt if r.get("pred_sec") and r.get("range_sec")]
    if issued:
        out["issued_forecasts"] = dict(n=len(issued),
                                       log_mae=round(statistics.fmean(abs(math.log(r["actual_sec"] / r["pred_sec"])) for r in issued), 3),
                                       coverage=round(sum(r["range_sec"][0] <= r["actual_sec"] <= r["range_sec"][1] for r in issued) / len(issued), 2))
    # 3.1 数据质量：伪迹/拆分/原因码；开工估计按型号族的偏差；v3 与影子 v4 在同一批样本上的对照（切换依据）
    causes: dict[str, int] = {}
    for r in steps:
        if r.get("cause"):
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    out["data_quality"] = dict(zero_weight=len(steps) - len(clean), split=sum(1 for r in steps if r.get("timing") == "split"),
                               paused=sum(1 for r in steps if r.get("paused_sec")), causes=causes,
                               abandoned_tasks=sum(1 for t in tasks if t.get("status") == "abandoned"))
    fam: dict[str, list[float]] = {}
    for r in clean:
        if r.get("base_sec"):
            fam.setdefault(r["model"], []).append(math.log(r["actual_sec"] / r["base_sec"]))
    out["base_bias_by_family"] = {k: dict(n=len(v), factor=round(math.exp(statistics.median(v)), 2)) for k, v in sorted(fam.items()) if len(v) >= 5}
    paired = [r for r in clean if r.get("pred_v4") and r.get("pred_sec") and r.get("range_sec")]
    if paired:
        out["shadow"] = dict(n=len(paired), v3=_score(paired, "pred_sec", "range_sec"), v4=_score(paired, "pred_v4", "range_v4"),
                             gate="v4 log_mae ≤ v3−0.05、|bias| ≤ 0.10、coverage 0.50–0.70，且 n ≥ 100 才切换")
    done_tasks = [t for t in tasks if t.get("status") != "abandoned" and t.get("pred_sec") and t.get("range_sec")]
    if done_tasks:
        out["task_forecasts"] = _score(done_tasks, "pred_sec", "range_sec")
    print(json.dumps(out, ensure_ascii=False, indent=None if a.compact else 1))
    return out


def cmd_selftest(a) -> dict:
    """10 秒内自检：不读真实状态，全在临时目录。"""
    import tempfile
    results = []

    def ok(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    with tempfile.TemporaryDirectory() as tmp:
        base = argparse.Namespace(state=tmp, at=1_760_702_400.0, tz="Asia/Shanghai", verbose=False)
        try:
            out = cmd_plan(argparse.Namespace(**vars(base), session="st", task="T1", model="selftest/high", goal="",
                                                  steps='[{"name":"甲","type":"impl","d":"D3","base_min":10},'
                                                        '{"name":"乙","type":"test","q":40,"u":"case","d":"D2","base_min":5}]',
                                                  slots=1, start=None, max_bytes=MAX_BYTES, max_lines=MAX_LINES))
            ok("plan 零历史即给数", out["ok"] and "【进度分析" in out["line"], out["line"][:40])
            prog = cmd_progress(argparse.Namespace(**vars(base), session="st", task="T1", done="s1", start=None, running=None,
                                                       units=None, actual=900.0, setup=None, evidence=None, block=None,
                                                       ask=False, no_auto_start=False))
            ok("done 写样本并修正偏移", prog["ok"] and prog["should_report"])
            hist = Path(tmp) / "history.jsonl"
            ok("history 可解析", all(parse_json(l) for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()))
            st = load_state(Path(tmp), "st", "T1")
            ok("θ 已更新", st["theta"]["k"] == 1)
            try:
                cmd_deliver(argparse.Namespace(**vars(base), session="st", task="T1", ref=None))
                ok("未完成拒绝交付", False)
            except ContractError:
                ok("未完成拒绝交付", True)
            ok("时区已解析", _TZ["known"] and tz_label() == "北京时间", tz_label())
            ok("文案无内部术语", not any(t in out["line"] for t in ("样本", "分位", "系数", "P20", "基线")))
        except Exception as exc:                     # 自检不应因单点失败中断
            ok("自检运行", False, f"{type(exc).__name__}: {exc}")
    bad = [n for n, c, _ in results if not c]
    for n, c, detail in results:
        print(("PASS " if c else "FAIL ") + n + (f"  [{detail}]" if detail else ""))
    return dict(ok=not bad, ev="selftest", passed=len(results) - len(bad), failed=bad)


def cmd_migrate(a) -> dict:
    """v2 样本 → v3：d 三档映射到五档（D1→D1、D2→D3、D3→D5），model 取 env 第一段。"""
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        raise ContractError("destination exists; refuse to overwrite")
    dmap = {"D1": "D1", "D2": "D3", "D3": "D5", None: "D3"}
    n_in = n_out = 0
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_in += 1
        try:
            r = parse_json(line)
        except ValueError:
            continue
        if r.get("v") != 2 or not r.get("actual_sec"):
            continue
        env = r.get("env", "")
        row = dict(v=3, kind="step", id=r["id"], session=r["session"], task=r["task"], type=r["type"], u=r["u"],
                   d=dmap.get(r.get("d"), "D3"), q=r["q"], model=str(env).split("|")[0] or "unknown", actual_sec=r["actual_sec"],
                   started_at=r["started_at"], completed_at=r["completed_at"], env=dict(legacy=env))
        for k in ("pred_sec", "range_sec", "base_sec", "setup_sec", "forecast_at", "evidence"):
            if r.get(k) is not None:
                row[k] = r[k]
        try:
            validate_step(row)
        except ContractError:
            continue
        lines.append(canonical(row))
        n_out += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    out = dict(ok=True, ev="migrate", read=n_in, written=n_out, dst=str(dst))
    print(json.dumps(out, ensure_ascii=False))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, task=True):
        p.add_argument("--state", help="状态目录")
        p.add_argument("--at", type=float, help="以此 epoch 秒代替当前时间（测试/补录）")
        p.add_argument("--tz", help="时区，如 Asia/Shanghai；不填按 $COMPLETION_PROGRESS_TZ/$TZ/系统推断")
        p.add_argument("--lang", choices=["zh", "en"], help="进度段语言，默认 zh；也可用 $COMPLETION_PROGRESS_LANG")
        p.add_argument("--verbose", action="store_true")
        if task:
            p.add_argument("--session", required=True, help="宿主名-稳定会话ID")
            p.add_argument("--task", required=True)

    p = sub.add_parser("plan", help="建立或重划基线：读历史 → 拟合 → 各步预测 → 写状态")
    common(p)
    p.add_argument("--model", help="执行模型 型号/推理强度，如 claude-opus-5/high；缺省读 $COMPLETION_PROGRESS_MODEL，按型号族归一")
    p.add_argument("--goal", default="")
    p.add_argument("--steps", required=True, help='JSON 数组、@文件 或 -（stdin）；元素 {id?,name,type,q?,u?,d?,base_sec?|base_min?,dep?,res?}')
    p.add_argument("--slots", type=int, default=1, help="可同时执行的步骤数，默认 1")
    p.add_argument("--start", help="立即开始的步骤 id，逗号分隔；默认首个待办")
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    p.add_argument("--max-lines", type=int, default=MAX_LINES)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("progress", help="推进：--done/--start/--running；无参数即查询")
    common(p)
    p.add_argument("--done", help="已完成步骤 id，逗号分隔")
    p.add_argument("--start", help="开始的步骤 id，逗号分隔")
    p.add_argument("--running", help="更新进行中步骤")
    p.add_argument("--units", type=float, help="已完成数量（批量步骤）")
    p.add_argument("--actual", type=float, help="实测用时秒，覆盖自动计时")
    p.add_argument("--setup", type=float, help="已测准备开销秒")
    p.add_argument("--evidence", help="完成证据位置；共享历史只存前 40 字符")
    p.add_argument("--cause", choices=CAUSES, help="明显超出预计时的原因：env/bug/scope/wait/rework/unclear/none")
    p.add_argument("--block", nargs="?", const="", help="等待用户确认的事项；不带值表示解除")
    p.add_argument("--ask", action="store_true", help="用户主动询问：总是返回文案，不计入汇报节奏")
    p.add_argument("--no-auto-start", action="store_true")
    p.set_defaults(fn=cmd_progress)

    p = sub.add_parser("status", help="等价于不带推进参数的 progress --ask")
    common(p)
    p.set_defaults(fn=lambda a: cmd_progress(argparse.Namespace(**vars(a), done=None, start=None, running=None, units=None, actual=None,
                                                                 setup=None, evidence=None, cause=None, block=None, ask=True, no_auto_start=True)))

    p = sub.add_parser("abandon", help="放弃任务：写放弃汇总（不进任务级偏移），之后不再计入")
    common(p)
    p.set_defaults(fn=cmd_abandon)

    p = sub.add_parser("deliver", help="交付：全部步骤完成后写任务汇总")
    common(p)
    p.add_argument("--ref", help="交付物位置")
    p.set_defaults(fn=cmd_deliver)

    p = sub.add_parser("audit", help="系数表、κ 选择、滚动回测、开工前估计可靠性")
    common(p, task=False)
    p.add_argument("--type")
    p.add_argument("--model")
    p.add_argument("--cv-rows", type=int, default=40)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--full", action="store_true", help="列出全部类型、各类型开工偏差与交互单元明细（默认只列有足够样本者）")
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    p.add_argument("--max-lines", type=int, default=MAX_LINES)
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("selftest", help="10 秒自检（临时目录，不碰真实状态）")
    common(p, task=False)
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("migrate", help="v2 history → v3（一次性，不覆盖）")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(fn=cmd_migrate)

    a = ap.parse_args(argv)
    try:
        resolve_lang(getattr(a, "lang", None))
        resolve_tz(getattr(a, "tz", None))
        out = a.fn(a)
    except ContractError as exc:
        msg = dict(ok=False, error=str(exc), hint="先跑 plan 建立任务；字段错误见 reference.md §2")
        print(json.dumps(msg, ensure_ascii=False), file=sys.stderr)
        if a.cmd not in ("audit", "migrate"):
            print(json.dumps(msg, ensure_ascii=False))   # 搭车 2>/dev/null 时 agent 仍能读到原因
        return 2
    except (ValueError, TypeError) as exc:
        # JSON 解析、类型错误等输入问题：同样给出可读原因，不抛栈
        msg = dict(ok=False, error=f"{type(exc).__name__}: {exc}"[:300], hint="检查 --steps 是否为合法 JSON；字段见 reference.md §2")
        print(json.dumps(msg, ensure_ascii=False), file=sys.stderr)
        if a.cmd not in ("audit", "migrate"):
            print(json.dumps(msg, ensure_ascii=False))
        return 2
    if a.cmd not in ("audit", "migrate"):
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
