# Completion Progress（一休）v3.1

> **让 AI Agent 执行长任务时告别“黑盒等待”与“拍脑袋估时”。**
> 纯 Python 3 标准库实现 · 零第三方依赖 · 跨 Agent 共享历史建模 · 开工即出预计 · 执行中实测动态修正

---

## 1. 它解决什么问题？

当 AI Agent 执行超过 3 分钟的复杂任务（多文件重构、全链路排查、批量迁移、多轮测试）时，常见问题有两类：
1. **过程黑盒**：不知道做好了什么、正在做什么、卡没卡住、还要等多久。
2. **拍脑袋瞎猜**：让大模型直接口头估计“还需要 5 分钟”，往往严重乐观或悲观，且执行中途慢了也不会自我修正。

**Completion Progress（一休）** 将估时从“大模型口头猜测”变为**“结构化拆步 + 历史统计校准 + 实测动态修正”**。Agent 只负责在关键节点上报客观事实，所有时间预测、概率区间与文案格式均由本地 `estimate.py` 脚本计算生成，并在回复末尾附带一行清爽的进度提示：

```markdown
> **【进度分析：共 4 步：定位根因 → 修复实现 → 回归验证 → 文档交付｜正在定位根因｜已用：刚开始 预计：还剩约 15–25 分钟，14:20–14:30（北京时间）完成】**

> **【进度汇报：约 68%，已完成 2/4 步（最近：修复实现）｜正在回归验证（15/40）｜已用：14 分钟 预计：还剩约 5–8 分钟，14:24–14:27 完成】**

> **【进度汇报：约 68%，等待您确认数据库迁移方案｜已用：14 分钟 预计：确认后还剩约 5–8 分钟】**

> **【进度汇报：4/4 步全部完成，14:26 交付（北京时间）｜各步：定位根因 4 分钟、修复实现 10 分钟、回归验证 5 分钟、文档交付 2 分钟，开工时预计 15–25 分钟｜用时：21 分钟】**
```

支持英文输出（`--lang en` 或 `export COMPLETION_PROGRESS_LANG=en`）：
```markdown
> **[Progress: ~68%, 2/4 steps done (latest: Fix implementation) | running Regression test (15/40) | elapsed: 14 min | remaining: ~5–8 min, ETA 14:24–14:27 (Beijing time)]**
```

---

## 2. 为什么它算得准？（核心原理通俗版）

传统做法要求“历史里有完全相同的任务”才能参考，而真实开发中任务数量、难度和使用的模型千差万别。本技能采用四维统计建模与三层实时修正（完整数学推导见 [`reference.md`](./reference.md)）：

1. **自动校准不同大模型的“估时脾气”**：
   - 基于数百步真实开发任务的回测发现，不同模型家族对自身耗时的估计存在显著且稳定的系统性偏差（例如部分模型实际耗时约为自估的 `0.5x`，而另一些模型约为 `1.6x–2.2x`）。
   - 脚本自动将模型名称归一到模型族（`claude`、`gpt-6.1`、`gpt-6`、`gemini`、`minimax`、`deepseek`、`qwen` 等），并通过 **Huber 稳健岭回归** 结合按类型校准的开工自估（`base_min`），自动纠偏。
2. **经验分位区间（Split-Conformal），拒绝虚假精确**：
   - 展示给用户的预计时间是 `P20–P80`（60% 预测区间），宽度直接取自本机最近 200 条已发出预测的真实样本外残差分布；当不确定性过高（上限 > 下限 3 倍）时，自动改写为“还剩至少 N 分钟”，绝不强行给出误导性的窄区间。
3. **执行中三路动态修正**：
   - **已完成步偏差传递（$\theta_{\text{task}}$）**：前两步比预期慢了 1.5 倍？脚本会通过经验贝叶斯自动上调本任务剩余步骤的耗时预计。
   - **批量任务实测速率混合**：处理 `15/40` 个文件时，自动按已完成数量将“实测单件速率”与“先验速率”在对数空间按 `1:k` 融合。
   - **单步超时重估**：当某个不可分步骤耗时已超过预测分布的 P95 时，不再机械显示“还剩 0 分钟”，而是自动标记超时并重估剩余工作量。
4. **脏数据自动免疫（v3.1）**：
   - 连续补打卡产生的 `<30 秒` 微型伪迹、忘记打点导致的超长空闲、等待用户确认的阻塞时间（`--block`），都会在拟合时自动扣除或降权为 `0`，不污染历史模型。

---

## 3. 快速安装（如何达到开箱即用的最佳效果）

> **核心秘诀：为什么有的人装了 Skill 却发现 AI 不主动汇报？**
> 要让 AI 在任何长任务中都**被动、稳定、零遗漏**地触发进度汇报，需要同时完成两步：
> 1. **安装运行时技能文件**到各 Agent 的 `skills/completion-progress` 目录；
> 2. **注入全局触发规则块**到各 Agent 的全局指令文件（如 `~/.codex/AGENTS.md`、`~/.gemini/GEMINI.md`、`~/.claude/CLAUDE.md` 等），让 Agent 知道“超过 3 分钟自动启用”以及 `estimate.py` 在你机器上的确切路径。

### 环境要求
- macOS / Linux（Windows 建议使用 WSL）
- **Python 3.10+**（仅使用标准库，无需 `pip install` 任何包）

### 一键全自动安装（推荐）

将本仓库克隆或放置到固定目录（建议放在 `~/Documents/completion-progress` 或 `~/.local/share/completion-progress`，切勿放在 `/tmp` 等会被系统清理的目录）：

```bash
git clone https://github.com/litany-lty/completion-progress.git ~/Documents/completion-progress
cd ~/Documents/completion-progress

# 1. 查看本机已安装哪些支持的 AI Agent
python3 install.py --list

# 2. 一键安装技能文件 + 自动向已存在的 Agent 全局规则文件注入触发规则（自动备份原文件）
python3 install.py --all --inject-rules

# 3. 运行 10 秒自检（在临时目录运行，不触碰真实历史数据）
python3 estimate.py selftest
```

支持自动识别并配置的 Agent / Harness 包括：
- **Claude Code** (`~/.claude/skills/completion-progress` & `~/.claude/CLAUDE.md`)
- **Codex CLI / App** (`~/.codex/skills/completion-progress` & `~/.codex/AGENTS.md`)
- **Gemini CLI / Antigravity** (`~/.gemini/skills/completion-progress`、`~/.gemini/config/skills/completion-progress` & `~/.gemini/GEMINI.md`)
- **Cursor / 通用 Agents** (`~/.agents/skills/completion-progress`)
- **Trae** (`~/.trae/skills/completion-progress` & `~/.trae/user_rules.md`)
- **MiniMax Code** (`~/.minimax/skills/completion-progress` & `~/.minimax/AGENTS.md`)
- **JetBrains Junie / GitHub Copilot / OpenClaw / Hermes / OpenCode / Aider / DeepSeek-Harness**

如果你只想针对单个 Agent 手动安装或查看详细配置，请参阅 **[《安装与多 Agent 配置指南 (INSTALL.md)》](./INSTALL.md)**。

---

## 4. 使用方法

### 4.1 作为用户，你平时怎么用？

配置完成后，你**不需要背诵任何命令**：
- **自动触发（动作判定，拒绝漏报）**：当你向 AI 下达任何需要**多步执行或实质操作**（改代码/文档、跑命令/测试/构建、发布推送、多步排查，即 ≥2 步或预计 ≥1 分钟）的任务时，AI 会在开工第 1 步先跑 `plan` 建立基线，在每步完成时跑 `progress --done` 更新进度，并在最终交付时跑 `deliver` 对比开工预计与实际耗时。
- **简单对话零打扰**：普通纯文字问答、单次秒级只读查看不会触发，保持对话清爽。
- **随时追问**：任务中途你可以直接问 *“现在进度到哪了？还要多久？”*，AI 会调用 `status` 立即为你生成最新进度。
- **随时微调**：你可以直接对 AI 说 *“这轮不用报进度”* 或 *“每完成一步都报一下进度”*，AI 会遵从你的当前会话指令。

### 4.2 Agent 侧命令速查（生命周期）

Agent 在后台通过调用 `estimate.py` 的六个子命令完成全生命周期计量：

```bash
# 1. 开工建基线（或中途范围变化时重划基线）
python3 estimate.py plan \
  --session codex-01a1 \
  --task auth-refactor \
  --model claude-opus-4-6/high \
  --tz Asia/Shanghai \
  --steps '[
    {"id":"s1","name":"梳理现有鉴权链路","type":"research","d":"D2","base_min":4},
    {"id":"s2","name":"改造 Token 刷新与中间件","type":"implement","d":"D4","base_min":15},
    {"id":"s3","name":"运行鉴权回归用例","type":"verify","q":24,"u":"case","d":"D3","base_min":6},
    {"id":"s4","name":"整理变更说明与交付","type":"deliver","d":"D1","base_min":2}
  ]'

# 2. 步骤完成（自动以该步开始到现在的时长计时，自动开启下一步）
python3 estimate.py progress --session codex-01a1 --task auth-refactor --done s1 --evidence "src/auth"

# 3. 批量步骤进行中（上报已完成 10/24 个用例，按实测速率修正剩余时间）
python3 estimate.py progress --session codex-01a1 --task auth-refactor --running s3 --units 10

# 4. 等待用户确认 / 解除等待（等待期自动从步骤和总用时中扣除！）
python3 estimate.py progress --session codex-01a1 --task auth-refactor --block "确认数据库字段兼容性"
python3 estimate.py progress --session codex-01a1 --task auth-refactor --block

# 5. 全部完成并正式交付（输出各步实际用时 vs 开工预计复盘）
python3 estimate.py deliver --session codex-01a1 --task auth-refactor --ref "PR #42"

# 6. 任务中途取消（写入 abandoned 状态，不污染正常任务偏移）
python3 estimate.py abandon --session codex-01a1 --task auth-refactor
```

返回值为精简 JSON（默认仅约 150 字节）：
- 当 `should_report: true` 时，Agent 将 `line` 字段原样粘贴到回复末尾；
- 当 `should_report: false`（例如刚汇报完不到 3 分钟且预计无显著变化）时，Agent 静默继续干活，避免刷屏。

---

## 5. 注意事项与最佳实践（必读！决定预估精度的 7 个关键点）

为了让你的使用效果与经过数百步真实任务调优后的效果完全一致，请留意以下事项（前 4 条已内置在全局规则块中约束 AI）：

### ① `--model` 必须填写真实模型名，切勿填“当前会话”
- **原因**：不同模型家族的执行速度与自估偏差相差可达 2–4 倍。如果 AI 偷懒填入 `--model current` 或 `--model 当前会话`，样本会被归入 `unknown`，失去最核心的模型族偏差校准。
- **做法**：填写真实型号，如 `claude-opus-4-6/high`、`gpt-6.1-sol/xhigh`、`gemini-3.1-pro`；或者在宿主环境中设置 `export COMPLETION_PROGRESS_MODEL="你的常用模型"`。

### ② 每步必给 `base_min`，且单步粒度不少于 1 分钟
- **原因**：`base_min`（AI 开工前对这一步的分钟数自估）包含了具体任务的结构信息，是冷启动和校准的核心锚点。同时，把十几秒的零碎动作拆成独立步骤，或等全部做完才一口气连续执行 `progress --done s1`、`--done s2`，会制造 `<30 秒` 的失真计时。
- **做法**：步骤控制在 **3–6 步**，每步预计 ≥1 分钟；**每步在真正完成的那一刻立即调用 `--done`**。若确实有两步同时完成，用 `--done s2,s3`（脚本会按预计比例分摊时长并标记为半权重）。

### ③ 需要等用户拍板或外部操作时，务必使用 `--block`
- **原因**：如果 AI 停下来等你确认方案，你过了 30 分钟才回复，不用 `--block` 会导致这 30 分钟被算进当前步骤的实测耗时里。
- **做法**：进入等待时调用 `progress ... --block "确认事项"`，收到用户回复后调用 `progress ... --block`（或直接 `--done`），等待期会自动从步骤与任务总用时中精确扣除。

### ④ 遇到意外超时，加上 `--cause` 原因码
- **原因**：偶尔因为本地端口冲突、外部接口挂了、或者用户临时加需求导致某一步耗时翻了 5 倍，这属于偶发异常，不应拉偏该类型任务的正常基准。
- **做法**：明显超时完成时加上 `--cause env|bug|scope|wait|rework|unclear`（如 `--done s2 --cause env`）。其中 `wait` 权重记为 0，`scope` 记为半权重，保护历史模型不被带偏。

### ⑤ 受控类型词表（`type`）
- 推荐使用 7 个标准类别：`research`（调研定位）、`implement`（编码修改）、`verify`（测试验证）、`build`（构建部署重启）、`docs`（文档交接）、`deliver`（交付收尾）、`other`。
- 如果 AI 写了自定义类型（如 `runtime-check`、`code-fix`），脚本会在读取时自动按关键词归类并保留原名到 `label` 字段，防止类型词表碎片化。

### ⑥ 用“搭车执行”省工具调用
- Agent 可以在执行实质测试或构建命令的同一条 Shell 中顺带更新进度，既不增加额外的工具调用轮次，也不改变主命令的退出码：
  ```bash
  pytest tests/ -q; rc=$?; python3 /path/to/estimate.py progress --session S --task T --done s3 >/tmp/cp.json 2>/dev/null; exit $rc
  ```

### ⑦ 定期运行 `audit` 查看本机校准体检
- 当累计样本达到 **20 条**以上时，运行一次校准审计：
  ```bash
  python3 estimate.py audit
  ```
- 它会自动通过时间顺序一步前瞻交叉验证选出最优先验强度 $\kappa$ 写入 `model.json`，并展示：
  - `base_bias_by_family`：你机器上各模型家族的实际耗时倍率；
  - `backtest` / `issued_forecasts`：真实预测误差与 60% 区间命中率（理想值在 `50%–70%` 之间）；
  - `data_quality`：被自动过滤的微型伪迹、暂停等待、放弃任务统计。

---

## 6. 数据存储与隐私安全

默认状态目录位于 `${XDG_STATE_HOME:-~/.local/state}/completion-progress`（所有同机 Agent 自动共享，无需额外配置）：

- `history.jsonl`：只追加的已完成步骤样本与任务汇总。**严格不含任何业务代码或对话正文**（`--evidence` 仅截取前 40 字符作为位置引用），可安全跨项目积累。
- `tasks/<sha1>.json`：当前进行中任务的状态文件，按 `session + task` 哈希隔离，多窗口并发互不干扰；超过 24 小时无事件未交付的任务会在下次 `plan` 时自动收口为 `abandoned`，满 30 天的旧状态文件自动清理。
- `model.json`：`audit` 生成的超参数缓存。

---

## 7. 仓库目录结构与验证体系

```text
completion-progress/
├── SKILL.md                # 核心技能定义（Agent 按需加载，≤5 KB）
├── estimate.py             # 估时与状态管理引擎（纯标准库单文件）
├── reference.md            # 数学模型推导、字段契约与 40 项验收清单
├── install.py              # 多 Agent 安装、校验、卸载与规则注入工具
├── rollout_all.py          # 全局一键同步脚本（技能目录 + 全局规则块 + 自动备份）
├── INSTALL.md              # 详细安装、沙箱共享历史与故障排查指南
├── env.sh                  # 环境变量辅助脚本
├── make_manifest.py        # 生成 manifest.json 哈希清单
├── manifest.json           # 文件角色与 SHA-256 校验清单
├── SHA256SUMS             # 标准 sha256sum 校验文件
├── tests/
│   ├── test_v3.py          # 基础回归测试套件（61 项断言）
│   └── test_v31.py         # v3.1 数据质量、等待扣除、拆分计时与影子模型测试（29 项断言）
└── studies/
    ├── calibration.py      # E1–E14 合成数据统计校准实验
    ├── crosscheck.py       # 独立求解器/调度器交叉验证、600 例模糊测试与完整性门禁
    └── backtest_real.py    # 真实历史样本一步前瞻回测工具
```

运行完整验证套件：

```bash
python3 estimate.py selftest          # 10 秒快速自检
python3 tests/test_v31.py             # v3.1 核心特性测试
python3 studies/crosscheck.py         # 独立数学对照 + 600 例模糊测试 + 文件哈希校验
```

---

## 8. 开源协议

本项目基于 [MIT License](./LICENSE) 开源。欢迎提交 Issue 或 PR 分享你在不同 Agent 与工作流下的校准数据与改进建议。
