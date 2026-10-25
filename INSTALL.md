# 安装与多 Agent 配置指南（INSTALL.md）

`completion-progress` 采用**“单一事实来源（Single Source of Truth）”**设计：本仓库目录是唯一的源码与规则中心。安装脚本将运行时三件套（`SKILL.md`、`estimate.py`、`reference.md`）分发到各个 AI Agent 的技能目录，并将被动触发规则写入各 Agent 的全局指令文件。

---

## 1. 一键全量安装（最省心，推荐）

如果你希望本机所有 AI Agent 都能达到**“超过 3 分钟任务自动启用、进度格式统一、共用同一套历史经验”**的效果，只需在仓库根目录执行：

```bash
# 查看本机检测到的 Agent 目录状态
python3 install.py --list

# 一键安装技能文件 + 自动向已存在的全局规则文件注入/更新触发规则块
python3 install.py --all --inject-rules
```

该命令会自动执行以下操作：
1. **备份**：在 `./backups/rollout-时间戳/` 下完整备份即将被覆盖的旧技能目录和全局规则文件；
2. **分发技能**：向本机已存在的 Agent 技能目录写入最新的 `SKILL.md`、`estimate.py`、`reference.md`，并清理旧版残留文件；
3. **注入全局规则块**：在各 Agent 的全局指令文件（如 `~/.codex/AGENTS.md`、`~/.gemini/GEMINI.md`、`~/.claude/CLAUDE.md` 等）中查找 `<!-- completion-progress:begin --> ... <!-- completion-progress:end -->` 标记：
   - 若已存在，则原地更新为最新版本与当前仓库路径；
   - 若不存在，则在文件末尾追加规则块（保留你原有的全部个人规则）。
4. **哈希校验**：逐文件比对 SHA-256，确保落盘内容与源目录完全一致。

> **提示**：如果某个已安装的 Agent 还没有创建全局规则文件（例如刚装好 Claude Code 但还没有 `~/.claude/CLAUDE.md`），可以运行：
> ```bash
> python3 rollout_all.py --create-rules
> ```

---

## 2. 按需安装到指定 Agent

如果你只想为某一个或几个 Agent 安装：

```bash
# 仅安装运行时三件套（SKILL.md / estimate.py / reference.md）
python3 install.py --target ~/.claude/skills/completion-progress
python3 install.py --target ~/.codex/skills/completion-progress
python3 install.py --target ~/.gemini/skills/completion-progress

# 连同测试、实验脚本与文档一起完整复制
python3 install.py --target ~/.claude/skills/completion-progress --full
```

手动安装后，建议运行以下命令打印全局规则块，并将其粘贴到对应 Agent 的全局指令文件中（如 `CLAUDE.md`、`AGENTS.md`、`GEMINI.md`、`.cursorrules`）：

```bash
python3 install.py --print-rule
```

输出的规则块会自动填入你机器上的真实仓库路径：

```markdown
<!-- completion-progress:begin -->
## 进度与预计（completion-progress 3.1）
凡需实质执行（改文件、写文档、跑命令/测试/构建、发布推送、多步排查，≥2 步或预计 ≥1 分钟）均启用；仅纯问答或单次只读查询不启用。只观察，不新增任务、不改业务文件、不写进交付物。
- 规则全文：/your/path/to/completion-progress/SKILL.md
- 脚本：python3 /your/path/to/completion-progress/estimate.py plan|progress|status|deliver|abandon|audit
- 三必报硬约束：① 开工第 1 个工具调用必跑 plan（支持流式输出时先贴【进度分析：…】再执行）；② 每步真正完成时立即 progress --done（不攒到最后，should_report 为真即贴）；③ 收尾必跑 deliver 并在回复末尾贴汇总行（单回合合并输出时，回复开头贴 plan 行、末尾贴 deliver 行）
- 精度要点：等待用 --block；明显超时加 --cause；--model 写真实型号（不写“当前会话”）
- 共享历史：默认 ~/.local/state/completion-progress（同机 agent 共用）；宿主沙箱可用 --state 指定
<!-- completion-progress:end -->
```

---

## 3. 多 Agent 共享历史与沙箱配置

### 3.1 默认共享机制
在同一台机器、同一个操作系统用户下，所有 Agent 默认读写：
```text
${XDG_STATE_HOME:-~/.local/state}/completion-progress
```
这意味着你在 Claude Code、Codex、Gemini/Antigravity、Cursor 中完成的任务样本会自动汇入同一个 `history.jsonl`，**任何一个 Agent 都在帮其他 Agent 积累校准数据**。

### 3.2 沙箱或容器环境配置
如果某个 Agent 运行在受限沙箱或容器中（例如其 `$HOME` 被重定向到临时目录，或默认只允许读写工作区与特定白名单目录），请将该目录显式指向统一路径：

```bash
# 打印环境变量配置命令
python3 install.py --print-env
```

将输出的 `export COMPLETION_PROGRESS_STATE="..."` 加入你的 `~/.zshrc`、`~/.bashrc` 或 Agent 宿主的环境变量配置中；或者在沙箱配置中将 `~/.local/state/completion-progress` 加入可写白名单。

### 3.3 多窗口并发安全保证
- **历史样本 (`history.jsonl`)**：仅在步骤完成或任务交付时单行原子追加（JSONL）；
- **任务状态 (`tasks/<sha1>.json`)**：按 `session + task` 的 SHA-1 前缀分文件存储，并通过临时文件 + `os.replace` 原子替换；
- **模型缓存 (`model.json`)**：同样采用临时文件原子替换。
因此，即使你同时开着 5 个终端窗口跑不同任务，也绝不会互相覆盖进度。

---

## 4. 环境变量一览

| 环境变量 | 作用 | 默认值 / 示例 |
|---|---|---|
| `COMPLETION_PROGRESS_STATE` | 指定状态与历史目录 | `~/.local/state/completion-progress` |
| `COMPLETION_PROGRESS_TZ` | 指定汇报时区（用于计算完成时刻） | `Asia/Shanghai`、`America/New_York`（未设时读 `$TZ` 或系统本地偏移） |
| `COMPLETION_PROGRESS_LANG` | 指定进度提示语言 | `zh`（默认中文）或 `en`（英文 `[Plan: ...]` / `[Progress: ...]`） |
| `COMPLETION_PROGRESS_MODEL` | 默认执行模型型号（当 `--model` 省略时读取） | 如 `claude-opus-4-6/high`、`gpt-6.1-sol/xhigh` |
| `COMPLETION_PROGRESS_ESTIMATOR` | 切换估计器内核 | `v3`（默认：v3 回归 + 经验分位区间）或 `v4`（影子残差估计器） |

---

## 5. 更新、校验与卸载

### 更新
拉取最新代码或修改本仓库文件后，重新执行安装命令即可无损更新（不会删除或覆盖任何历史样本）：

```bash
python3 install.py --all --inject-rules
```

如果你修改了源码并希望重新生成完整性校验清单：

```bash
python3 make_manifest.py
python3 studies/crosscheck.py
```

### 校验与自检
```bash
# 校验某个已安装目录的文件哈希是否与 manifest.json 一致
python3 install.py --check ~/.claude/skills/completion-progress

# 运行 10 秒功能自检（使用临时目录，不触碰真实状态）
python3 estimate.py selftest
```

### 卸载
```bash
# 仅移除目标目录中的运行时文件（保留历史数据）
python3 install.py --uninstall ~/.claude/skills/completion-progress
```
若需彻底清除所有历史数据，手动删除 `~/.local/state/completion-progress`（或 `$COMPLETION_PROGRESS_STATE` 指向的目录）并从全局规则文件中移除 `<!-- completion-progress:begin --> ... <!-- completion-progress:end -->` 段落即可。

---

## 6. 常见问题排查（FAQ）

1. **Q：为什么任务跑了很久，AI 却没有在末尾贴进度条？**
   - 检查该 Agent 的全局规则文件（如 `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`）是否已包含 `completion-progress` 规则块（运行 `python3 install.py --all --inject-rules` 可自动补齐）。
   - 若单步耗时变化不大且距离上次汇报不足 3 分钟，脚本会返回 `"should_report": false` 以防刷屏；你可以随时直接追问“还要多久”，AI 会调用 `status` 强制返回最新进度。

2. **Q：提示 `no task state for S/T; run plan first` 怎么办？**
   - 说明 Agent 在未调用 `plan` 建立基线的情况下直接调用了 `progress` 或 `deliver`，或者前后的 `--session` / `--task` 参数不一致。按错误提示中的 `hint` 先执行 `plan` 即可。

3. **Q：为什么预计时间写着“还剩至少 15 分钟，完成「xxx」后收窄”，而不是具体区间？**
   - 当预测分布的 P80 上限超过 P20 下限的 **3 倍**（常见于零历史冷启动且存在高不确定度步骤）时，脚本会自动将过宽的区间改写为“至少 N 分钟”，避免给出诸如“5–60 分钟”这种无参考价值的区间；待首步完成后，脚本会根据实测速度自动收窄为具体区间。

4. **Q：只读沙箱环境下脚本报错会影响主任务吗？**
   - 不会。当状态目录只读或不可写时，`estimate.py` 仍会正常输出预测文案（退出码为 `0`），仅在 `warn` 字段提示 `state_not_persisted` 或 `history_not_written`，绝不阻断宿主主任务。
