# Mantis 新冠阳性率预测 — 前置工作（2026-11-13）

> 状态：**只在本地写了草稿，还没往任何仓库推过任何东西，也没向上游提过 PR。**
> 同步 fork 的事已交给浏览器去点“Sync fork”，结果还没回来；
> 下面这些文件都在 `~/workspace/mantis-test/work/`，等你检查完再决定推不推。

## 逻辑，一句话

每周取 Hub 目标数据里最新 112 周的新冠阳性率 → 用 Mantis v1（8 周、无协变量）往后推 8 周 → 把 Mantis 的 9 个分位数插值成 Hub 要的 23 个分位数 → 写成 Hub 格式的 CSV → 先人工检查，检查通过才提 PR。

## 关键映射（已用真实数据验算过）

- Hub 目标：`wk inc covid prop ili`，单位是百分点（13.5 = 13.5%），地点 `CN`。
- `reference_date` 必须是周一；`target_end_date = reference_date + horizon × 7 天`，也是周一。
- Horizon 范围 -5..6；每 horizon 必须给齐 23 个分位数：
  `0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99`，值 ≥ 0，分位数单调不交叉。
- Mantis 输出 9 个分位数：`0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95`，中位数是第 5 个。
- Mantis 第 i 步（i=0..7）预测的是“最后观测周 + 7×(i+1) 天”那周。脚本是按这个动态算 horizon 的，不写死 -1..6。
  - 实测：数据截至 2026-10-26（1.3%），`reference_date=2026-11-16` 时，8 步对应 horizon **-2..5**，共 184 行。
  - 等 Hub 数据更新到 2026-11-09 那周，同一轮就会正好对应 -1..6。
- 文件名：`<reference_date>-GZNL-Mantis.csv`，放在 `model-output/GZNL-Mantis/`。

## 三个必须先知道的坑（都写进脚本和 YAML 里了）

1. **Mantis 是用在计数上的，不是阳性率。** 它训练时只有 cases / hosp / deaths 三种类型，我们拿阳性率（0-100）当成 cases 喂进去，是 off-label 用法。第一次跑出来的数一定要对着近期趋势看（最近是从 21.2% 一路跌到 1.3%），离谱就别交。
2. **23 个分位数是插值出来的，不是 Mantis 原生的。** 0.01 / 0.99 两端被钳制在 Mantis 的 0.05 / 0.95 上，尾部是低估的。
3. **数据有滞后。** 2026-11-13 当天 Hub 的 `target-data/time-series.csv` 里新冠阳性率最新只到 2026-10-26，共 204 周（2022-12-05 起），刚好够 112 周输入窗口，但离 11 月 16 日那轮差了 3 周。

## 文件清单（都在 work/ 里）

| 文件 | 作用 |
|---|---|
| `scripts/run_mantis_forecast.py` | 主脚本：读数据 → 跑 Mantis → 转 Hub CSV。`--mode dummy` 可以不用 torch 跑通格式全流程（测试用，绝不能提交）。已用 dummy 模式实测通过校验。 |
| `scripts/validate_hub_csv.py` | 本地校验：列、BOM、文件名、周一、horizon、23 分位数、单调、0-100 范围。 |
| `scripts/requirements.txt` | numpy + pandas。torch 不放这里，workflow 里单独装 CPU 版。 |
| `.github/workflows/mantis-forecast.yaml` | **仅手动触发的测试版 workflow**：`workflow_dispatch`，`contents: read`，不推分支、不提 PR、不写 schedule，只把 CSV 当 artifact 上传给你下载检查。schedule 那段是注释掉的。 |
| `model-metadata/GZNL-Mantis.yaml` | 模型注册草稿。注意 contributor 邮箱现在还是 `gznl@example.com` 这个占位符，提交前要换成真实的。 |
| `README_PREWORK.md` | 本文件。 |

## 你回家后要做的事（按顺序）

1. **同步 fork**：去 `99mrliu/China-COVID-19-Forecast-Hub` 点 `Sync fork → Update branch`（如果浏览器那边已经同步成功了，这步就跳过，我会告诉你结果）。
2. **开 Actions**：fork 的 `Actions` 页点启用；`Settings → Actions → General → Workflow permissions` 改成 Read and write，并勾上允许创建 PR（现在这个测试版 workflow 只要 read 就够，改成 write 是为以后自动提 PR 做准备）。
3. **检查上面这些草稿**，特别是逻辑和那三个坑。没问题了告诉我，我再把它们推到 fork 的 `mantis-test` 测试分支上（不动 `main`，不向上游提 PR）。
4. **手动跑一次**：Actions 里选这个 workflow，`Run workflow`，`reference_date` 填下个周一（2026-11-16），下载 artifact 里的 CSV，用 `validate_hub_csv.py` 过一遍，再看中位数合不合理。
5. **都通过了，再决定**：要不要注册模型（`GZNL-Mantis.yaml` 提 PR 给同事合并）、要不要把 schedule 打开改成每周自动跑。

## 还没做的事（故意留着，等你点头）

- 没同步完 fork 之前，不往 fork 推任何文件。
- 没注册模型之前，不提任何预测 PR。
- 没跑通第一次手动运行之前，不开 schedule，不做自动提 PR 的步骤。
