# FM-v6 因子挖掘

负责因子构想、计算、A 段评估、冻结、B 段验证、Goal 和创意卡交付。

从 [crypto-quant-research-platform](https://github.com/stellanhou/crypto-quant-research-platform) 拆出，保留相关文件历史及 2026-10-08 本地未提交的研究改动。原仓库继续保存完整历史、数据工具和统一控制台。

## 安装与使用

```bash
uv sync --frozen --all-extras
.venv/bin/factor-mining factor-mine --help
make test
```

Python 版本由 `.python-version` 固定。每个仓库使用自己的 `.venv`，两者沿用 `crypto_quant` 包路径，不应装入同一个环境。历史文档中的 `python -m crypto_quant.cli factor-mine ...` 在本仓库仍可使用。

## 数据与两系统衔接

配套仓库：[多因子策略研究](https://github.com/stellanhou/multifactor-research)。

行情库和历史运行结果仍位于原工作区 `/Users/stellan/量化投资/market_data/`、`/Users/stellan/量化投资/experiments/`，未复制、移动或提交到 GitHub。命令的 `--db`、`--run-dir`、`--goal-dir`、`--idea-pool` 等路径应显式指向所需文件；合同中的相对路径按原有合同规则解析，移到新目录前需核对。新运行结果默认写入本仓库的 `experiments/`。

因子挖掘生成 FM-v6 JSON 创意卡，多因子系统通过合同中的卡片路径读取。历史卡片及归档按原有相对位置保留。拆仓不启动或恢复研究任务。

共享公式、数据访问、模型调用及证据工具按实际依赖保存在各仓库中，不要求安装另一仓库；`multifactor-research` 中的 `research/factor_mining/` 仅保留所需支持模块。统一网页控制台继续从原仓库运行。

模型凭据通过环境变量或各仓库本地 `.env` 提供；本次不复制凭据。仅提交空配置模板。

## 文档

- [模块说明](src/crypto_quant/research/factor_mining/README.md)
- [文档索引](docs/README.md)
- [迁移记录](docs/repository-split.md)

## 验证状态

2026-10-08 拆分验收：258 项测试通过、2 项既有测试失败；另有 233 项子测试通过。两项失败均在原仓库复现为 `KeyError: horizon_comparison`，详情见[迁移记录](docs/repository-split.md#拆分验收)。
