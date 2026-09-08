# 棋语 v3_long 候选模型包

这个包不是网页链接，而是实际文件。请把本包内容解压/合并到你的 `棋语智能体` 项目根目录。

## 包内关键文件

- `artifacts/training_v3_long/best_model.pt`：第三版长训候选模型权重。
- `artifacts/master_data_v3/opening_book.json`：第三版使用的职业开局库。
- `artifacts/evaluation_v3_long/evaluation.json`：第三版冻结测试评估结果。
- `qiyu/agent.py`、`qiyu/model.py`、`qiyu/server.py`：支持 v3 搜索字段和新版 PyTorch 加载的关键代码。
- `web/index.html`：能显示搜索深度、节点数、耗时的网页。

## 启动 v3 演示

在 `棋语智能体` 项目根目录执行：

```bash
python -m qiyu.server \
  --checkpoint artifacts/training_v3_long/best_model.pt \
  --opening-book artifacts/master_data_v3/opening_book.json
```

然后浏览器打开：

```text
http://127.0.0.1:8765
```

## 如果没有依赖

建议新建虚拟环境后安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy cchess
```

## v3_long 主要指标

- 冻结测试 Top-1：27.35%
- 冻结测试 Top-3：44.16%
- 冻结测试 MRR：40.75%
- 严格未见 Top-1：22.06%
- 严格未见 Top-3：36.98%

说明：v3_long 静态模仿指标高于 v0.2，但还没有完成 100 局 v3-v0.2 成对实战和 95% 置信区间，所以仍标记为候选模型。
