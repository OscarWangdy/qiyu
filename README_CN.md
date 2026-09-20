# 棋语 v3 互动升级版

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/OscarWangdy/qiyu)

这是可直接在本地或 Render 运行的中国象棋 Transformer 智能体。

## 新增能力

- 内置“棋语助教”对话框，可询问推荐走法、决策原因、规则和注意力热图。
- 热图同时使用颜色深浅和前五位百分比表示关注度。
- 使用者可选择执红或执黑；执黑时 AI 执红先行，棋盘自动旋转。
- 所有用户可见走法使用“炮二平五”等中国象棋记谱，不再显示西式坐标。
- v3 权重继续训练 20 轮，训练日志、环境、每轮指标与可恢复权重全部留存。

## 包内关键文件

- `artifacts/training_v4/best_model.pt`：本次 20 轮续训的最佳模型权重。
- `artifacts/training_v3_long/best_model.pt`：续训前的 v3 基线权重。
- `artifacts/master_data_v3/opening_book.json`：第三版使用的职业开局库。
- `artifacts/evaluation_v3_long/evaluation.json`：第三版冻结测试评估结果。
- `qiyu/agent.py`、`qiyu/model.py`、`qiyu/server.py`：支持 v3 搜索字段和新版 PyTorch 加载的关键代码。
- `web/index.html`：能显示搜索深度、节点数、耗时的网页。

## 启动 v3 演示

在 `棋语智能体` 项目根目录执行：

```bash
python -m qiyu.server
```

然后浏览器打开：

```text
http://127.0.0.1:8765
```

## 发布公网试玩地址

点击文档顶部的 **Deploy to Render** 按钮，登录 Render 并确认创建免费 Web Service。部署完成后会得到一个公开的 `onrender.com` 地址。云端会自动加载本包中的模型和开局库，每位访客拥有独立棋局。

免费实例在一段时间无人访问后会休眠，下一位访客首次打开时需要等待服务唤醒。

## 如果没有依赖

建议新建虚拟环境后安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy cchess
```

## 模型指标

| 冻结测试指标 | 续训前 v3 | 20 轮续训后 |
|---|---:|---:|
| Top-1 | 27.35% | **30.06%** |
| Top-3 | 44.16% | **47.70%** |
| MRR | 40.75% | **43.44%** |
| 严格未见 Top-1 | 22.06% | **22.64%** |
| 严格未见 Top-3 | 36.98% | **38.58%** |

最佳权重来自续训第 17 轮，验证 Top-1 为 32.73%。完整训练记录见
`artifacts/training_v4/metrics.csv`，冻结测试结果见 `artifacts/evaluation_v4/evaluation.json`。
这些是“模仿职业棋谱下一步”的指标，不是职业棋力或官方等级分。
