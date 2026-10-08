# 棋语 v3 互动升级版

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/OscarWangdy/qiyu)

这是可直接在本地或 Render 运行的中国象棋 Transformer 智能体。当前默认加载 v5 权重。

## 新增能力

- 接入 DeepSeek `deepseek-flash`：支持自由对话，且每轮传入当前局面、行棋方和合法着作为事实约束。
- 支持上传 JPEG、PNG、GIF 或 WebP 残局图片；中国象棋专用 ONNX 模型会先定位四角、透视校正，再分类90个交叉点。
- 识图结果先进入可编辑草稿棋盘，可逐格修正；正式导入时再检查双方将帅和棋子数量。
- 未配置 DeepSeek 时，“棋语助教”仍可离线回答推荐走法、决策原因、规则和热图问题。
- 热图同时使用颜色深浅和前五位百分比表示关注度。
- 使用者可选择执红或执黑；执黑时 AI 执红先行，棋盘自动旋转。
- 所有用户可见走法使用“炮二平五”等中国象棋记谱，不再显示西式坐标。
- v4 权重用红黑换位旋转增强继续训练 12 轮，并保留训练日志、每轮指标和权重。
- 推理中的合法着打分与训练目标统一；搜索修正了候选着之间错误共用剪枝界限的问题。

## 包内关键文件

- `artifacts/training_v5/best_model.pt`：当前默认权重，来自新训练第 10 轮。
- `artifacts/training_v4/best_model.pt`：上一版权重，可用于回退和对比。
- `artifacts/training_v3_long/best_model.pt`：续训前的 v3 基线权重。
- `artifacts/master_data_v3/opening_book.json`：第三版使用的职业开局库。
- `artifacts/vision/*.onnx`：真实场景棋盘四角定位与90点分类模型。
- `artifacts/evaluation_v3_long/evaluation.json`：第三版冻结测试评估结果。
- `qiyu/agent.py`、`qiyu/model.py`、`qiyu/server.py`：支持 v3 搜索字段和新版 PyTorch 加载的关键代码。
- `web/index.html`：能显示搜索深度、节点数、耗时的网页。

## 启动 v3 演示

在本项目目录执行：

```bash
python -m qiyu.server
```

然后浏览器打开：

```text
http://127.0.0.1:8765
```

## 配置 DeepSeek

本地启动后，可在网页「DeepSeek 连接」中粘贴 API Key。Key 只会写入已被 Git 忽略且权限为 `0600` 的 `.env`，且此接口只允许本机访问。也可复制 `.env.example` 为 `.env`，填入：

```bash
DEEPSEEK_API_KEY=your_key_here
```

Render 等公网部署不提供网页存 Key，请在服务器环境变量中配置 `DEEPSEEK_API_KEY`。可选变量为 `DEEPSEEK_BASE_URL`、`DEEPSEEK_TEXT_MODEL`、`DEEPSEEK_VISION_MODEL`。实现依据 [DeepSeek 对话补全文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/) 和 [图像理解文档](https://api-docs.deepseek.com/zh-cn/guides/vision/)。

安全边界：DeepSeek 负责意图理解和语言表达；专用 ONNX 模型负责图片定位与分类，DeepSeek 仅作识图失败时的草稿兜底。最终走法仍由本地 Transformer、搜索和规则层约束。专用视觉模型的来源与许可见 `THIRD_PARTY_NOTICES.md`。

## 发布公网试玩地址

点击文档顶部的 **Deploy to Render** 按钮，登录 Render 并确认创建免费 Web Service。部署完成后会得到一个公开的 `onrender.com` 地址。云端会自动加载本包中的模型和开局库，每位访客拥有独立棋局。

免费实例在一段时间无人访问后会休眠，下一位访客首次打开时需要等待服务唤醒。

## 如果没有依赖

建议新建虚拟环境后安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_v3_runtime.txt
```

## 新训练与棋力评估

v5 从 v4 最佳权重继续训练 12 轮，每轮使用 53,885 个职业棋谱局面；除原有左右镜像，还随机旋转棋盘 180 度并交换红黑棋子。训练集与验证集的哈希见 `artifacts/training_v5/environment.json`。最佳验证 Top-1 为 32.84%，v4 最佳为 32.73%。

使用同一份 `train.jsonl`、`validation.jsonl` 数据可复现训练命令：

```bash
python -m qiyu.retrain --data-dir /path/to/master_data --checkpoint artifacts/training_v4/best_model.pt --output-dir artifacts/training_v5 --epochs 12 --learning-rate 5e-5 --minimum-learning-rate 5e-6 --color-rotation
```

以下两列都用修正后的同一套合法着打分、同一冻结测试集重算：

| 冻结测试指标 | v4 | v5 |
|---|---:|---:|
| Top-1 | **31.02%** | 30.88% |
| Top-3 | 49.29% | **50.22%** |
| MRR | 44.78% | **44.89%** |
| 严格未见 Top-1 | 23.19% | **23.33%** |
| 严格未见 Top-3 | 40.68% | **41.51%** |

在 10 个固定开局、双方换边共 20 盘的限时对局中，v5 对 v4 为 4 胜 1 负，另 15 盘重复或到达 80 步上限，v5 平均局末子力差 +3.21。对局结果见 `artifacts/arena_v5_vs_v4_extended.json`。样本仍小，这些指标不能换算为官方等级分，也不足以保证对所有人类棋手更强。

完整记录见 `artifacts/training_v5/metrics.csv`、`artifacts/evaluation_v4_consistent/evaluation.json`、`artifacts/evaluation_v5/evaluation.json` 和 `MODEL_CARD_V5.md`。旧权重仍在，可使用 `python -m qiyu.server --checkpoint artifacts/training_v4/best_model.pt` 回退。
