# TFD-Bench: Uncertainty Benchmark for Fault Diagnosis

面向一维故障诊断信号的不确定性量化基准。项目提供统一的数据加载、训练、OOD 检测、噪声鲁棒性评估和多随机种子统计流程。

## 功能

- 使用一个 YAML 配置批量运行 `datasets × methods × backbones`
- 每种方法也可以通过命令行独立运行
- 统一保存 checkpoint、逐次实验指标和均值/标准差
- 支持 ID、OOD 和多等级噪声评估
- 支持确定性、集成、贝叶斯、证据学习和后处理方法

## 项目结构

```text
.
├── configs/default.yaml    # 一键实验总配置
├── methods/                # 每种不确定性方法的独立入口
├── src/                    # 数据、模型、损失、指标和训练核心
├── analysis/               # 结果汇总、表格和可视化
├── requirements.txt        # Python 依赖及验证版本
├── run.py                  # 批量实验入口
└── results/                # 默认输出目录
```

## 环境

建议使用 Python 3.10。以 Conda 为例：

```bash
conda create -n tfd python=3.10 -y
conda activate tfd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装完成后可以检查主要依赖：

```bash
python -c "import torch, lightning, torchmetrics; print(torch.__version__)"
```

[`requirements.txt`](requirements.txt) 的核心依赖版本来自当前 `fd` 环境，并额外包含：

- `laplace-torch`：供 `laplace_approx` 方法使用
- `mamba-ssm`：仅供 Mamba backbone 使用

默认的 ResNet 实验不需要调用 Mamba。PyTorch、CUDA 与 `mamba-ssm` 对系统环境有较强依赖；如果安装 Mamba 时发生编译错误，可先确保 PyTorch 已安装，再关闭构建隔离重新安装：

```bash
pip install torch==2.12.1 torchvision==0.27.1
pip install mamba-ssm==2.3.2.post1 --no-build-isolation
```

## 快速开始

### 方式一：通过总配置一键运行

修改 [`configs/default.yaml`](configs/default.yaml)：

```yaml
methods:
  - max_softmax
  - edl
  - mc_dropout

backbones:
  - resnet
  - transformer

datasets:
  - name: seu
    root: /mnt/d/Data/Machine/SEU
  - name: mgb
    root: /mnt/d/Data/Machine/MGB

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  seeds: [0, 1, 2]
  val_split: 0.2

evaluation:
  noise: true

hardware:
  accelerator: auto
  devices: 1
  strategy: auto
  precision: 32

output:
  dir: results
```

先预览将要执行的命令，不启动训练：

```bash
python run.py --dry-run
```

运行配置中的全部组合：

```bash
python run.py
```

默认会覆盖所选数据集、模型和方法对应的旧实验目录；不会删除其他未选中的实验结果。可在 `runner` 配置中通过 `overwrite` 关闭覆盖行为。训练完成后，再按“结果汇总与可视化”一节中的命令生成汇总、表格和图片。

`run.py` 会依次执行每个 `dataset × method × backbone` 组合。默认在某个组合失败后继续运行其他组合，并在结束时汇总失败项；可通过配置中的 `runner.continue_on_error` 修改该行为。

> 并非所有方法都支持所有 backbone。`resnet` 的兼容性最完整；增加其他 backbone 前建议先用 `--dry-run` 检查实验矩阵，并单独运行一次对应方法。

### 方式二：使用另一份配置

复制默认配置并修改：

```bash
cp configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml
```

Windows PowerShell：

```powershell
Copy-Item configs/default.yaml configs/my_experiment.yaml
python run.py --config configs/my_experiment.yaml
```

### 方式三：直接运行单个 method

每个 `methods/*.py` 都是独立入口。它会读取默认配置，再使用命令行参数覆盖配置值。

使用默认配置运行 EDL：

```bash
python methods/edl.py
```

指定配置和 backbone：

```bash
python methods/edl.py \
  --config configs/default.yaml \
  --backbone transformer
```

完全通过命令行覆盖常用参数：

```bash
python methods/edl.py \
  --dataset mgb \
  --data-root /mnt/d/Data/Machine/MGB \
  --backbone resnet \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --seeds 0 1 2 \
  --val-split 0.2 \
  --reg-weight 0.5 \
  --loss-type digamma \
  --no-eval-noise
```

查看某个方法支持的全部参数：

```bash
python methods/edl.py --help
```

### 方式四：在 YAML 中设置方法专属参数

方法专属参数放在 `method_args` 下，只会传给对应方法：

```yaml
method_args:
  deep_ensemble:
    num_estimators: 8
    ood_criterion: entropy

  edl:
    reg_weight: 0.5
    loss_type: digamma

  mc_dropout:
    dropout_rate: 0.1
    num_estimators: 50
    ood_criterion: entropy

  conformal_aps:
    alpha: 0.01
    randomized: true
    enable_ts: false
```

YAML 中的下划线参数会自动转换为命令行的连字符形式，例如 `num_estimators` 会转换为 `--num-estimators`。布尔值会转换为 `--flag` 或 `--no-flag`。

## 参数优先级

运行单个方法时，参数优先级为：

```text
命令行参数 > --config 指定的 YAML > 代码默认值
```

常用公共参数：

| 参数 | 说明 |
|---|---|
| `--config` | YAML 配置路径 |
| `--dataset` | 数据集名称 |
| `--data-root` | 数据集根目录 |
| `--backbone` | 模型名称 |
| `--epochs` | 训练轮数 |
| `--batch-size` | batch size |
| `--lr` | 学习率 |
| `--seeds` | 明确指定随机种子列表 |
| `--n-runs` | `seeds` 未设置时生成 `0..n_runs-1` |
| `--val-split` | 验证集比例 |
| `--eval-noise` | 启用噪声评估 |
| `--no-eval-noise` | 禁用噪声评估 |
| `--accelerator` | `auto`、`gpu` 或 `cpu` |
| `--devices` | 设备数量或 `auto` |
| `--strategy` | 多卡策略 |
| `--precision` | 训练精度 |
| `--output-dir` | 输出根目录 |

## 支持的方法

| 类别 | 方法标识 |
|---|---|
| 基线 | `max_softmax` |
| 集成方法 | `deep_ensemble`, `packed_ensemble`, `batch_ensemble`, `snapshot_ensemble`, `checkpoint_ensemble` |
| 贝叶斯方法 | `variational_bnn`, `swag`, `sgld`, `sghmc` |
| 证据方法 | `edl`, `tessa`, `tessav1` |
| 共形预测 | `conformal_aps`, `conformal_raps`, `conformal_thr` |
| 后处理/采样 | `temperature_scaling`, `laplace_approx`, `mc_dropout`, `mc_batch_norm` |

方法名称与 `methods/<方法名称>.py` 一一对应。

`tessa` 是本 benchmark 对重构后方法采用的名称，其模型来源是 ICLR 2022 的
[Evidential Turing Processes](https://openreview.net/forum?id=84NMXTHYe-)（原论文名称为 ETP）。
入口文件只保留实验组装，模型主体位于 `src/models/tessa.py`。

TESSA 的主要参数：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `reg_weight` | Dirichlet KL 正则权重 | `1e-3` |
| `memory_size` | 特征空间中的外部 memory slot 总数 | `20` |
| `memory_decay` | 旧 memory 的保留比例 | `0.99` |
| `memory_std` | memory 查询时的采样标准差 | `0.1` |
| `context_size` | 每个 batch 随机 memory context 的上限 | `50` |
| `prior_precision` | 变分线性层的高斯先验精度 | `10.0` |
| `weight_decay` | Adam 的权重衰减 | `0.0` |

EDL 的 `reg_weight=0.5` 与 TESSA 的 `reg_weight=1e-3` 不应直接设成相同值：
两种方法的正则项定义和数值尺度不同，因此不应直接设置成相同数值。

EDL 输出的是非负 evidence，而不是普通 logits。训练时使用 `DECLoss`，验证和测试时
先计算 `alpha = evidence + 1`，再以 Dirichlet 均值 `alpha / alpha.sum()` 作为类别概率；
因此 EDL 的 NLL 和 ECE 不会对 evidence 直接使用 softmax。

TESSA 的 memory 位于 backbone 的特征空间中：每个类别拥有独立的 memory slots，
训练时按类别和余弦相似度更新，预测时将查询到的 memory 特征与当前样本特征拼接后输出证据。
当前实现要求 backbone 提供 `feats_forward()` 和线性 `fc` 分类头（默认的 `resnet` 满足要求）。

`tessav1` 是独立的实验方法，保留 `tessa` 的结构和结果，在此基础上增加
prototype-contrastive 特征约束，使样本靠近同类 memory slots 并远离其他类别 slots。
该约束仅作用于训练，不包含 temperature scaling，也不改变测试阶段的概率计算。
其新增参数为：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `feature_reg_weight` | 特征与同类 memory slots 的对比约束权重 | `0.05` |
| `feature_temperature` | prototype-contrastive 相似度温度 | `0.1` |
| `feature_warmup_epochs` | 特征约束从弱到强的 warm-up epoch 数 | `10` |

两种方法分别写入 `results/<dataset>/<backbone>/tessa` 和
`results/<dataset>/<backbone>/tessav1`，不会相互覆盖。

单独运行 TESSAv1：

```bash
python methods/tessav1.py --dataset seu --data-root /mnt/d/Data/Machine/SEU --backbone resnet
```

## 支持的模型

公共 CLI 当前接受：

```text
resnet
lenet
mlp
transformer
mamba
lstm
timesnet
```

部分方法需要模型提供专用的 Bayesian 或 Packed 变体，因此不保证上述每个模型都能与所有方法组合。

## 支持的数据集

```text
seu
wt
pu
xjtu
hit
cwru
thu
mgb
```

数据输入统一为一维时域信号。一键运行多个数据集时，在 `datasets` 列表中为每个数据集同时填写 `name` 和 `root`。直接运行单个 method 时，默认使用列表中的第一个数据集，也可以通过 `--dataset` 和 `--data-root` 覆盖。

## 输出结构

默认结果保存在：

```text
results/<dataset>/<backbone>/<method>/
├── config.json
├── raw_all_seeds.csv
├── summary.csv
└── seed<seed>/
    ├── metrics.csv
    ├── ckpt/
    └── logs/
```

- `metrics.csv`：单个随机种子的 clean/noise 指标
- `raw_all_seeds.csv`：所有种子的原始结果
- `summary.csv`：按测试配置汇总的均值和标准差
- `config.json`：本次方法运行的最终参数

启用噪声评估后，每个随机种子还会运行数据集定义的不同噪声类型和严重程度，因此运行时间会明显增加。调试代码时建议使用：

```bash
python methods/max_softmax.py \
  --epochs 1 \
  --seeds 0 \
  --no-eval-noise
```

## 结果分析

汇总实验结果：

```bash
python analysis/collect_results.py
```

默认同时汇总 clean 和所有噪声等级。只读取 clean 或指定噪声等级：

```bash
python analysis/collect_results.py --test-config clean
python analysis/collect_results.py --test-config gaussian_s3
```

也可以筛选数据集、模型或方法：

```bash
python analysis/collect_results.py \
  --dataset seu \
  --backbone resnet \
  --method edl
```

生成对比表格：

```bash
python analysis/generate_tables.py
```

Markdown 表格默认保存到 `results/table.md`。表格只展示 `ACC`、`ECE`、`AUROC`，并按数据集、模型和测试配置分别生成表格，例如 `clean`、`gaussian_s1`、...、`gaussian_s5` 各有一张表。ACC、ECE 和 AUROC 均以百分数显示，粗体表示同一张表中的最优结果。

同时在终端中显示表格，或生成其他格式：

```bash
python analysis/generate_tables.py --print
python analysis/generate_tables.py --format latex
python analysis/generate_tables.py --format html
```

后两条命令默认分别保存到 `results/table.tex` 和 `results/table.html`；仍可使用 `--output` 指定其他保存位置。

完整生成命令：

```bash
python analysis/collect_results.py
python analysis/generate_tables.py
```

所有方法均使用验证集 `NLL` 最小的 epoch 作为最佳 checkpoint。测试集以及噪声测试结果不参与模型选择；多个随机种子分别选取 checkpoint 后，再汇总均值和标准差。

生成按数据集、模型和噪声等级分组的 ACC/ECE/AUROC 对比图，以及跨噪声等级趋势图：

```bash
python analysis/visualization/plot_all.py
```

对比图中的点为多个随机种子的均值，误差线为标准差。需要先清理之前生成的基准图片时使用：

```bash
python analysis/visualization/plot_all.py --clean
```

图片默认保存到 `figures/<dataset>/<backbone>/`。`summary.json` 只包含聚合指标，
因此 ROC 曲线、可靠性图和不确定性分布图需要额外的逐样本预测分数，不能由汇总文件直接还原。
