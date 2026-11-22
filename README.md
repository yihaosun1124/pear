# PEAR Cluv controller on Mila

管理项目位于 Mila 的 `$SCRATCH/pear-cluv`（`/network/scratch/s/sunyi/pear-cluv`）。Cluv 0.1.8 通过官方推荐的
`uv tool install cluster-uv` 安装在独立工具环境，训练继续使用四站已经验收的 PEAR 环境。

官方教程：<https://mila-iqia.github.io/cluv/guides/getting-started/>。

## 登录和查看任务

```bash
ssh mila
cd "$SCRATCH/pear-cluv"
./login --check
# 仅在某站连接失效时登录该站，例如：
./login fir
# 完成认证后 Ctrl-b d 脱离 tmux，保留会话，再执行：
./status
cluv status jobs --all-jobs
cluv sh -- squeue --me --name=pear-stage1,cluv-pear-stage1
```

连接保存在 Mila 的 `pear-cluv-access` tmux 会话中，复用标准 SSH ControlMaster，
Cluv 原生命令会识别这些连接。本站实测短命令退出后未保留连接，因此 `./login` 保持交互通道。
当前会话位于 **login-2.server.mila.quebec**；先运行 `hostname` 核对，SSH socket 只能在同一 Mila 登录节点复用。
`./login` 不带参数可返回已有 tmux，会话里的远端 shell 请用 Ctrl-b d 脱离而不是 `exit`。
首次连接或 master 断开后，仍可能需要 DRAC 认证。
Mac 上的已认证 socket 不能直接供 Mila 使用。密钥需要在对应集群获准，Duo 仍按集群要求完成。
项目已完成初始化，所有源码和管理项目文件均在 scratch。Cluv 的 `init` 有 Home 路径检查；
本项目已经为各站显式配置 `project_dir`，直接在上述 scratch 目录使用其他命令即可。
`trillium-gpu` 经 CPU 登录节点进入 `trig-login01`，其 Slurm 集群名是 `grillium`。
不要把 Trillium CPU 登录节点的队列当作 GPU 队列。

`./status` 先读取各站权威 `monitor.json` 的当前作业号，使用 Cluv 自身的 `Job` / `save_job`
接口登记已有作业，再显示实时状态。登记不提交或取消 Slurm 作业。
当前阶段若是 CPU 修复验证，会使用相应 CPU 提交收据登记，不把它标成 GPU 正式训练。
已结束并从 `scontrol` 清除的作业使用 `sacct` 查询；失败状态会如实显示。
原始提交命令、源码 commit 和来源保存在私有 `.state/imported-jobs.jsonl` 中。
若连接未认证，Cluv 会显示 disconnected；这不是远端任务失败。

## 同步管理项目

```bash
cluv sync tamia fir rorqual trillium-gpu --no-sync-datasets
```

仅同步本管理项目，远端目录是各站 scratch 下的 `pear_cluv_control`。
该项目没有训练依赖，`uv sync` 不会安装 PEAR / torch 或改动已有训练 venv。
Cluv 自动对齐各站 uv 版本，首次同步前应核对工具版本。
当前工具统一为 uv 0.12.13，从官方 PyPI wheel 安装并核对完整 SHA-256，旧工具留有备份。
Tamia 的 Home 配额不足，其 uv/uvx 位于 `pear_work/tools/cluv-uv-0.12.13/bin`，
`~/.local/bin` 仅放软链接。管理项目使用系统 Python 3.9+，禁止自动下载 Python；
远端控制项目的 venv 和 uv 缓存均在 scratch，不安装训练环境到 Home。
`results_path` 只指向 `pear_work/cluv_runs` 的管理输出，不包含模型、训练 checkpoint 或数据集。
`cluv sync` 不会把约 306GiB 的训练 checkpoint 拉回 Mila；完整训练日志和结果仍在 `pear_work/runs`。

## 四站竞速提交 PEAR 训练

在 Mila 的 login-2 上执行：

```bash
cd "$SCRATCH/pear-cluv"
./submit
./submit --status
```

`./submit` 严格只向 Tamia、Fir、Rorqual、Trillium GPU 各提交一份申请。
它复用 Cluv 原生的代码同步、`get_submissions`、`submit_job` 和作业缓存 API，
外层补充防重复、资源审核、monitor 收据和后台竞速。
所有新候选先 hold；四站都真实提交、登记且核对资源后，一起 release。
第一个真实进入 RUNNING 的作业获胜，其余候选被取消并核对终态。
预测 StartTime、FAILED 和 CANCELLED 不算获胜；两站同时运行时按真实开始时间选择。
取消意图先写入各站 monitor，再 scancel 本轮具体 jobid，落选不是训练故障。

竞速控制器运行在独立 tmux `pear-cluv-race`，当前终端断开后继续等待。
状态为 `.state/race.json`，每轮记录在 `.state/races/`，日志为 `.state/race.log`。
重复运行入口会复用当前控制器或已有候选，不重复申请；提交回执不明确或只有部分 held 申请时，
必须先核对队列与收据，不自动再申请四份。SSH 失效不会自动发 Duo。

不能直接把 `cluv submit first` 当作本项目入口：原生实现还会包含运行命令的 Mila 本机，
且未强制四个目标都齐备。上述封装明确排除 Mila 和 Trillium CPU 调度器。
官方竞速说明：<https://mila-iqia.github.io/cluv/guides/submit/racing-clusters/>。

2026-11-22 首轮接管了已有四个排队作业，保留其队列位置，没有重新申请四份 GPU。
Fir `5119676` 首先运行，Tamia `774175`、Rorqual `27386847`、Trillium `1388261` 已确认取消。
赢家继续训练；修复监控只跟踪赢家，不会把落选重新提交。

`pyproject.toml` 配置 8 个节点、每节点 4 张 H100、2 小时；`scripts/job.sh` 调用已有 PEAR
入口，分配 2 个 actor 节点（8 卡）和 6 个 rollout 节点（24 卡）。
保留 BATCH128、N8、4096+160000 token、FSDP8/SP8、TP4、MTP3、每副本64 sessions、官方 group filter。
LMCache64GiB，checkpoint 每步保存、保留一份，普通密集文件。
Rorqual NUMA headroom32GiB，其他站默认100GiB；Trillium 要求 `mem=0`，其余站 `450G`。
这些完整长序列参数仍需运行验证，已有短 smoke 不保证正式负载内存容量。

取消单个任务可用对应的 SSH 别名，例如 `ssh fir scancel <jobid>`；不要把 `scancel` 广播到所有集群。
Cluv 的新提交会使用 `cluv-pear-stage1` 作业名，现有登记作业仍保持原来的 `pear-stage1` 名称。
统一入口会原子更新各站 monitor 的 current job、RUN、提交收据和竞速结果。
保留原有每 4 小时故障监控；正常两小时预算结束后不无限续跑。

## 调用链与文件来源

```text
Mila ./submit
  → scripts/submit_race.py：同步管理项目、四站 sbatch、资源审核、选赢家和取消落选
  → 各站 Slurm 启动 scripts/job.sh
  → scripts/launch_pear.py：校验统一文件，选择站点路径，传入统一训练参数
  → bash $PEAR_ROOT/deploy/train.sbatch
  → scripts/gpu_process_variables.sh + scripts/start_ray_node.sh
  → examples/models/qwen3_8_27b_pear/train/run_stage1.sh
  → python -m pear.trainer.main
  → TrainingRunner / PearTrainer / verl：rollout、奖励与过滤、actor 更新、checkpoint
```

| 文件 | 来源 | 功能 |
|---|---|---|
| `pyproject.toml` | Cluv 初始化骨架，本项目填写 | 四站 account、分区、CPU、内存、32GPU/2h 和管理目录 |
| `clusters.json` | 本项目新增 | PEAR/WORK/monitor 路径、真实 Slurm 集群名、NUMA 和临时盘策略 |
| `launcher_manifest.json` | 本项目新增 | 统一启动层三个文件的 SHA-256，提交前和启动时核对 |
| `submit`、`scripts/submit_race.py` | 本项目新增 | 固定四站竞速、后台等待、防重复和状态收据 |
| `scripts/job.sh` | Cluv 模板改写 | Slurm 作业入口，执行 Python 适配器 |
| `scripts/launch_pear.py` | 本项目新增 | 找到已有 PEAR 环境，调用统一训练脚本；不重新申请节点 |
| PEAR `deploy/train.sbatch` | 本次 DRAC 部署新增 | 节点角色、预检、本地缓存、Ray 集群和清理；四站内容一致 |
| PEAR `scripts/start_ray_node.sh` | 官方 PEAR，附部署兼容补丁 | 各节点启动 Ray，注册 trainer/rollout/LIBERO 资源和 CPU slots |
| PEAR `scripts/gpu_process_variables.sh` | 官方 PEAR，附部署兼容补丁 | 环境、CUDA/运行库、本地 JIT/cache 和短 Ray socket 路径 |
| PEAR `examples/.../train/run_stage1.sh` | 官方 Stage1 recipe | 将超参数组合成 Hydra 配置并启动训练 Python |
| PEAR `pear/trainer/main.py`、`runner.py`、`trainer.py` | 官方 PEAR | 配置加载、Ray/TransferQueue 调度与实际 RL 训练 |
| PEAR `pear/configs/training.yaml` | 官方配置，附用户要求的 checkpoint 保留设置 | GRPO/group filter、优化器及训练基础设置 |
| `status`、`scripts/import_jobs.py`、`login` | 本项目新增 | 查询/登记已有任务和保持 SSH 连接，不在 RL 计算路径内 |

Cluv 负责外层 sbatch 申请；内层 `train.sbatch` 用 bash 执行，其 `#SBATCH` 行是注释，
因此不会在训练作业内再提交一份作业。account/分区/CPU/memory 的站点差异来自 `pyproject.toml`。

统一启动层的源文件在 fork 的 `drac` 分支，版本 `0356762`。
四站只定点同步这三个文件，并保留各训练 ROOT 的原 HEAD、其它兼容补丁及同站缓存。
这不表示整个 PEAR 工作树已完全同版本；算法源码和环境仍按各站已经验收的部署使用。
当前 Fir 的 Slurm batch 启动快照保留不变，新提交才读取统一后的启动文件。
后续更新启动层应先验证、同步三文件并更新 manifest，避免调用时混用不同版本。

## 目录与配置

`clusters.json` 保存已有 PEAR 根目录、工作目录、权威状态路径和实际 Slurm 集群名；
`pyproject.toml` 保存 Cluv 的 account、分区和资源参数。
四站训练源码和环境不由本项目覆盖；管理项目使用独立 Git 分支。
Cluv 的 GPU 总量是调度器可见节点统计，不代表你的 account 能申请全部型号或保留分区。
