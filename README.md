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
`results_path` 只指向 `pear_work/cluv_runs` 的管理输出，不包含模型、训练 checkpoint 或数据集。
`cluv sync` 不会把约 306GiB 的训练 checkpoint 拉回 Mila；完整训练日志和结果仍在 `pear_work/runs`。

## 后续提交 PEAR 训练

当前四个正式任务已经提交，不需要再次执行下面的命令。
确认上一轮任务结束后，可提交新的独立运行：

```bash
cluv submit fir
# 或 cluv submit tamia / rorqual / trillium-gpu
```

每次仅使用一个 account，不使用 `submit first` 或多 account 竞速。
`pyproject.toml` 配置 8 个节点、每节点 4 张 H100、2 小时；`scripts/job.sh` 调用已有 PEAR
入口，分配 2 个 actor 节点（8 卡）和 6 个 rollout 节点（24 卡）。
保留 BATCH128、N8、4096+160000 token、FSDP8/SP8、TP4、MTP3、每副本64 sessions、官方 group filter。
LMCache64GiB，checkpoint 每步保存、保留一份，普通密集文件。
Rorqual NUMA headroom32GiB，其他站默认100GiB；Trillium 要求 `mem=0`，其余站 `450G`。
这些完整长序列参数仍需运行验证，已有短 smoke 不保证正式负载内存容量。

取消单个任务可用对应的 SSH 别名，例如 `ssh fir scancel <jobid>`；不要把 `scancel` 广播到所有集群。
Cluv 的新提交会使用 `cluv-pear-stage1` 作业名，现有登记作业仍保持原来的 `pear-stage1` 名称。
原有 Codex 监控跟踪各站 monitor 的当前任务；手工新提交的后续运行需同步更新其权威状态，不能沿用旧 jobid。

## 目录与配置

`clusters.json` 保存已有 PEAR 根目录、工作目录、权威状态路径和实际 Slurm 集群名；
`pyproject.toml` 保存 Cluv 的 account、分区和资源参数。
四站训练源码和环境不由本项目覆盖；管理项目使用独立 Git 分支。
Cluv 的 GPU 总量是调度器可见节点统计，不代表你的 account 能申请全部型号或保留分区。
