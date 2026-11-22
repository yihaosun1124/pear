# PEAR: Cluv 五站竞速

管理目录：Mila `login-2.server.mila.quebec` 的 `$SCRATCH/pear-cluv`
（`/network/scratch/s/sunyi/pear-cluv`）。PEAR、模型、环境和 checkpoint 留在各站 scratch。

## 使用

```bash
ssh mila
cd "$SCRATCH/pear-cluv"
hostname                         # 认证连接保留在 login-2
./login --check                   # 五站连接必须全部有效
cluv status jobs --all-jobs       # 先确认没有已有 PEAR 排队/运行任务
tmux new -s pear-race
cluv submit first --no-sync-datasets --no-fetch-results
```

Cluv 原生负责同步代码、提交、轮询和取消落选。Ctrl-b d 脱离 tmux 后继续等待；
`tmux attach -t pear-race` 返回。等待期间 Ctrl-C 会让 Cluv 取消本轮候选。
同一实验只运行一次提交命令；SSH 或提交结果不明确时先查队列和原生提交日志，不重复提交。
首个任务运行后，Cluv 取消其余候选并返回赢家；训练是否成功需继续看真实 RL 指标和 checkpoint。
Cluv 的同步/提交是异步的，已经有赢家时可能跳过仍未提交的站点；沿用其原生行为。

`first` 使用已认证且启用的集群，因此每轮提交前必须确认五站连接全部有效。
连接失效时用 `./login fir`（或其它站名）完成认证，再 Ctrl-b d 保留连接。
`login` 只保持 SSH 会话，不参与竞速。不要关闭已有 `pear-cluv-access` 会话。
Mila 在本项目中已执行 `cluv disable mila`，不参加训练；不要将其重新启用。
Trillium GPU 别名经 CPU 登录节点到 `trig-login01`，实际 Slurm 集群名为 `grillium`。

官方竞速教程：<https://mila-iqia.github.io/cluv/guides/submit/racing-clusters/>。

## 仅保留的提交配置

| 文件 | 作用 |
|---|---|
| `pyproject.toml` | 共同资源，以及五站节点数、每节点卡数、account、分区、CPU、内存、路径、NUMA 和临时盘差异 |
| `scripts/job.sh` | 在已有分配内用 bash 调用各站同一份 PEAR `deploy/train.sbatch` |
| `patches/cluv-0.1.8.patch` | 当前 Cluv 的两处小兼容修正，不实现竞速逻辑 |

调用链：`cluv submit first → scripts/job.sh → PEAR/deploy/train.sbatch → 官方 Stage1 → RL 训练`。
内层 `train.sbatch` 的 `#SBATCH` 行作为注释，不会再次申请资源。
已删除自定义 `submit`、`submit_race.py`、`status/import_jobs.py`、Python 启动适配器及其重复配置。
原生 Cluv 作业缓存已登记的历史任务仍保留；旧 `.state/race.json` 仅为历史证据，不再控制提交。

五站都申请 32 张 H100、24 小时，actor 8 卡＋rollout 24 卡。
Tamia/Fir/Rorqual/Trillium 使用 8 节点×4 卡（2 个 actor＋6 个 rollout 节点）；
Nibi 使用 4 节点×8 卡（1 个 actor＋3 个 rollout 节点），每 rollout 节点 64 个模拟器，仍共 192 个。
保持 BATCH128、N8、PROMPT4096、RESPONSE160000、STEPS1000、EPOCHS100、FSDP8/SP8、TP4、
MTP3、sessions64/副本及官方 group filter。LMCache64GiB，每步保存、保留最近一份普通密集 checkpoint。
keep1 在新 checkpoint 完整后才删除旧份，保存峰值仍约两份核心（611.5GiB），另计轨迹等数据。

| 集群 | Account / Partition | CPU/节点 | 内存/节点 | NUMA 预留 |
|---|---|---:|---|---:|
| Tamia | `aip-plbacon` / `gpubase_bynode_b3` | 48 | `450G` | 100GiB |
| Fir | `rrg-bengioy-ad_gpu` / `gpubase_bynode_b3` | 48 | `450G` | 100GiB |
| Rorqual | `rrg-bengioy-ad_gpu` / `gpubase_bynode_b3` | 64 | `450G` | 32GiB |
| Trillium GPU | `rrg-bengioy-ad` / `compute_full_node` | 96 | `0` 整节点 | 100GiB |
| Nibi | `rrg-bengioy-ad_gpu` / `gpubase_bynode_b3` | 96 | `450G` | 100GiB |

Fir 暂时排除 `fc10609`：正式任务 `5132847` 在该节点发生 CUDA 初始化失败，
同轮其余五个 rollout 副本均成功 pin 满 64GiB。节点底层原因尚未确认；恢复后移除
`pyproject.toml` 中 Fir 的 `exclude`。每个节点的 CUDA 运算与 pinned memory 预检日志在实际 RUN 下。

24 小时申请需使用三站的 `b3` 分区；`b1` 最长 3 小时，`b2` 最长 12 小时。
改时限时应同时核对实际分区限制，或省略 partition 让站点包装器自动选择。
Trillium 包装器只允许 `mem=0`，没有 450G 硬上限；其内存临时盘也计入主存使用。
其他站使用节点本地磁盘。短 smoke 通过不保证正式 160k 长序列负载容量。

## 同步与日志

`cluv submit first` 自动同步这个小型管理项目；单独同步可用：

```bash
cluv sync tamia fir rorqual trillium-gpu nibi --no-sync-datasets
cluv status jobs --all-jobs
cluv sh -- squeue --me --name=pear-stage1,cluv-pear-stage1
```

各站管理目录为 PEAR 同级的 `pear_cluv_control`。完整 PEAR 源码/环境不由这个管理项目覆盖，
不跨站复制 venv、JIT、模型或 checkpoint。需要更新 PEAR 启动层时，单独验证并同步四站同一文件。

Tamia 的 WORK 是 `/scratch/s/sunyh/pear_work`，其余三站是 `/scratch/sunyh/pear_work`。
Slurm stdout/stderr 合并在 `WORK/cluv_runs/slurm-<jobid>.out`，`logs` 软链接由 Cluv 同步时创建。
完整训练日志、指标、进度和 checkpoint 位于 `WORK/runs/train-<jobid>`。
本配置使用平铺 Slurm 日志，Cluv 按运行目录抓取结果的功能不会自动拉回这些平铺文件；需要时单独读取。
原生提交收据和候选 jobid 在 `~/.cache/cluv/pear-cluv/submission_logs/`，赢家写入原生作业缓存。
原有监控 JSON 保留已接管实验的历史；后续原生提交以原生收据、赢家缓存及真实 Slurm 状态为准。

## Cluv 0.1.8 一次性兼容修正

本机 Cluv 0.1.8 有两处行为需要修正：`first` 仍加入已禁用的本机，且覆盖显式指定的 stdout 路径。
补丁让 `cluv disable mila` 生效，并使用已有 `logs/` 下的平铺日志，避免 Slurm 启动前目录不存在。
竞速、取消和同步仍全部由 Cluv 原生实现。Mila 上已应用；重新安装工具后需重新核对补丁。

```bash
CLUV_PYTHON="$HOME/.local/share/uv/tools/cluster-uv/bin/python"
CLUV_SITE=$($CLUV_PYTHON -c 'import pathlib, cluv; print(pathlib.Path(cluv.__file__).parent.parent)')
patch --dry-run -d "$CLUV_SITE" -p0 < patches/cluv-0.1.8.patch
patch -d "$CLUV_SITE" -p0 < patches/cluv-0.1.8.patch
cluv disable mila
```

若 dry-run 提示补丁已应用，不再重复应用。uv 为 0.12.13，Cluv 工具环境独立；
管理项目和小型环境在 scratch，不在 Home 安装 PEAR。
