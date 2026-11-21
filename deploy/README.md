# PEAR：DRAC 集群部署与训练

部署入口只有三个文件：

- `deploy/smoke_test.sbatch`：两节点、4+4 H100，验证一个真实 RL step。
- `deploy/train.sbatch`：64 卡 Stage 1，16 卡训练 + 48 卡 rollout；同时提供 smoke 的公共启动逻辑。
- `deploy/README.md`：安装、提交、参数、故障处理与验证记录。

启动器调用官方 `examples/models/qwen3_8_27b_pear/train/run_stage1.sh`，不另写训练循环。
账户和分区通过 `sbatch` 参数指定；文件名和代码目录不绑定某个集群。
当前适配的是 Qwen3.8-27B / H100，迁移到其他 GPU 架构需要重新选择官方模型配置及编译目标。

## 1. 路径与版本

以下以 `/scratch/$USER/pear` 和 `/scratch/$USER/pear_work` 为例。
实际迁移时先查看 `$SCRATCH`：Tamia 为 `/scratch/s/$USER`，应使用
`PEAR_ROOT=$SCRATCH/pear WORK_DIR=$SCRATCH/pear_work`，并覆盖 Slurm 日志路径。
本次用户为 `sunyh`，实际部署路径就是 `/scratch/sunyh/pear`。

| 内容 | 路径 |
| --- | --- |
| PEAR 工作树及本地兼容性修复 | `/scratch/$USER/pear` |
| 训练、rollout 环境，Python 3.12.15 | `pear_work/envs/verl` |
| LIBERO 环境，Python 3.10.22 | `pear_work/envs/libero` |
| Mesa 软件渲染库 | `pear_work/envs/mesa` |
| uv 管理的 Python / micromamba | `pear_work/envs/uv-python`、`pear_work/envs/micromamba` |
| Qwen 模型权重，约 52 GB | `pear_work/models/Qwen3.8-27B` |
| LIBERO / FlashAttention / LMCache 源码 | `pear_work/src/{LIBERO,flash-attention,LMCache}` |
| 带官方补丁的 verl 源码，editable 安装 | `pear/3rdparty/verl/src` |
| 下载缓存 / 编译缓存 / 编译后 wheel | `pear_work/cache`、`pear_work/cache/ccache`、`pear_work/wheels` |
| C++ 运行库 / CUDA 链接 stub | `pear_work/runtime/lib`、`pear_work/runtime/cuda-driver-stubs` |
| Slurm 输出 / 训练结果与 checkpoint | `pear_work/logs`、`pear_work/runs` |
| 每个作业、每个节点自己的 JIT 与 Ray 临时目录 | `/tmp/$USER/pear-<jobid>`，退出时清理 |

本次版本：PEAR `2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2` 加下表修复，
torch `2.14.0+cu130`、vLLM `0.34.0`、Ray `2.58.0`、transformers `5.22.2`。
模型 revision 为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
依赖锁定以 `scripts/install/versions.env`、各环境的 constraints / overrides 和
`3rdparty/*/UPSTREAM` 为准，不要单独升级 torch、vLLM 或 transformers。

以下修改是运行所必需的适配，清理时保留：

| 文件 | 原因 |
| --- | --- |
| `scripts/install/verl/install.sh` | 离线编译时跳过联网 fetch / uv refresh，同时检查固定 commit 存在 |
| `scripts/gpu_process_variables.sh` | 加载可用的 C++ 运行库，设置 CUDA stub 的链接路径及 TileLang 本地缓存 |
| `pear/models/qwen3_5/rollout/hopper/all_reduce.py` | 不支持 NVLink multicast 时全 TP 组一致回退，FP32 求和后一次转 BF16 |
| `tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py` | 验证自动检测与强制回退路径，保留原有 FP64 误差界 |
| `pear/rollout/lmcache.py` | NUMA 空闲预留可配置，默认仍为 100 GiB；Rorqual 显式设为 32 GiB |

迁移时 clone 本文的 fork `drac` 分支，即可获得上述修复和 `deploy/`；
仅 clone 官方仓库不会得到这些修改。不要删除 `src/LIBERO` 或 `3rdparty/verl/src`：
环境通过 editable 安装直接引用它们。`cache` 是可重新生成的数据，`src` 不全是可删的缓存。

## 2. 安装：登录节点下载，CPU 计算节点编译

现有部署已安装完毕，无需重新安装。以下记录从头部署的步骤。
在可联网的登录节点准备源码、Python 包和模型；较重的 CUDA 编译放在 CPU 计算节点，
显式指定 H100 的 `CUDA_ARCH=9.0`，不需要占用 GPU 编译。

### 2.1 官方源码、模拟器、基础环境、模型

```bash
# 新部署才需要 clone；已有当前适配工作树时直接 cd。
cd /scratch/$USER
git clone --branch drac --single-branch https://github.com/yihaosun1124/pear.git
cd pear
# drac 分支包含部署入口及兼容性修复；upstream 仍为官方固定版本。

export WORK_DIR=/scratch/$USER/pear_work
mkdir -p "$WORK_DIR/logs"

# 官方脚本下载 uv、独立 Python、LIBERO 和 Mesa，并验证模拟器初始观测。
bash scripts/install.sh libero

# 将可在此系统运行的 C++ 库留给训练环境使用。
mkdir -p "$WORK_DIR/runtime/lib"
cp -L "$WORK_DIR/envs/mesa/lib/libstdc++.so.6" "$WORK_DIR/runtime/lib/"
cp -L "$WORK_DIR/envs/mesa/lib/libgcc_s.so.1" "$WORK_DIR/runtime/lib/"

NO_GPU=1 bash scripts/install.sh verl
bash scripts/download_model.sh
command -v numactl
# 若集群没有 numactl，使用官方安装目标：bash scripts/install.sh numactl
```

从干净 shell 运行，不要先激活 Conda 或加载一套混用的 CUDA/GCC 模块。
官方安装脚本会清理编译器、Conda 与动态库环境变量，使用自身锁定的环境。

### 2.2 下载 GPU 依赖，准备固定版本源码

仍在联网登录节点执行。`source common.sh` 只设置路径、版本与函数，不执行完整安装。

```bash
export WORK_DIR=/scratch/$USER/pear_work
source scripts/install/common.sh
export UV_CONSTRAINT=$ROOT/scripts/install/verl/constraints.txt
export UV_OVERRIDE=$ROOT/scripts/install/verl/overrides.txt
PYTHON=$WORK_DIR/envs/verl/bin/python

uv pip install --python "$PYTHON" --find-links "$FLASHINFER_CUBIN_WHEELS_URL" \
  -r "$ROOT/pyproject.toml" --extra verl_gpu pip setuptools_scm ninja grpcio-tools aiofile aiofiles
uv pip install --python "$PYTHON" --no-deps --index "$FLASHINFER_JIT_CACHE_INDEX_URL" \
  -r "$ROOT/pyproject.toml" --extra verl_jit_cache
uv pip install --python "$PYTHON" --no-deps --prerelease=allow flash-attn-4

FA=$WORK_DIR/src/flash-attention
if [ ! -d "$FA/.git" ]; then git clone "$FLASH_ATTN_URL" "$FA"; fi
git -C "$FA" fetch origin "$FLASH_ATTN_COMMIT"
git -C "$FA" checkout "$FLASH_ATTN_COMMIT"
git -C "$FA" submodule update --init --depth 1 csrc/cutlass
LMCACHE_SRC=$WORK_DIR/src/LMCache bash 3rdparty/LMCache/scripts/checkout_lmcache.sh
```

FlashAttention 固定在 `eed1971f5132630dc296fe37601e834d4b57a248`，
不能直接用任意 PyPI 版本替代：当前 torch 的头文件要求 C++20。
LMCache 使用官方提供的补丁版本，即使训练时关闭其 sidecar，环境依赖仍需要安装。
复用已构建的 LMCache wheel 时，应在联网阶段用上面的 constraints / overrides 安装该 wheel，
让解析器缓存其完整 runtime 依赖，再提交离线安装检查。仅下载 `aiofile` 不够：
`aiofiles` 是另一个包，还需要 wheel 元数据声明的 cloud / gRPC / Redis 等依赖。
若此时其他安装进程正在修改同一个环境，可使用 `uv pip install --target` 临时目录预缓存这些包，
完成后删除临时目录；不要同时修改正在安装的 venv。

### 2.3 补齐 FlashInfer JIT 链接所需的 driver stub

pip 安装的 CUDA toolkit 没有提供 `-lcuda` 所需的无版本链接文件。
以下只从 NVIDIA 官方归档提取 stub，并验证 SHA-256。

```bash
export WORK_DIR=/scratch/$USER/pear_work
source scripts/install/common.sh
COMPONENT=cuda_cudart-linux-x86_64-13.0.48-archive
ARCHIVE=$WORK_DIR/cache/$COMPONENT.tar.xz
URL=https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/$COMPONENT.tar.xz
SHA256=30ffafae23833dafc965cf4e4d034f8553805586959328c64e2c47b3ef3bcd55
download_checked "$URL" "$SHA256" "$ARCHIVE"
DEST=$WORK_DIR/runtime/cuda-driver-stubs
mkdir -p "$DEST"
tar -xOf "$ARCHIVE" "$COMPONENT/lib/stubs/libcuda.so" > "$DEST/libcuda.so"
tar -xOf "$ARCHIVE" "$COMPONENT/LICENSE" > "$DEST/LICENSE"
printf 'url=%s\nsha256=%s\n' "$URL" "$SHA256" > "$DEST/SOURCE"
```

适配代码仅将这个目录加入 **`LIBRARY_PATH`**，供编译链接使用。
不要加入 `LD_LIBRARY_PATH`，也不要伪造 `libcuda.so.1`；运行时必须加载计算节点真实的驱动。

### 2.4 CPU 计算节点离线编译

在 CPU 集群的登录节点提交。账户和分区按所在集群替换；本次使用
`def-bengioy` / `debug`，32 个编译 CPU、1 小时，`MAX_JOBS=16`。
下面直接从标准输入提交，不需要保留额外的 build 脚本。
支持自定义内存的站点可申请下面的 160G；Trillium 包装器禁止非零 `--mem`，
在该站点需改成 `--mem=0` 整节点分配。`MEMORY_GIB=160` 是编译并行度的预算，不是作业硬上限。

```bash
cd /scratch/$USER/pear
sbatch --account=def-bengioy --partition=debug --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --mem=160G --time=01:00:00 \
  --output=/scratch/$USER/pear_work/logs/build-%j.log <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
export WORK_DIR=/scratch/$USER/pear_work
export CUDA_ARCH=9.0 MAX_JOBS=16 MEMORY_GIB=160 UV_LINK_MODE=copy
export UV_OFFLINE=1 PIP_NO_INDEX=1
export CCACHE_DIR=$WORK_DIR/cache/ccache XDG_CACHE_HOME=$WORK_DIR/cache
mkdir -p "$CCACHE_DIR"
cd /scratch/$USER/pear
bash scripts/install.sh verl
SBATCH
```

完整安装会构建 FA2、Hopper FA3、带补丁的 LMCache，并执行 CPU 环境检查。
若日志显示 wheel 已生成，但由于缺少缓存依赖（曾遇到 `aiofile==3.12.3`、`aiofiles==25.1.0`）而退出，
回到联网登录节点安装已生成的 wheel，随后重新提交上面的安装作业：

```bash
export WORK_DIR=/scratch/$USER/pear_work
source scripts/install/common.sh
export UV_CONSTRAINT=$ROOT/scripts/install/verl/constraints.txt
export UV_OVERRIDE=$ROOT/scripts/install/verl/overrides.txt
mapfile -t WHEELS < <(find "$WORK_DIR/wheels" -path '*/lmcache-*/*.whl')
printf '%s\n' "${WHEELS[@]}"
# 新部署应只有一个匹配；多个版本时明确选择本次 build 日志输出的 wheel。
test "${#WHEELS[@]}" -eq 1
uv pip install --python "$WORK_DIR/envs/verl/bin/python" "${WHEELS[0]}"

export LOCAL_DIR=$WORK_DIR/cache/environment_test
source scripts/gpu_process_variables.sh "$WORK_DIR/envs/verl"
python -m pytest tests/test_training_environment_on_cpu.py
```

不需要删除环境重装；官方 installer 使用输入哈希和缓存 wheel 跳过已完成的构建。
本次 CPU 环境检查为 17/17 通过。

## 3. 提交 smoke test 与正式训练

**从仓库根目录提交**。脚本通过 `SLURM_SUBMIT_DIR` 找到仓库；也可传
`PEAR_ROOT=/absolute/path/to/pear`。参数用脚本的 `NAME=value` 传入，
以避免 DRAC 的 `--export=NONE` 包装器丢掉临时 shell 变量。

脚本默认申请 `--mem=450G`。**Trillium 的 sbatch 包装器不接受此参数**，提交检查返回
`The --mem=... request is not allowed nor necessary`。如要在该站点提交，必须接受整节点分配并
在 `sbatch` 中覆盖 `--mem=0`；此时不具备 450G 的 Slurm 硬限制。
若要求严格的 450G cgroup 上限，应选择支持自定义内存申请的站点。
下面的账户/分区是此次部署的示例，跨集群要替换，Trillium 还需上述内存覆盖。

```bash
cd /scratch/$USER/pear
export WORK_DIR=/scratch/$USER/pear_work
mkdir -p "$WORK_DIR/logs"

# 替换为此 GPU 集群的实际账户与分区。
ACCOUNT=rrg-bengioy-ad
SMOKE_PARTITION=debug_full_node
TRAIN_PARTITION=compute_full_node

sbatch --account="$ACCOUNT" --partition="$SMOKE_PARTITION" \
  --gpus-per-node=h100:4 deploy/smoke_test.sbatch

# 正式 Stage 1：默认申请 24 小时；不会自动重复申请。
sbatch --account="$ACCOUNT" --partition="$TRAIN_PARTITION" \
  --gpus-per-node=h100:4 deploy/train.sbatch

# 指定可恢复的固定 RUN；训练超参数仍保持官方默认值。
sbatch --account="$ACCOUNT" --partition="$TRAIN_PARTITION" --time=04:00:00 \
  --gpus-per-node=h100:4 deploy/train.sbatch \
  RUN="$WORK_DIR/runs/stage1"

# 只组合配置，不启动 Ray、不执行训练、不申请 GPU。
bash deploy/smoke_test.sbatch --config-only
bash deploy/train.sbatch --config-only
```

Slurm 选项放在脚本路径之前，训练参数放在之后。其他 Hydra 参数放在 `--` 后，例如：

```bash
sbatch --account="$ACCOUNT" --partition="$TRAIN_PARTITION" deploy/train.sbatch \
  RUN="$WORK_DIR/runs/stage1" -- trainer.save_freq=5
```

默认每节点 96 CPU、4 GPU，申请 **450G（450 GiB）** 内存；硬限制取决于站点是否支持该请求。
smoke 分配两个节点；正式分配 16 个节点，
前 4 个为 trainer（16 GPU），后 12 个为 rollout（48 GPU）并提供 CPU LIBERO。
第一个 trainer 是 Ray head。启动器验证 Ray 注册了正确的节点数与两类 GPU 数。
Ray 每节点留出 8 CPU 给系统，其余作为调度资源。账户、分区、H100 的资源名称、时间上限、
CPU 数和每节点内存需要按目标集群调整。
若目标节点有 8 张卡，还需确认 Slurm 的设备隔离让官方 Ray 启动器只看到申请的 4 张卡。

默认 Slurm 输出为 `/scratch/%u/pear_work/logs/%x-%j.{out,err}`。
自定义 `WORK_DIR` 时，先创建日志目录，并用 `sbatch --output=... --error=...` 同时覆盖输出位置。

### 默认参数

| 参数 | smoke test | 正式 Stage 1 |
| --- | --- | --- |
| Slurm 时间 | 20 分钟，最短 15 分钟 | 24 小时 |
| Slurm 内存上限 / 每节点 | 450G | 450G |
| 训练节点 / rollout 节点 | 1 / 1 | 4 / 12，每节点 4 卡 |
| 训练 GPU / rollout GPU | 4 / 4 | 16 / 48 |
| FSDP / Ulysses SP / rollout TP | 4 / 4 / 4 | 8 / 8 / 4，沿用官方 H100 |
| 训练数据 | spatial task 0，trial 0–3 | 官方 `libero_gso_all.json`，1500 episodes |
| STEPS / EPOCHS 上限 | 1 / 1 | 1000 / 100，沿用官方 |
| BATCH：每步 prompt groups | 4 | 128，沿用官方 |
| N：每个 group 的轨迹 | 2 | 8，沿用官方 |
| 每个训练 batch 的轨迹 | 8 | 1024 个被接受的轨迹 |
| PPO mini-batch | 4 groups × 2 | 128 groups × 8，整步一个 mini-batch |
| MAX_PROMPT / MAX_RESPONSE | 4096 / 4096 | 4096 / 160000，沿用官方 |
| 每 GPU 动态 micro-batch token cap | 4096 | ceil((MAX_PROMPT + MAX_RESPONSE) / 8) = 20512 |
| SESSIONS_PER_REPLICA / 每节点 LIBERO 服务器 | 8 / 4 | 64 / 32，沿用官方 |
| rollout 副本 / 全局 inflight sessions | 1 / 8 | 12 / 768 |
| MTP speculative tokens | 0 | 3，沿用官方 |
| LMCache sidecar | 关闭 | 开启，64 GiB/副本；官方为 700 GiB |
| rollout GPU memory utilization / enforce_eager | 0.8 / true | 0.85 / false，沿用官方 |
| Ray object store / 每节点 | 8 GiB | 64 GiB，沿用官方启动器 |
| TransferQueue storage units | 2 | 32，沿用官方 |
| 无奖励差异的 group filter | 关闭 | 开启，沿用官方 |
| checkpoint / 恢复 | 每步、保留 1、禁止恢复 | 每步、保留 2、同一 RUN 自动恢复 |

正式脚本只调整资源拓扑为每节点 4 卡、4 个 trainer 节点和 12 个 rollout 节点；
训练与 rollout 默认值沿用固定版本的官方 Stage 1 / H100 recipe，包括 batch、N、响应预算、MTP、
group filter、logger、checkpoint 保留策略。默认将 LMCache 主存缓存从 700 GiB 减至 64 GiB；
Rorqual 还使用用户批准的 NUMA 预留 32 GiB，其他站点默认仍为官方的 100 GiB。
这是系统缓存容量调整，不改变 GRPO、训练数据或训练目标；可能增加缓存淘汰、搬运和重算。
每节点 Ray CPU slots 按 Slurm 的 CPU 分配扣除系统余量。
FSDP8 / SP8 会跨两个 4 卡节点，16 个训练 GPU 的数据并行度为 2。
可通过 `TRAINER_NNODES=... ROLLOUT_NNODES=...` 显式调整角色节点数，但必须同时用
`sbatch --nodes=...` 改变总申请数量；每节点仍是 4 GPU。

**实际通过 GPU 验证的是 smoke 配置；64 卡正式配置尚未执行。**
静态配置合法性与实际硬件容量是两件事，下面记录缓存调整依据及尚未验证的容量边界。

### 450G 主存预算与 LMCache 调整

官方 H100 配置原本针对 8×H100、2 TB 主存节点，每个 TP4 rollout 副本配 700 GiB LMCache。
`pear/rollout/lmcache.py` 的启动检查要求该副本第一张 GPU 对应的**单个 NUMA 域**有
`700 + 100 = 800 GiB` 可用内存，并非全作业内存加起来满足即可。
当前节点 Slurm `RealMemory=770000 MiB`，整机可分配量约 752 GiB，单 NUMA 域更小；
因此未调整的官方缓存配置会在 LMCache 启动时失败，也不满足本部署的每节点 450G 上限。
参考：[固定版本的官方 H100 recipe](https://github.com/mit-han-lab/pear/blob/2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh)、
[LMCache 的 NUMA 内存检查](https://github.com/mit-han-lab/pear/blob/2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2/pear/rollout/lmcache.py)、
[SciNet 节点规格](https://scinethpc.ca/trillium/)。
当前正式脚本默认 `LMCACHE_GB=64`。每个 rollout 节点只有一个 TP4 副本，因此每节点固定缓存是
64 GiB，再加官方 Ray object store 的 64 GiB，为其他进程留下约 322 GiB 的预算。
这不是整节点实际内存占用预测：模型加载、模拟器、Ray worker、临时张量还需从余量中分配。
NUMA headroom 默认仍为官方的 100 GiB，因此对应 NUMA 域启动时需至少 **164 GiB 可用内存**。
迁移到其他 DRAC 集群需单独核对这一项；整机能申请 450G 不代表单 NUMA 域一定够。
可通过 Hydra 的 `+actor_rollout_ref.rollout.custom.lmcache.numa_headroom_gb=...` 显式覆盖；
这是启动容量检查的预留阈值，不是额外分配的缓存，也不会取消 NUMA 绑定或 95% 本地内存检查。
若要改缓存容量，可传 `LMCACHE_GB=...`，不必改训练脚本，例如：

```bash
sbatch --account="$ACCOUNT" --partition="$TRAIN_PARTITION" deploy/train.sbatch \
  RUN="$WORK_DIR/runs/stage1" LMCACHE_GB=64
```

rollout 的显存也不能按全局 48 卡合并计算：每个副本独立使用 TP4。
按当前模型 16 个 full-attention 层、4 个 KV heads、head dim 256、BF16 KV 估算，
仅标准 attention 的 KV 就是每 token 每副本 64 KiB，均分到 4 卡。
一个 164096-token 上下文约占 2.50 GiB/卡；64 条都到上限时约 160 GiB/卡，
还未计 recurrent state、权重、MTP 和临时张量。因此需要 CPU cache、淘汰和调度配合；
64 GiB 缓存不支持让所有满长上下文同时常驻，也不能用这个估算保证满负载吞吐。

训练侧 27B 在 16 卡 FSDP8/SP8 下的参数分片本身可行，长上下文使用官方 activation offload；
160k 序列和 1024 条轨迹完整 batch 的激活、主存与通信开销仍需实测，不能从短 smoke 保证。
SP8 跨节点的通信会比同机 8 卡布局更重。450G 是通用脚本的内存申请目标，不能自动消除这些内存开销；
Trillium 不能通过该参数设置硬限制。
需要在相同限制下实际训练，再通过 Slurm cgroup / sacct 统计验证；此前成功 smoke 使用整节点内存，
不能当成 450G 上限下已经验收通过。

异步 rollout 会预取下一批，所以日志里的累计生成轨迹可以多于实际训练 batch。
开启 group filter 后，奖励完全相同的组会被丢弃并继续采样，生成数量还会进一步增加。
`STEPS` / `EPOCHS` 是上限，作业也可能先到 Slurm wall time；自动恢复只读取已经保存的 checkpoint。
默认每次提交创建新 RUN，要续训需再次传入相同的 `RUN=...`。

## 4. 成功判据与已有验证

整理后的两个入口通过 Bash 语法、ShellCheck 和嵌入 Python 语法检查。
正式配置通过 verl `validate_config`；统一输出路径和实验名后，与官方 recipe 的完整配置比较，
差异仅为所要求的 GPU 拓扑和 LMCache 64 GiB。64 卡及 450G 约束下尚未执行 GPU 训练。
Trillium 的提交检查确认拒绝 `--mem=450G`，不是一次成功的 450G 作业验收。

smoke 先检查 FlashInfer JIT、四卡 all-reduce 数值和 Hopper delta-rule backward，
再启动 Ray、实际 LIBERO rollout、backward、optimizer、checkpoint 和权重同步。
训练退出后检查 step 1 的有限梯度指标，以及四个 rank 各自的 model / optim / extra_state
共 12 个非空文件，写入 `verified-step.json`，输出 `PEAR_RL_STEP_VERIFIED`。

每个 RUN 包含 `train.log`、`metrics_*.jsonl`、`progress.json`、`ckpt/`、模拟器日志，
以及按 job id 记录的源码 diff、启动脚本、硬件、Ray 资源、Ray 日志和 `job-result-<jobid>.json`。
Ray token 文件权限仅限当前用户，不要提交到代码仓库或分享它。

本次原始验证在 Trillium 完成，记录日期为 2026-11-21（Asia/Shanghai）：

- 作业 `1378365`，`debug_full_node`，`trig0017` 训练 + `trig0058` rollout，各 4 H100 80GB。
- Slurm `COMPLETED`，exit `0:0`，02:05:52–02:19:39，耗时 13 分 47 秒。
- 真实 LIBERO 8 条轨迹参与 1 个 RL step；异步预取另完成 8 条，累计生成 16 条。
- FlashInfer SiLU JIT、四卡 all-reduce FP64 数值检查、Hopper backward/recompute 均通过。
- step 1 的 12 个 checkpoint 分片完整；模型和优化器 checkpoint 总计约 306 GiB。
- 四个 rank 各抽取的首个 optimizer step counter 都为 1。
- reward、policy loss、grad norm 都为 0，符合这批短轨迹没有成功奖励的 GRPO 行为。
  这证明执行链路跑通，不代表模型学到了有效策略。
- actor 峰值 allocated / reserved 显存约 69.18 / 75.33 GiB；CPU 指标约 422.43 GiB。
  后者来自 `psutil.virtual_memory().used`，是整节点当时占用，不是作业 cgroup 的独立峰值。

原始证据保留在 `/scratch/sunyh/pear_work/runs/smoke-1378365`，包括
`completion-audit.json`、`verified-step.json`、`slurm-accounting.txt`、`job-result.json`、
`source.patch` 与当时的 `launch/`。此次整理未改写历史运行记录；旧记录中的旧脚本名属于当次快照。
代码目录不再保留重复的 `evidence/` 和临时排障脚本。

## 5. 本次排查出的坑

| 症状 / 限制 | 原因与处理 |
| --- | --- |
| CPU / GPU 队列查到的内容不同 | 部分 DRAC 站点有独立 CPU/GPU Slurm。必须到 GPU 提交主机上执行 sbatch/squeue/sacct。此次 CPU 登录可通过 `ssh -o BatchMode=yes trig-dm1` 访问 GPU 调度器；这是该站点的路由，不写进通用脚本。 |
| 登录节点做了检查，进入 srun 却缺变量 | sbatch 包装器使用 `--export=NONE --get-user-env`。脚本内部重建环境，并设置 `SLURM_EXPORT_ENV=ALL`；参数通过脚本实参传入。 |
| 8 卡 debug 作业申请 30 分钟被拒 | 此次 debug QoS 每作业限制 180 GPU 分钟，8×20=160，8×30=240。smoke 使用 20 分钟；正式训练走普通分区。其他站点需重新查限制。 |
| `QOSMaxSubmitJobPerUserLimit` | debug 并发/提交数受限；不要再挂一个辅助 GPU 检查作业。当前预检在 smoke 自身的两节点分配内运行。 |
| 计算节点连不上 PyPI / GitHub | 先在登录节点缓存包、源码、CUTLASS 和固定 commit，再离线编译。`UV_OFFLINE=1` 不会自动下载缺失的内容。 |
| `--refresh-package` 与 offline 冲突 / git fetch 卡住 | installer 的本地修复在 offline 模式跳过这两步，仍验证固定 commit。 |
| 缺 `pip`、`setuptools_scm`、`ninja` 或 LMCache runtime 依赖 | 先下载构建工具；若 wheel 已构建则联网安装该 wheel 补齐依赖，再重跑安装检查，不必重新编译全部源码。 |
| `GLIBCXX_3.4.30` 缺失 | 系统 libstdc++ 太旧；直接使用 CVMFS 的新版库又可能要求系统没有的 glibc 2.36。使用 Mesa conda prefix 中兼容当前系统的 libstdc++ / libgcc_s，复制到 runtime/lib。 |
| rollout 在 NVLink multicast 初始化失败 | H100 名称不保证有 multicast。所有 TP rank 先一致投票，不支持则使用 vLLM all-reduce + PEAR add/RMSNorm。BF16 中间累加曾超出原误差界，已改为 FP32 求和后一次 BF16 舍入，未放宽数值检查。 |
| 预检意外编译 FlashInfer 通信模块 | 预检不在 Ray runtime_env 内，需要显式匹配 `NCCL_MNNVL_ENABLE=0`、`VLLM_ALLREDUCE_USE_FLASHINFER=0`、`NCCL_LAUNCH_ORDER_IMPLICIT=1`。 |
| FlashInfer JIT 报 `cannot find -lcuda` | pip CUDA toolkit 缺 driver 链接 stub。按 2.3 节补齐，仅加 LIBRARY_PATH；执行用真实 libcuda.so.1。 |
| actor 已初始化、rollout 也完成，却在 backward 失败 | FLA backward 延迟导入 TileLang，默认缓存写只读 home。`TILELANG_CACHE_DIR` 和 `TILELANG_TMP_DIR` 必须指向每节点 LOCAL_DIR；仅检查 import 或模型加载不够。 |
| 多进程共享编译缓存冲突 / 临时目录残留 | JIT、CUDA、Ray 的临时目录按 job id 隔离；退出前归档 Ray 日志、停止 Ray、清理该作业的确切目录。不要假定 /tmp 是独立 SSD；若站点使用 tmpfs，临时文件也计入主存预算。 |
| smoke 一直生成却不训练 | 官方 group filter 会丢弃全部奖励相同的组。minimal smoke 关闭它，使零奖励批次也能完成真实更新流程；正式训练保留官方设置。 |
| `lr_scheduler.step()` 的顺序 warning | streamed AdamW 直接调用底层更新函数，PyTorch 的外层计数可能未识别。不能只凭 warning 判定没有更新；此次独立读取四个 rank 的 optimizer counter，均为 1。 |
| 磁盘用量突然增长 | smoke 四 rank 的模型 + optimizer checkpoint 实测约 306 GiB/step；正式为 16 rank，复制维度与 TransferQueue 快照可能进一步增加体积，不能直接按 306 GiB 预算。默认保留两份，还需保存时的余量。 |

GPU 环境检查应在已分配的 H100 上执行：加载 `scripts/gpu_process_variables.sh` 后运行
`python -m pytest tests/test_training_environment_on_gpu.py`。本次为 3/3 通过；
真实 LIBERO 也执行过 5 个原生动作并返回两张 512×512 图像。
不要把登录节点的无 GPU 导入提示当成训练 GPU 的验收结果。


## 8. 多集群迁移

当前代码源为 [yihaosun1124/pear 的 drac 分支](https://github.com/yihaosun1124/pear/tree/drac)，
由官方仓库 fork；`main` 保持官方代码，部署修复提交在 `drac`。
环境、模型、wheel、运行日志和 checkpoint 不上传 GitHub。

2026-11-21 已通过真实登录、Slurm association 和 `sbatch --test-only` 核对下表。
`--test-only` 只校验请求，不表示已经提交或已经跑通训练。

| 集群 | 仓库 / 运行目录前缀 | GPU account | GPU partition | 每节点 CPU | 内存请求 |
| --- | --- | --- | --- | --- | --- |
| Tamia | `/scratch/s/sunyh` | `aip-plbacon` | `gpubase_bynode_b1` | 48 | `450G` |
| Fir | `/scratch/sunyh` | `rrg-bengioy-ad_gpu` | `gpubase_bynode_b1` | 48 | `450G` |
| Rorqual | `/scratch/sunyh` | `rrg-bengioy-ad_gpu` | `gpubase_bynode_b1` | 64 | `450G` |

这些 H100 节点均为每节点 4 卡。不能沿用原 96 CPU 请求，否则会超出节点资源。
48 CPU 时 Ray 使用 40 CPU，仍满足正式配置的每节点 32 个 LIBERO server；训练超参数不变。
三个站点的 450G 请求均通过提交校验；运行中的 cgroup 上限仍需在实际分配后核对。

Rorqual 的 H100 硬件探测作业 `27335380` 在 `rg21706` 实测 4 个 NUMA 域，
每域约 126 GiB，无法满足默认的 `64 + 100 = 164 GiB` 检查。
该站点经用户批准保留 64 GiB 缓存，将 NUMA 预留改为 32 GiB，即启动时要求至少 96 GiB 可用。
提交 smoke 和正式任务时，在脚本参数最后追加：

```bash
-- +actor_rollout_ref.rollout.custom.lmcache.numa_headroom_gb=32
```

普通 smoke 默认关闭 LMCache；若要同时验证该站的真实 64 GiB 缓存分配，在同一个 `--` 后追加
`actor_rollout_ref.rollout.custom.lmcache.enable=true actor_rollout_ref.rollout.custom.lmcache.l1_size_gb=64`。
容量检查通过仍需实测缓存 pinning、NUMA 放置和 RL step；不能据此宣称正式负载已验证。

```bash
# 在目标集群登录节点执行。按上表填写 ACCOUNT 和 CPUS。
ROOT=$SCRATCH/pear
WORK=$SCRATCH/pear_work
ACCOUNT=rrg-bengioy-ad_gpu  # Tamia 改为 aip-plbacon
CPUS=48                   # Rorqual 改为 64
mkdir -p "$WORK/logs"
cd "$ROOT"
sbatch --account="$ACCOUNT" --partition=gpubase_bynode_b1 \
  --cpus-per-task="$CPUS" --gpus-per-node=h100:4 --mem=450G \
  --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" \
  deploy/smoke_test.sbatch PEAR_ROOT="$ROOT" WORK_DIR="$WORK"

# 只有该集群的 smoke 验证真实 step、指标和完整 checkpoint 后才运行正式任务。
sbatch --account="$ACCOUNT" --partition=gpubase_bynode_b1 --time=02:00:00 \
  --cpus-per-task="$CPUS" --gpus-per-node=h100:4 --mem=450G \
  --output="$WORK/logs/%x-%j.out" --error="$WORK/logs/%x-%j.err" \
  deploy/train.sbatch PEAR_ROOT="$ROOT" WORK_DIR="$WORK"
```

在各站重新创建 Python 环境，避免复制含绝对路径的 venv 和 editable 安装。
同为 x86_64 / H100 / 相同 Python、torch、源码和补丁版本时，可复用本部署已验证的
FA2、FA3、LMCache wheel；先核对 SHA-256、编译输入和 ABI，再进行 CPU 环境检查和该站 GPU smoke。
不要复用跨架构的二进制或旧 JIT 缓存；缺少匹配 wheel 时按第 2 节在 CPU 计算节点重新编译。
