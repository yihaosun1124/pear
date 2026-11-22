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
| 每个作业、每个节点自己的 JIT 与 Ray 临时目录 | 优先磁盘上的 `$SLURM_TMPDIR/pear-<jobid>`；Ray 使用同一作业磁盘的 `$SLURM_TMPDIR/r`。无合适的 Slurm 目录时按启动器选择 `/localscratch/$USER` 或磁盘 `/tmp` |

本次版本：PEAR `2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2` 加下表修复，
torch `2.14.0+cu130`、vLLM `0.34.0`、Ray `2.58.0`、transformers `5.22.2`。
模型 revision 为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
依赖锁定以 `scripts/install/versions.env`、各环境的 constraints / overrides 和
`3rdparty/*/UPSTREAM` 为准，不要单独升级 torch、vLLM 或 transformers。

以下修改是运行所必需的适配，清理时保留：

| 文件 | 原因 |
| --- | --- |
| `scripts/install/verl/install.sh` | 离线编译时跳过联网 fetch / uv refresh，同时检查固定 commit 存在 |
| `scripts/gpu_process_variables.sh` | 加载可用的 C++ 运行库，设置 CUDA stub / TileLang 路径，跳过共享 bytecode 写入，并清理 NVIDIA 进程中的 AMD 设备变量 |
| `pear/models/qwen3_5/rollout/hopper/all_reduce.py` | 不支持 NVLink multicast 时全 TP 组一致回退，FP32 求和后一次转 BF16 |
| `tests/models/test_qwen3_5_rollout_kernels_hopper_on_gpu.py` | 验证自动检测与强制回退路径，保留原有 FP64 误差界 |
| `pear/rollout/lmcache.py` | NUMA 空闲预留可配置，默认仍为 100 GiB；Rorqual 显式设为 32 GiB |
| `3rdparty/verl/patches/0054-data-fix-preserve-the-token-axis-of-uniform-mRoPE-position-batches.patch` | 统一 mRoPE 位置张量的 token 轴，保留正确的 values、offsets 和序列化元数据 |

迁移时 clone 本文的 fork `drac` 分支，即可获得上述修复和 `deploy/`；
仅 clone 官方仓库不会得到这些修改。不要删除 `src/LIBERO` 或 `3rdparty/verl/src`：
环境通过 editable 安装直接引用它们。`cache` 是可重新生成的数据，`src` 不全是可删的缓存。

## 2. 安装：登录节点下载，CPU 计算节点编译

已完成该站部署的环境无需重复安装。以下记录从头部署的步骤。
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
完成后删除临时目录；不要同时修改正在安装的 venv。但这不能代替离线解析检查：
Fir 实测临时目录安装成功后，离线解析仍会按 registry 元数据查找 `aiofiles` 并失败。
此时应保存显式 wheel 目录，并在 CPU 作业中使用 `--no-index --find-links`。

下面为当前固定版本准备缺少的 13 个 LMCache runtime 包；其余依赖来自已安装的 GPU 环境。
在联网阶段、GPU 依赖安装结束后执行。版本和文件哈希直接从官方 constraints 提取，
不使用另一套版本约束；升级 LMCache wheel 后需重新核对这份缺包列表。

```bash
export RUNTIME_WHEELS=$WORK_DIR/wheels/lmcache-runtime
mkdir -p "$RUNTIME_WHEELS"
"$PYTHON" - "$ROOT/scripts/install/verl/constraints.txt" "$RUNTIME_WHEELS/requirements.txt" <<'PY'
import pathlib
import sys

needed = {
    "aiofile", "aiofiles", "awscrt", "cufile-python", "google-cloud-bigtable",
    "google-cloud-core", "google-cloud-monitoring", "google-crc32c",
    "grpc-google-iam-v1", "grpcio-observability", "grpcio-status", "redis",
    "sortedcontainers",
}
lines = pathlib.Path(sys.argv[1]).read_text().replace("\\\n", "").splitlines()
selected = [line for line in lines if line.split("==", 1)[0] in needed]
assert {line.split("==", 1)[0] for line in selected} == needed
assert all("--hash=sha256:" in line for line in selected)
pathlib.Path(sys.argv[2]).write_text("\n".join(selected) + "\n")
PY
"$PYTHON" -m pip download --no-deps --require-hashes --only-binary=:all: \
  -r "$RUNTIME_WHEELS/requirements.txt" -d "$RUNTIME_WHEELS"
(
  cd "$RUNTIME_WHEELS"
  sha256sum ./*.whl > SHA256SUMS
)
```

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

若采用上述显式 runtime wheel 目录，在 CPU 作业中、调用官方 installer 之前，
先对选定的 LMCache wheel 做完整离线解析并安装。此处沿用上面选定的 `WHEELS[0]`，
`RUNTIME_WHEELS` 要指向实际保存的目录：

```bash
export RUNTIME_WHEELS=$WORK_DIR/wheels/lmcache-runtime
(cd "$RUNTIME_WHEELS" && sha256sum --check SHA256SUMS)
uv pip install --dry-run --offline --no-index --find-links "$RUNTIME_WHEELS" \
  --python "$WORK_DIR/envs/verl/bin/python" "${WHEELS[0]}"
uv pip install --offline --no-index --find-links "$RUNTIME_WHEELS" \
  --python "$WORK_DIR/envs/verl/bin/python" "${WHEELS[0]}"
bash scripts/install.sh verl
```

`--no-index` 仅用于这次本地 LMCache 安装，不全局覆盖后续安装器的索引设置。

若依赖安装已完成、CPU 作业在最终验证阶段达到时限，可直接验证现有环境。
Fir 的一次 1 小时 CPU 作业在共享文件查询中超时；重提验证时使用 8 CPU、64G、3 小时，
把以下 17 项环境检查、18 项 LMCache 检查、6 项位置张量回归和 6 项依赖元数据回归放进同一 Python 进程，
减少重复冷导入。正式 64 卡训练仍申请 2 小时。临时缓存放在计算节点本地：

```bash
export LOCAL_DIR=${SLURM_TMPDIR:-/tmp/$USER}/pear-cpu-$SLURM_JOB_ID
mkdir -p "$LOCAL_DIR"
source scripts/gpu_process_variables.sh "$WORK_DIR/envs/verl"
python -m pytest -vv --durations=20 tests/test_training_environment_on_cpu.py \
  tests/rollout/test_lmcache_on_cpu.py tests/test_verl_position_ids_on_cpu.py \
  tests/test_package_metadata_on_cpu.py
```

Fir 的调用栈还定位到 Transformers 导入时的 `importlib.metadata.packages_distributions()`：
Python 3.12 在推断顶层包名之前，会查询 RECORD 中每个文件是否存在，数万份 PyTorch 头文件和
cubin 的 Lustre 查询会拖慢每个进程。GPU 环境脚本默认启用 `PEAR_FAST_PACKAGE_METADATA=1`，
由启动 hook 对每个候选包名查到一个现存文件就停止该包的重复查询；首个文件缺失时继续寻找，
整个包缺失时仍不返回它。声明的顶层包名、namespace 分发顺序和旧式 metadata 保留原语义，
不修改已安装 wheel 或 RECORD，不跳过版本检查；其他 Python 版本沿用原函数。
设为 `0` 可做原函数对照。在同一 Tamia 环境的 290 个 distribution 上，两种方法得到完全相同的
279 项顶层包映射，分别耗时 2.23 秒和 33.64 秒；Fir 的文件查询延迟更高，实际启动仍由 smoke 验证。

本地目录为 XFS、ext 或其他支持的磁盘文件系统时，GPU 环境脚本将 `PYTHONPYCACHEPREFIX`
设到 `$LOCAL_DIR/python-bytecode`，允许在该节点磁盘上生成 bytecode，供同一节点后续进程复用。
共享文件系统仍禁止新 bytecode 写入；不把编译缓存或环境复制到其他集群。

CPU 验证作业内调用 `sbatch` 时，Slurm 可能把已导出的 JIT 缓存变量一起传给 GPU 作业。
Fir 的第一次 smoke `4899948` 因此在 FlashInfer 预检访问旧 CPU 作业的
`/localscratch/sunyh.4894675.0`，报 `PermissionError`，尚未进入 RL 训练。
现在每个 Slurm 作业都会按自己的 `LOCAL_DIR` 重设 vLLM、Inductor、Triton、TileLang、
Torch extensions、CUDA、FA4 和 FlashInfer 的节点缓存；batch shell 在生成 Ray token 前也初始化这些变量，
清理阶段同样使用当前作业的路径。共享的 `HF_HOME` 和 `CCACHE_DIR` 保留。
回归检查为 `python tests/test_gpu_process_variables_on_cpu.py`：覆盖父作业路径泄漏、GPU 分配保留、
非 Slurm 显式覆盖和不支持磁盘的 Python 缓存回退。

Ray 的 Unix socket 完整路径不能超过 107 字节。Fir 的 `5061338` 已通过 Hopper backward、
FlashInfer 和四卡 allreduce，却因
`/localscratch/sunyh.5061338.0/pear-5061338/ray/session_.../sockets/plasma_store`
过长，在 Ray head 启动时失败，尚未执行 RL 更新。
启动器现在调用 `pear_set_ray_tmpdir`：若 `LOCAL_DIR` 直接位于当前作业的 `SLURM_TMPDIR` 下，
Ray 使用该私有目录中的短子目录 `r`；否则仍使用已选择的磁盘 `LOCAL_DIR/ray`。
路径由 batch shell 选定并传给所有 step，不会移入 tmpfs；Ray 日志归档读取实际的 `RAY_TMPDIR`。
回归检查通过真实 Unix socket 绑定覆盖原长路径失败、短路径成功及磁盘回退，
CPU 计算作业还应使用该路径启动 Ray 并执行一个独立任务，再发布新版本的启动缓存。

在 CPU 计算作业内完成全部检查后，可保存这个节点已经生成的 bytecode，供该集群 GPU 作业使用。
下面的发布命令必须放在成功的 pytest 命令之后，并由 `set -e` 保证失败时不执行。
它只保存 Python bytecode；原生库仍保留在原环境，GPU 启动时顺序预读以减少随机缺页 I/O。
Fir 实测顺序读取 67 个已映射原生库、共 3.65 GB 用时 8.53 秒，代码和环境内容均未改变。

```bash
python - <<'PY'
import csv, hashlib, json, os, pathlib, subprocess, sys, tarfile, time
work = pathlib.Path(os.environ["WORK_DIR"])
prefix = pathlib.Path(sys.pycache_prefix)
assert prefix.is_relative_to(pathlib.Path(os.environ["LOCAL_DIR"]))
cache = work / "cache"
cache.mkdir(exist_ok=True)
temporary = cache / (".python-bytecode-" + os.environ["SLURM_JOB_ID"] + ".tar.gz.partial")
def bytecode_only(member):
    return member if member.isdir() or (member.isfile() and member.name.endswith(".pyc")) else None
with tarfile.open(temporary, "w:gz") as bundle:
    bundle.add(prefix, arcname=".", filter=bytecode_only)
digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
archive = cache / ("python-bytecode-" + digest + ".tar.gz")
os.replace(temporary, archive)
env = pathlib.Path(sys.prefix).resolve()
site = env / "lib/python3.12/site-packages"
native = set()
for record in site.glob("*.dist-info/RECORD"):
    for row in csv.reader(record.read_text().splitlines()):
        if row and ".so" in pathlib.PurePosixPath(row[0]).name:
            library = (site / row[0]).resolve()
            if library.is_relative_to(env):
                native.add(str(library))
ready = {
    "cluster": os.environ["SLURM_CLUSTER_NAME"],
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "python_env": str(env), "archive": str(archive), "sha256": digest,
    "native_libraries": sorted(native), "created_at": time.time(),
}
pointer = cache / "python-bytecode-ready.json"
temporary = pointer.with_suffix(".json.partial")
temporary.write_text(json.dumps(ready, indent=2) + "\n")
os.replace(temporary, pointer)
PY
```

启动脚本只接受相同 Slurm 集群、环境路径和仓库 commit 的 seed，核对 SHA256 后在每个 GPU 节点
提取到该作业的节点磁盘，再预读环境中的原生库；源文件仍由 Python 正常校验。缺少 seed 或版本不符时
沿用普通启动，不复制 venv，不跨集群复用此缓存，不改变模型、算子、训练参数或 checkpoint 格式。

Fir 的后续 smoke `4903543` 在 30 分钟预算内只完成了 FlashInfer 和训练反向算子预检，
没有进入 RL；其中反向算子检查耗时 1261 秒。有效 bytecode 仍不能避免 Transformers 为
lazy import 索引递归打开源码。可在 CPU 计算作业内为该环境建立**相同内容的源码快照**，
供每个计算节点在自己的磁盘读取；不修改已安装包，不复制环境或 GPU JIT 缓存。

```bash
SOURCE_CACHE=pear/utils/sitecustomize/pear_source_cache.py
python "$SOURCE_CACHE" build "$ROOT" "$WORK_DIR"
/usr/bin/python3 -B -S "$SOURCE_CACHE" stage-candidate "$ROOT" "$WORK_DIR" "$LOCAL_DIR"
# 新进程的 startup hook 此时会选择本地源码；检查全部依赖及启动回归。
python -m pytest tests/test_training_environment_on_cpu.py tests/rollout/test_lmcache_on_cpu.py \
  tests/test_verl_position_ids_on_cpu.py tests/test_package_metadata_on_cpu.py \
  tests/test_gpu_process_variables_on_cpu.py tests/test_source_cache_on_cpu.py
# 仅在检查成功后发布；外层脚本必须 set -e。
mv "$WORK_DIR/cache/transformers-source-candidate.json" "$WORK_DIR/cache/transformers-source-ready.json"
# 运行时 startup hook 已变化，随后重新生成上文的 bytecode seed，不能只修改旧 seed 的 commit。
```

快照逐文件保存 SHA256，包含源码和包资源，排除 bytecode。GPU 启动时要求相同集群、
环境路径和 commit，检查安装包 RECORD、归档及提取文件的哈希，拒绝越界路径或链接，
然后原子发布本地目录。startup hook 只对匹配的 Python 环境和 cache tag 启用它；
LIBERO 的另一环境、缺失或不匹配快照均沿用原包，distribution metadata 仍来自原环境。
Fir 的 `4923135` 在 1 小时后仍停留在启动阶段，未完成 RL。Hopper 反向算子和 FlashInfer
已通过；只读 strace 确认，虽然 `.pyc` 已命中本地缓存，OmegaConf 等普通包仍会在共享盘
探测 `.so` 后缀并检查源文件，部分查询约 0.4–0.8 秒。源码快照因此也包含已安装的
pytest、pluggy、Hydra、OmegaConf、ANTLR 等纯 Python 启动包及资源；安装包 metadata
仍来自原环境，快照拒绝原生扩展，不复制 venv。更改后重新运行完整 CPU 检查再发布缓存。
Rollout 的 FlashInfer 与四卡 allreduce 预检现在在同一个 Python 进程内顺序执行，复用已导入的
模块，并保留分开的日志和全部断言；训练节点的反向算子检查仍并行执行。
Fir 首次 cold JIT 的 smoke 使用 2 小时预算；正式任务仍固定 2 小时，所有 GPU 预检保留。

验证作业应为 TERM/INT 设置非零退出，并以 `sacct` 的最终状态核对结果。
旧作业 `4838738` 的 EXIT trap 曾在 Slurm TIMEOUT 后写出 `exit_code=0`，原收据保留，
它不能证明这些检查已完成。

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

# 正式 Stage 1：默认 8 节点、32 卡、2 小时；不会自动重复申请。
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
  RUN="$WORK_DIR/runs/stage1" -- trainer.max_actor_ckpt_to_keep=1
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
| Slurm 时间 | 20 分钟，最短 15 分钟 | 2 小时 |
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
| checkpoint / 恢复 | 每步、保留 1、禁止恢复 | 每步、保留 1、同一 RUN 自动恢复 |

正式脚本使用每节点 4 卡、2 个 trainer 节点和 6 个 rollout 节点，共 8 卡 actor 更新和 24 卡 rollout。
训练与 rollout 默认值沿用固定版本的官方 Stage 1 / H100 recipe，包括 batch、N、响应预算、MTP、
group filter 和 logger；checkpoint 按用户要求改为每步保存、只保留最新一份。
LMCache 主存缓存从 700 GiB 减至 64 GiB；
Rorqual 还使用用户批准的 NUMA 预留 32 GiB，其他站点默认仍为官方的 100 GiB。
这是系统缓存容量调整，不改变 GRPO、训练数据或训练目标；可能增加缓存淘汰、搬运和重算。
每节点 Ray CPU slots 按 Slurm 的 CPU 分配扣除系统余量。
FSDP8 / SP8 跨两个 4 卡节点；当前 8 个训练 GPU 构成一个分片组。
可通过 `TRAINER_NNODES=... ROLLOUT_NNODES=...` 显式调整角色节点数，但必须同时用
`sbatch --nodes=...` 改变总申请数量；每节点仍是 4 GPU。

**实际通过完整 RL/checkpoint 验证的是 smoke 配置；32 卡正式长序列负载仍需运行验收。**
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
需要在相同限制下实际训练，再通过 Slurm cgroup / sacct 统计验证。
Trillium 原始 smoke 使用整节点内存；下述 Rorqual smoke 已在 450G 硬限制下完成，
其短序列、4 卡训练结果仍不能保证 64 卡正式长上下文负载。

异步 rollout 会预取下一批，所以日志里的累计生成轨迹可以多于实际训练 batch。
开启 group filter 后，奖励完全相同的组会被丢弃并继续采样，生成数量还会进一步增加。
`STEPS` / `EPOCHS` 是上限，作业也可能先到 Slurm wall time；自动恢复只读取已经保存的 checkpoint。
默认每次提交创建新 RUN，要续训需再次传入相同的 `RUN=...`。

正式任务的 checkpoint 要按实际 FSDP mesh 检查。本版本 FSDP2 HSDP 会去重复制维度：
16 个训练 rank、FSDP8 的常规 2×8 mesh 中，仅 rank 0–7 写 model / optim 分片，
全部 16 个 rank 各写一份 extra_state，即 32 个核心分片，另外还有 HF 元数据和 TransferQueue 状态。
这些文件的 world size 仍是 16；不能要求每个 rank 都有重复的 model / optim，
也不能用 world-size-4 的 smoke checkpoint 恢复正式训练。
恢复前核对实际 mesh 的分片拥有者、所有 rank 的 extra_state、训练 step 和完整 TransferQueue 快照。

按用户要求，每步仍保存 checkpoint，每个 RUN 只保留最新一份；这项保留策略覆盖官方的两份默认值。
存储按普通完整文件预留：smoke 实测约 306 GiB/份，独立 smoke 加一份正式 checkpoint
的核心文件稳态合计约 612 GiB，另留 TransferQueue、日志和安装缓存余量；正式大小仍需实测。
注意保存峰值：当前 checkpoint manager 在 `max_actor_ckpt_to_keep=1` 时，会先完整保存新份，
成功后才删旧份，因此保存瞬间仍有两份正式核心文件，约 612 GiB；若同时保留独立 smoke，
核心文件的总峰值仍约 918 GiB。这与最终只保留一份是不同的容量要求。
其中模型主参数以 FP32 保存，约 102 GiB；AdamW 的一阶、二阶动量也以 FP32 保存，合计约
204 GiB。即使只训练一步，保存的也是完整状态。分片总量已包含所有写文件的 rank，不能再乘卡数。

TransferQueue 快照还包含多模态 `pixel_values`、token、位置编码和 score-centering 使用的
`rollout_topk_ids` / `rollout_topk_log_probs`。实测后两项分别为 int32 / FP32，每个 token 各有
32 项，合计 256 字节/token。以 1024 条、每条 164096 token 的允许长度场景计算，仅这两项就约
40.1 GiB/份；这是场景估算，实际大小取决于完成的轨迹长度、图像数及队列中保存的样本数。
不能从短 smoke 的队列大小推断正式长轨迹的额外空间。

不使用稀疏存储，也不把零梯度带来的空间节省计入预算。已有 checkpoint 若曾做过文件系统稀疏整理，
恢复为普通完整文件时须保留逐字节内容、完整 SHA256、权限、mtime 和优化器状态，并在验证后原子替换。
重新核算时，已有 smoke 已计入 quota.used，不要再次从 quota.free 扣除；同时确认配额用量已完成结算。

节点临时目录也要区分内存盘和磁盘。启动器优先使用磁盘上的 `SLURM_TMPDIR`，否则使用
`/localscratch/$USER`；没有该目录时才检查并回退到磁盘上的 `/tmp`。Fir 实测 `/tmp` 是 tmpfs，
`/localscratch` 是本地磁盘。JIT 缓存、Ray 临时文件若放在 tmpfs，会占用作业的内存预算。
对只有内存盘的站点，可由站点配置显式传 `ALLOW_TMPFS_LOCAL=1`，沿用该站原有临时目录部署；
默认值为 0，其他站仍要求磁盘。Cluv 的 Trillium GPU profile 使用此选项且申请 `mem=0` 整节点，
不能把它解释成 450G 内存硬上限。

四站使用相同的 `deploy/train.sbatch` 和 `scripts/{start_ray_node,gpu_process_variables}.sh`。
站点 account、分区、CPU、内存由 Cluv 管理项目的 `pyproject.toml` 提供，路径、NUMA 与临时盘策略
由 `clusters.json` 提供。更新启动文件不意味着重装环境或覆盖模型/算法兼容补丁。
已经提交的 Slurm batch 副本不会随仓库文件更新；当前运行保留自己的启动快照，下一次统一入口提交才使用新脚本。
同一个 Lustre 文件系统可能承载多个目录配额，`lfs quota -u` 的跨目录用户用量不能代替目标目录配额。
支持 Lustre project quota 的站点可这样检查实际工作目录：

```bash
PROJECT_ID=$(lfs project -d "$WORK_DIR" | awk '{print $1}')
[[ $PROJECT_ID =~ ^[0-9]+$ ]]
lfs quota -p "$PROJECT_ID" -h "$WORK_DIR"
```

## 4. 成功判据与已有验证

整理后的两个入口通过 Bash 语法、ShellCheck 和嵌入 Python 语法检查。
正式配置通过 verl `validate_config`；统一输出路径和实验名后，与官方 recipe 的完整配置比较，
默认差异为所要求的 GPU 拓扑和 LMCache 64 GiB；Rorqual 另使用批准的 NUMA 预留 32 GiB。
64 卡正式负载尚未执行；450G 限制下的 Rorqual smoke 结果记录在下方。
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

2026-11-21 的 Rorqual 验证：

- smoke 作业 `27340419`，`rg32001` 训练 + `rg32003` rollout，各 4 H100；`COMPLETED/0:0`，耗时 17 分 45 秒。
- 1 个 RL step 消耗 8 条轨迹，异步预取另生成 8 条；12 个 world-size-4 checkpoint 分片共 328288668516 字节，约 306 GiB。
- 独立读取四个 optimizer 分片，每个 rank 的 1184 个 step counter 均为 1；梯度指标有限。
  本批奖励、优势、梯度范数均为 0，仅验证更新链路，不代表策略学习效果。
- LMCache 实际 pin 完整 64 GiB，指定 NUMA 域占进程内存约 99.8%，保留 32 GiB 启动预留。
- 两节点 cgroup `memory.max` 均为 483183820800 字节（450 GiB）。保存 checkpoint 期间采样的训练节点峰值为 448.14 GiB，
  这是已观测到的峰值下限，非退出前最终峰值；内存余量很小，正式负载仍需监控。
- 证据在该站 `pear_work/runs/smoke-27340419/` 的 `independent-audit-27340419.json`、`verified-step.json`、
  `numa-placement-27340419.log`、`memory-27340419.jsonl` 和原始日志中。

前一次 Rorqual smoke `27337745` 在 worker 初始化时因 CUDA / ROCR 变量冲突失败；修复后才重提，失败日志保留。
正式作业 `27340420` 已按 16 节点 × 4 H100、16 卡训练 + 48 卡 rollout、每节点 450G、2 小时提交，
并在 smoke 成功后解除依赖，训练超参数沿用本文正式配置。

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
| 已用临时 `--target` 预缓存，离线仍称 `aiofiles` 需要下载 | 缓存中的 registry 元数据不一定满足实际解析路径。按 2.2 节保存带官方哈希的 runtime wheels，再用局部 `--no-index --find-links` 完成 LMCache 离线解析和安装；目录存在本身不能证明解析成功。 |
| `GLIBCXX_3.4.30` 缺失 | 系统 libstdc++ 太旧；直接使用 CVMFS 的新版库又可能要求系统没有的 glibc 2.36。使用 Mesa conda prefix 中兼容当前系统的 libstdc++ / libgcc_s，复制到 runtime/lib。 |
| rollout 在 NVLink multicast 初始化失败 | H100 名称不保证有 multicast。所有 TP rank 先一致投票，不支持则使用 vLLM all-reduce + PEAR add/RMSNorm。BF16 中间累加曾超出原误差界，已改为 FP32 求和后一次 BF16 舍入，未放宽数值检查。 |
| 预检意外编译 FlashInfer 通信模块 | 预检不在 Ray runtime_env 内，需要显式匹配 `NCCL_MNNVL_ENABLE=0`、`VLLM_ALLREDUCE_USE_FLASHINFER=0`、`NCCL_LAUNCH_ORDER_IMPLICIT=1`。 |
| FlashInfer JIT 报 `cannot find -lcuda` | pip CUDA toolkit 缺 driver 链接 stub。按 2.3 节补齐，仅加 LIBRARY_PATH；执行用真实 libcuda.so.1。 |
| actor 已初始化、rollout 也完成，却在 backward 失败 | FLA backward 延迟导入 TileLang，默认缓存写只读 home。`TILELANG_CACHE_DIR` 和 `TILELANG_TMP_DIR` 必须指向每节点 LOCAL_DIR；仅检查 import 或模型加载不够。 |
| 多进程共享编译缓存冲突 / 临时目录残留 | JIT、CUDA、Ray 的临时目录按 job id 隔离；退出前归档 Ray 日志、停止 Ray、清理该作业的确切目录。不要假定 /tmp 是独立 SSD；若站点使用 tmpfs，临时文件也计入主存预算。 |
| Ray head 报 `AF_UNIX path length cannot exceed 107 bytes` | GPU 预检通过仍可能在创建 Ray socket 时失败。保持节点磁盘和作业隔离，使用作业私有 `SLURM_TMPDIR/r` 缩短前缀，归档使用实际 `RAY_TMPDIR`；不要通过把 Ray 数据移入 tmpfs 解决路径长度。 |
| CPU 检查很慢并输出 `Timeout (0:05:00)!` 栈 | pytest 的 300 秒 faulthandler 只打印诊断栈，不会终止测试。Fir 实测栈停在 importlib 的 `.pyc` 写入与目录查询，进程等待 Lustre RPC 且读写量继续增长。共享文件系统禁止新 bytecode 写入；有节点磁盘时使用本地 `PYTHONPYCACHEPREFIX`。先查真实进程、退出码和最终测试结果，不要把这条诊断当失败后重启安装。 |
| 禁用 `.pyc` 写入后仍长时间停在 collecting | 查看调用栈是否在 `packages_distributions → Distribution.files → Path.exists` 遍历头文件。第 2.4 节的元数据 hook 合并同一顶层包的重复查询；依赖识别与原函数对照一致，不修改包版本或关闭训练模块。保留当前活跃验证，新启动进程经 GPU 环境脚本使用修复。 |
| NVIDIA worker 报 `Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set` | Rorqual 的 GPU 预检已通过，但 Ray worker 收到了同时存在的 CUDA / ROCR 设备变量。NVIDIA 运行环境在每个 Slurm step 内加载 `gpu_process_variables.sh` 时清除 `ROCR_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES`，保留 CUDA 分配；不修改 GPU 数量或训练参数。 |
| 登录节点安装 GPU 依赖被 `Killed`，退出 137 | Tamia 在完成大包下载后安装阶段被 SIGKILL；日志本身不能证明具体 OOM 原因。保留模型、venv 和已完成的 uv 缓存，在有明确 CPU / 内存配额的计算作业中修复安装；可能写到一半的包强制重装，不从零清空环境。 |
| 更新 actor 时 `split_with_sizes` 报序列长度 5140、分片长度却都是 4 | 等长 mRoPE 位置张量可能沿通道轴打包。只修改 `_ragged_idx` 会让存储和 offsets 指向不同轴。补丁 0054 使用公开的 `jagged_dim` 构造参数，并在修复序列化元数据时按真实打包轴重建 token 布局。六项 CPU 回归覆盖等长、变长、重复索引及 TensorDict 序列化；该站真实 RL 更新仍需 smoke 验证。 |
| 结果 JSON 为 0，但 CPU 验证日志只有部分测试 | 核对 `sacct`：TIMEOUT/CANCELLED 优先于旧 EXIT trap 的结果文件。依赖完整时可按 2.4 节合并验证，记录逐测试进度并正确处理 TERM；不要据此放行 GPU 训练。 |
| LMCache 的两个 0.5 秒超时单测找不到 record，或缺少本次 partial-progress 行 | 慢文件系统上的模拟 Python 尚未启动，测试已经耗尽 pinning 的短时限。超时测试 fixture 等待本次模拟进程输出首条 partial-progress 行后，才调用原生产 pinning 等待函数；0.5 秒断言、杀进程和旧日志隔离断言都保留。生产的 1800 秒 pinning 时限、NUMA 检查和 64 GiB 缓存设置不变。 |
| smoke 一直生成却不训练 | 官方 group filter 会丢弃全部奖励相同的组。minimal smoke 关闭它，使零奖励批次也能完成真实更新流程；正式训练保留官方设置。 |
| `lr_scheduler.step()` 的顺序 warning | streamed AdamW 直接调用底层更新函数，PyTorch 的外层计数可能未识别。不能只凭 warning 判定没有更新；此次独立读取四个 rank 的 optimizer counter，均为 1。 |
| 磁盘用量突然增长 | 普通 checkpoint 约 306 GiB/份。每个 RUN 最终保留一份，但安全保存会短暂保留新旧两份，正式核心峰值约 612 GiB；另留独立 smoke 时峰值约 918 GiB，TransferQueue 等另算。不使用稀疏存储。 |

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
48 CPU 时 Ray 使用 44 CPU，另留 4 CPU 给 Ray 外的进程。正式 rollout 的 placement group
每张 GPU 预留 2 个 CPU：32 个 LIBERO server + 4×2 个 GPU worker CPU + 至少 1 个控制进程
需要 41 个 Ray CPU。旧的 40 CPU 配置在 Fir 正式任务 5100692 上只启动了 191/192 个 server，
最后一个因 CPU 资源不足排队至 600 秒超时；短 smoke 的 4 server/node 未触发该问题。
当前启动检查要求至少 41 个可调度 CPU；GPU、Slurm CPU 申请与训练超参数不变。
`start_ray_node.sh` 也会在已获 Slurm 48 CPU 的范围内修正旧脚本传来的 40-slot 预算，
因此尚在排队的旧 sbatch 快照可使用此运行时修复，无需仅为更新脚本重新排队。
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
