# 快速上手

配置环境，init官方conda之后，用ai帮忙把pytorch和cuda版本修正到128，适配5090，验证用根目录verify_gpu_training.py可以直接跑。

*mujoco-py 默认装成了 CPU/OSMesa 软件渲染，在 5090 上必须手动改成 EGL GPU 渲染，否则 env 会慢 60 倍。改法见文末「mujoco-py EGL GPU 渲染改造」。

首先运行metaworld的training使用如下CLI：
python train_mw.py task=coffee-push agent=drm_mw

## config文件

1、主要关注agent基础配置DrM/cfgs/agent/drm_mw.yaml、不同的task在DrM/cfgs/task下都有会有个基础yaml配置，包含任务名或者一些覆盖参数。

另外，对某个具体task, e.g. soccer，这种时候会继承来自：

- esay
- medium
- hard
- metaworld

这些是来自上一级的yaml配置。比如，soccer继承了medium。

2. action repeat

控制policy在输出action时，输出两个连续且同样的action，连续执行两次后再返回latest obs。

所以对于这个量，会让其他的如num_train_frames实际的触发采集obs的步数是 num_train_frames / action_repeat

---

# mujoco-py EGL GPU 渲染改造（2026-08-28，drm-cu128 环境）

## 问题

`conda_env.yml` 装出来的 mujoco-py 编译产物是 `cymj_2.0.2.13_39_linuxcpuextensionbuilder_39.so`，
即 **OSMesa 软件光栅化**，离屏渲染完全跑在 CPU 上（并行时 GPU 显存增量恒为 0）。
DrM 是像素输入的 visual RL，每个 env step 都要 render 一张 84x84，于是渲染吃掉了 96% 的 env 时间。

注意两个容易踩的误解：

- **`MUJOCO_GL=egl` 对 mujoco-py 完全无效**。这个变量是给 dm_control / 新版 mujoco 绑定用的。
  mujoco-py 的 GL 后端是**编译期**决定的（`gl/eglshim.c` vs `gl/osmesashim.c`），运行时改环境变量没有任何作用。
  `train_mw.py:8`、`metaworld_env.py:25`、`train_dmc.py:8`、`train_adroit.py:8` 里那几行属于无效代码（无害，暂未清理）。
- **根因在 `mujoco_py/builder.py::get_nvidia_lib_dir()`**。它判断「本机有没有 N 卡」的方式是看
  `/usr/local/nvidia/lib64` 或 `/usr/lib/nvidia-XXX` 这类**十年前的驱动目录布局**是否存在。
  5090 的现代驱动没有这些目录，函数恒返回 `None`，`load_cython_ext` 就永远退回 `LinuxCPUExtensionBuilder`。

## 前置事实（本机）

```
env          : drm-cu128 (python 3.9)
mujoco_py    : fork 版, version_info=(2,0,2,13)，自带 mujoco210 binaries
MJ_BIN       : $CONDA_PREFIX/lib/python3.9/site-packages/mujoco_py/binaries/linux/mujoco210/bin
```

`mujoco_py/__init__.py` 在 import 时会把 `MUJOCO_PY_MUJOCO_PATH` 和 `LD_LIBRARY_PATH`
**强制覆写**成上面那个 bundled 路径，所以 shell 里的 `LD_LIBRARY_PATH` / `~/.mujoco` 都不参与，
改造完全落在 conda env 内部，不影响机器上其他环境。

## 改造步骤

### 0. 前置检查

```bash
conda activate drm-cu128
python -c "from mujoco_py.utils import discover_mujoco; print(discover_mujoco())"   # 确认 MJ_BIN 前缀
which patchelf                                                                      # 缺则 conda install -c conda-forge patchelf
ls /usr/include/GL/glew.h /usr/include/EGL/egl.h                                    # 编译头
ls /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libOpenGL.so.0   # 运行库
ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json                                     # NVIDIA EGL ICD
```

### 1. 备份 builder.py

```bash
MJ=$CONDA_PREFIX/lib/python3.9/site-packages/mujoco_py
cp -n $MJ/builder.py $MJ/builder.py.bak
```

CPU 的 `.so` **不用备份**：GPU 产物文件名不同（`..._linuxgpuextensionbuilder_39.so`），两者天然共存。

### 2. 把 EGL / OpenGL 库软链进 MJ_BIN

```bash
ln -sfn /usr/lib/x86_64-linux-gnu/libEGL.so.1    $MJ/binaries/linux/mujoco210/bin/libEGL.so.1
ln -sfn /usr/lib/x86_64-linux-gnu/libOpenGL.so.0 $MJ/binaries/linux/mujoco210/bin/libOpenGL.so.0
```

**为什么必须放进同一个目录**：`LinuxGPUExtensionBuilder._build_impl` 用 patchelf 把依赖的**绝对路径**
硬写进 `.so`，而它取路径的写法是 `f'{os.environ["LD_LIBRARY_PATH"]}/libEGL.so.1'`
—— 把整个 `LD_LIBRARY_PATH` 当成**单个目录**用。所以 `libmujoco210.so`、`libglewegl.so`、
`libEGL.so.1`、`libOpenGL.so.0` 这 4 个必须挤在同一目录里，路径才拼得对。
（前两个 mujoco210 自带，后两个是这一步补的。）

### 3. 打补丁：builder.py 的 get_nvidia_lib_dir()

把整个函数体替换成：

```python
def get_nvidia_lib_dir():
    # LOCAL PATCH (EGL GPU render): 上游实现只认 /usr/local/nvidia/lib64 与 /usr/lib/nvidia-XXX 这类
    # 老驱动目录布局，在现代驱动上恒为 None，导致永远退回 OSMesa 软件光栅。返回值必须与
    # <mujoco_path>/bin 完全一致，否则 LinuxGPUExtensionBuilder 里 patchelf 拼路径会拿到多段
    # LD_LIBRARY_PATH。被 pip reinstall 冲掉后需重新打这个补丁（备份见 builder.py.bak）。
    return join(discover_mujoco(), 'bin')
```

返回值之所以必须**恰好等于** `<mujoco_path>/bin`：`load_cython_ext` 会对 `<mujoco_path>/bin` 和
`get_nvidia_lib_dir()` 各做一次 `_ensure_set_env_var`，两者不在 `LD_LIBRARY_PATH` 里就直接抛异常；
只有两者是同一个字符串，单目录的 `LD_LIBRARY_PATH` 才能同时满足校验和上面 patchelf 的约束。

### 4. 触发重建

```bash
python -c "import mujoco_py; print(mujoco_py.cymj.__file__)"
```

GPU 版 `.so` 还不存在，`load_cython_ext` 会自动走 GPU 分支编译（约 20s）。
**不需要单独的构建脚本，也不需要 `MUJOCO_PY_FORCE_REBUILD`。**
输出应为 `.../generated/cymj_2.0.2.13_39_linuxgpuextensionbuilder_39.so`。

## 验证

```bash
# 1. 后端确认
python -c "import mujoco_py; print(mujoco_py.cymj.__file__)"   # 含 linuxgpuextensionbuilder

# 2. 链接确认：路径里绝对不能出现 ':'，否则说明踩了 patchelf 拼路径的坑
ldd $MJ/generated/cymj_*_linuxgpuextensionbuilder_39.so | grep -Ei "mujoco|glew|EGL|OpenGL"

# 3. 渲染基准 + 像素回归（脚本在 tmp_claude/bench_render.py）
PYTHONPATH=. python tmp_claude/bench_render.py --tag egl
PYTHONPATH=. python tmp_claude/bench_render.py --compare cpu egl

# 4. 并行 env 回归与吞吐
PYTHONPATH=. python tmp_claude/tmp_test_parallel_eval.py
PYTHONPATH=. python tmp_claude/tmp_bench_parallel_env.py
```

## 实测结果

单 env，coffee-push，84x84，camera corner2，500 step：

| | physics | render | total | step/s |
|---|---|---|---|---|
| CPU / OSMesa | 1.249 ms | **31.813 ms** | 33.185 ms | 30.1 |
| GPU / EGL | 0.336 ms | **0.177 ms** | 0.540 ms | 1852.4 |

渲染快了 **180 倍**，单 env 整体快 **61 倍**。

并行 env 吞吐（`tmp_bench_parallel_env.py`，原来 ~275 step/s 就饱和且显存增量为 0）：

| n | step/s | GPU MB |
|---|---|---|
| 1 | 1013 | 209 |
| 8 | 2823 | 1669 |
| 10 | 2860 | 2086 |
| 25 | 3533 | 5216 |

每个 env 约 209 MB 显存（EGL context），显存增量不再为 0，确认走的是 GPU。
并行扩展性现在受限于 pipe/pickle IPC 和 CPU，不再是渲染。

像素回归（同 seed 同 action 序列，10 帧）：`mean|diff| ≈ 1.9/255`，仅 1.6~1.9% 像素差 > 8，
肉眼比对相机位姿、物体位置、机械臂姿态完全一致 —— 属于软/硬光栅在边缘抗锯齿上的正常差异。

端到端训练（`train_mw.py task=coffee-push agent=drm_mw`）：**FPS ≈ 285**
（改造前 env 单步 33ms，理论上限只有 ~50 fps）。2.1M frames 的完整实验从约 12h 降到约 2h。

## 回滚

```bash
cp $MJ/builder.py.bak $MJ/builder.py
```

`get_nvidia_lib_dir()` 恢复返回 `None` → 下次 import 自动加载回 CPU 的 `.so`。
GPU `.so` 和两个软链留在原地无害。

## 注意事项

- 这个补丁改的是 site-packages，`pip install --force-reinstall mujoco-py` 会冲掉，需重打（函数里有 `LOCAL PATCH` 注释标记）。
- 改动**只影响 drm-cu128 这一个 conda 环境**：不用 sudo，不动驱动、`/usr/lib`、glvnd 配置，其他环境的 mujoco-py 不受影响。
- 多进程（`parallel_env.py` 用 forkserver/spawn）下每个 worker 会各自建一个 EGL context，按 ~209 MB/env 估显存。
- 渲染现在只占训练主循环约 8%，**下一个瓶颈是 `agent.update`**，不再是 env。
