## Simulation Assets

仿真案例、Oghma 对齐数据与 Jupyter notebook（`ipynb/simulation/TMM/`）。

### 前置

1. simulation_core 根目录编译得 Release `build/simulation.so`（见 [README.md](../README.md)）。
2. 设置运行时环境变量：

```bash
source scripts/init-simulation-build-env.sh build
```

材料数据在 [`database/`](database/)（须有 `og/materials/` YAML）。Oghma 工程默认在 `database/og/oghma_projects/`（可选本机 symlink `assets/oghma_projects` → OneDrive，**非**可移植路径）。

TMM 目录脚本 [`ipynb/simulation/TMM/run_tmm.sh`](ipynb/simulation/TMM/run_tmm.sh) 校验 `SIMULATION_ARTIFACTS_DIR` / `SIMULATION_DATABASE_DIR`，并 `cd` 到 TMM 目录。

**不要**用 toykits 的 `init-toykits-build-env.sh` 或 `.simulation_toolkits` 跑 TMM（脚本会拒绝）。

### 测试

```bash
# 在 simulation_core 根目录
source scripts/init-simulation-build-env.sh build
./assets/ipynb/simulation/TMM/run_tmm.sh test
```

若缺少 Oghma 工程数据：

```bash
python assets/database/update_all.py --only oghma_projects
```

### Jupyter

`run_tmm.sh jupyter` 会安装 TMM `requirements.txt`；init 脚本已 source infrastructure 环境：

```bash
# 在 simulation_core 根目录
source scripts/init-simulation-build-env.sh build
./assets/ipynb/simulation/TMM/run_tmm.sh jupyter
```

在 notebook 中勿要求 cwd=`build/`；由 `oghma_runtime.bootstrap_tmm_session()` / `run_tmm.sh` 处理 env 与路径。`import simulation` 前须已 source init；插件由 C++ eager-load artifact 内副本，**勿**把源码树 `simulation_plugins` / `py_core_plugins` 塞进 `PYTHONPATH`。

其它目录（如 `ipynb/virtuallab/`）的旧 notebook 同样须先 `source scripts/init-simulation-build-env.sh`，勿手动挂源码 plugins。
