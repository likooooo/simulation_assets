## Simulation Assets

仿真案例、Oghma 对齐数据与 Jupyter notebook（`ipynb/simulation/TMM/`）。

### 前置

1. simulation_core 根目录编译得 `build/simulation.so`（见 [README.md](../README.md)）。
2. 设置环境变量（或 `source scripts/init-simulation-build-env.sh`）：

```bash
export SIMULATION_TMM_ASSETS_DIR="$(pwd)/assets/ipynb/simulation/TMM"
export SIMULATION_DATABASE_DIR="$(pwd)/assets/database"
```

材料数据在 [`database/`](database/)。Oghma notebook 另需本地 `oghma_projects/`、`oghma_spectra/`（路径自定）。

### Jupyter

在 **simulation_core 根目录**：

```bash
source 3rdparty/infrastructure/.venv/bin/activate
source 3rdparty/infrastructure/scripts/init-inf-build-env.sh
pip install -r assets/ipynb/simulation/TMM/requirements.txt
source scripts/init-simulation-build-env.sh
jupyter notebook assets/ipynb/simulation/TMM/
```

`import simulation` 前须 source init 脚本；插件路径由 C++ 自动 append，勿手动改 `PYTHONPATH` 子目录。
