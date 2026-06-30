## Simulation Assets

仿真案例、Oghma 对齐数据与 Jupyter notebook（`ipynb/simulation/TMM/`）。

### 前置

1. simulation_core 根目录编译得 `build/simulation.so`（见 [README.md](../README.md)）。
2. 设置运行时环境变量（或 `source scripts/init-simulation-build-env.sh`）：

```bash
source scripts/init-simulation-build-env.sh
```

材料数据在 [`database/`](database/)。Oghma notebook 另需本地 `oghma_projects/`、`oghma_spectra/`（路径自定）。

TMM 目录脚本 [`ipynb/simulation/TMM/run_tmm.sh`](ipynb/simulation/TMM/run_tmm.sh) 负责校验运行时 env 并 `cd` 到 TMM 目录。

### 测试

先手动 source 运行时 env，再跑 TMM 脚本：

```bash
source scripts/init-simulation-build-env.sh build
./assets/ipynb/simulation/TMM/run_tmm.sh test
```

toykits 根目录：`source scripts/init-toykits-build-env.sh` 后同样调用 `run_tmm.sh test`。

### Jupyter

在 **simulation_core 根目录**：

```bash
source 3rdparty/infrastructure/.venv/bin/activate
source 3rdparty/infrastructure/scripts/init-inf-build-env.sh
pip install -r assets/ipynb/simulation/TMM/requirements.txt
source scripts/init-simulation-build-env.sh
./assets/ipynb/simulation/TMM/run_tmm.sh jupyter
```

`import simulation` 前须 source init 脚本；插件路径由 C++ 自动 append，勿手动改 `PYTHONPATH` 子目录。
