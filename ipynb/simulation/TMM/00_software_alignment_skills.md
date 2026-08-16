> **只读参考** — 外部 Oghma/VirtualLab 路径不在本 submodule 内。完整字段映射见 [`01_interface_mapping.md`](01_interface_mapping.md)。

#### Oghma ↔ TMM 对齐

**ABC/PML 边界**：有限膜两侧各加厚度为 0 的半无限层（bookend），`nk` 取 Oghma 外侧介质，即等效 ABC。
**nk对齐经验**：`_read_oghma_material()` → `simulation_database.read()` + `nk_at_wavelength_um()`

#### Oghma 文件解析
- `photons_escape_prob_yl2.csv` — escape 单点对齐基准（重写 `oled_escape` 模块时使用；旧验证脚本已移除）
```txt
第二列 : 深度方向位置坐标, 是一个相对值, 偏移量是第一个 active layer 到顶部的距离
第三列：escape probability (η)
```

#### Oghma 入参对齐（sim.json）

下面表格是“**必须对齐**”的 Oghma `sim.json` 入参清单，以及 `assets/ipynb/simulation/TMM` 里当前实现的读取/解释方式。

> 关键发现：`ray_theta_`* 在 OghmaNano/gpvdm 里是 **ray tracing 射线角度网格**；`transfer_matrix` outcoupling 手册写明法向入射，simulation 固定 `u=0`（`oghma_emission_u_values_from_project`）。


| Oghma JSON path                                               | OghmaNano 定义/语义                       | simulation 读取/语义            | 对齐结论     | 对齐动作                              |
| ------------------------------------------------------------- | ------------------------------------- | --------------------------- | -------- | --------------------------------- |
| `optical.outcoupling.outcoupling_model`                       | `off/transfer_matrix/ray_trace/unity` | `load_outcoupling_config()` | **一致**   | 保持同名；`transfer_matrix` → `u=[0]`  |
| `optical.outcoupling.incoherent_wavelengths`                  | `struct outcoupling`                  | `load_outcoupling_config()` 记录，**未传 TMM** | **未传 TMM** | 保持                                |
| `optical.outcoupling.Dphotoneff`                              | 光子效率缩放                                | 未读（可选）                      | 字段存在     | P1 补读                             |
| `optical.light_sources.lights.segment0.light_illuminate_from` | `y0/y1/xyz`                           | `load_outcoupling_config()` | **一致**   | OLED 固定 `y0`                      |
| `optical.boundary.optical_y0/y1`                              | 边界标记字符串                               | 报告用                         | 仅校验      | 不映射为角参数                           |
| `optical.mesh.mesh_l.segment0.`*                              | 波长网格                                  | 阈值+`*_u` 兼容                 | **一致**   | 保留历史单位兼容                          |
| `optical.mesh.mesh_y.segment0.`*                              | y 深度网格                                | baseline CSV 优先             | **基本一致** | 对比用 baseline y 轴                  |
| `epitaxy.segment*.shape_pl.pl_emission_enabled`               | EML 开关                                | `load_oghma_project()`      | **一致**   | 保持                                |
| `epitaxy.segment*.shape_pl.ray_theta_`*                       | **ray tracing only**                  | 仅 `ray_trace` 模式读取          | **已修正**  | 不得用于 `transfer_matrix` emission u |
| `optical.light_sources.lights.segment0.ray_theta_`*           | 外部 ray 光源                             | passive/xyz 专用              | 分模式      | `xyz` 光源才用                        |


#### Oghma vs simulation 结果对齐
- [`test_oghma_oled_alignment.py`](test_oghma_oled_alignment.py) — 01_hello_oled R/T、outcoupling
- [`test_oghma_oled_jv.py`](test_oghma_oled_jv.py) — 02_oled_jv JV/EQE（`--profile default|gummel|newton`）
- [`test_fdtd_equivalent_source.py`](test_fdtd_equivalent_source.py) — Bragg/Fabry Case B/C vs detector lam_E
- [`test_oghma_bragg_fabry_alignment.py`](test_oghma_bragg_fabry_alignment.py) — FDTD steady field / lam_e_norm 等

[`test_oghma_oled_alignment.py`](test_oghma_oled_alignment.py) 门控：

- **OLED R/T**（`01_hello_oled` ITO/Al 栈）：使用 `oghma_oled_utils.OLED_RT_GATES`（滤光片用 `oghma_pytest_helpers.RT_R_*` 默认门限）。短波 UV（≈303 nm）材料 tabulated nk 外推导致 max|ΔR|≈0.088，故 R 门限宽于滤光片用 `RT_R_*`（0.005/0.012）。
- 项目路径：`assets/database/og/oghma_projects/oled/01_hello_oled`；基准：`optical_output/reflect.csv`、`transmit.csv`。

#### FDTD 光源 ↔ 普通 TMM

**正向构造等效源谱**

1. 从 `sim.json` 读取 `fdtd_src_waveform`（如 `gaus_sin`）与 `light_spectra`（如 `AM1.5G`）。
2. 合成时域 `E(t)`（Oghma 手册：[FDTD 光源](https://www.oghma-nano.com/zh/manual/finite-difference-light-sources.html)），FFT 得功率谱 `S_E(f)=|E(f)|²`。
3. 加载光谱包络 `Intensity(f)`（`simulation_database` → `og/spectra/AM1.5G.yml`，经 `_read_oghma_spectrum`）。
4. **新源谱**（时间项已消除）：`I_new(f) = S_E(f) × Intensity(f)`。
5. TMM：各 λ 用**单位振幅**单色波，得结构传递 `T(λ)`（及 `r(λ)`/`t(λ)`，经 `TMM_get_r_t_power_s` / `_p` / `TMM_get_r_t_from_tmm`）。
6. **非相干输出**：`S_out(λ) = I_new(λ) · T(λ)`。
7. **归一化传递比**（与 `lam_E_norm.csv` 对齐）：`S_out/S_in = T/G_in`，`G_in = |1+r|²`（源谱在比值中抵消）。
8. **总能量**：`W = ∫ I_new(λ) T(λ) dλ`；绝对尺度取决于 FFT 归一化，形状对比用 peak-normalize。

#### VirtualLab ↔ TMM（椭偏 Δ）

- TMM 内部：`Δ_internal = arg(−ρ)`，`ρ = r_p / r_s`（`filmstack_visualizer.compute_psi_delta`）
- VirtualLab 绘图：`Δ_VL = 180° − Δ_internal`（弧度下等价 `π − Δ_TMM`，`Δ_TMM = Δ_internal`）