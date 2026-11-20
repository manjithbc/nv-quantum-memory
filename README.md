# Phonon-Aware Electron-to-Nuclear Spin Quantum Memory in NV Centers



Simulation code for the paper:

> **Phonon-Aware Electron-to-Nuclear Spin Quantum Memory in NV Centers**
> Manjith B C, Indian Institute of Information Technology Kottayam
## Research paper

[Read the paper](Phonon_Aware_NV_Quantum_Memory.pdf)

The code builds an open-quantum-system model of a write → store → read quantum memory in a single
nitrogen-vacancy (NV⁻) center. The NV electron spin (spin‑1) is the interface qubit, a nearby ¹³C
nuclear spin (spin‑½) is the memory qubit, and a truncated single phonon mode stands in for the lattice.
The dynamics are solved with the Lindblad master equation in [QuTiP](https://qutip.org). The model includes:

- Hyperfine coupling (A∥ SzIz + A⊥(SxIx + SyIy)) and an effective SWAP-type write/read drive
- Spin–phonon coupling g_ph Sz(a + a†) with a thermally occupied phonon mode
- Electron and nuclear relaxation and dephasing, plus phonon damping (Lindblad collapse operators)
- A phenomenological temperature-dependent rate model for 1–300 K sweeps

## Repository layout

```
├── nv_memory.py                     # the model: Hamiltonians, collapse operators, protocol, plots
├── run_all.py                       # regenerates every figure + Table 1 from the paper
├── notebooks/
│   └── nv_quantum_memory.ipynb      # original notebook, with outputs as reported in the paper
├── figures/                         # created by run_all.py
├── results/                         # created by run_all.py (table1_results.csv, summary.json)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

You need Python 3.10 or newer. The paper's results were produced with Python 3.11 and QuTiP 5.

```bash
git clone https://github.com/manjithbc/nv-quantum-memory.git
cd nv-quantum-memory
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the paper

```bash
python run_all.py           # all figures + Table 1 (the heatmap takes the longest)
python run_all.py --quick   # everything except the heatmap
```

You can also open `notebooks/nv_quantum_memory.ipynb` in Jupyter and run it top to bottom.

| Paper item | Function in `nv_memory.py` | Output |
|---|---|---|
| Fig. 3 – model schematic | `plot_nv_memory_concept()` | `figures/fig3_model_concept.png` |
| Fig. 4 – electron-spin energy levels | `plot_energy_levels()` | `figures/fig4_energy_levels.png` |
| Fig. 5 – write–store–read sequence | `plot_memory_pulse_sequence()` | `figures/fig5_pulse_sequence.png` |
| Sec. 5.1 – single run | `run_memory_protocol(10.0, 0.02)` | console, `results/summary.json` |
| Fig. 6 – fidelity vs storage time | `plot_fidelity_vs_storage_time()` | `figures/fig6_fidelity_vs_storage_time.png` |
| Fig. 7 – fidelity vs temperature | `plot_fidelity_vs_temperature()` | `figures/fig7_fidelity_vs_temperature.png` |
| Fig. 8 – fidelity heatmap F(g_ph, Tₛ) | `plot_fidelity_heatmap()` | `figures/fig8_fidelity_heatmap.png` |
| Table 1 | `generate_results_table()` | `results/table1_results.csv` |

Expected values for the single run (Sec. 5.1): write fidelity ≈ 0.9994, final fidelity ≈ 0.9982,
transfer efficiency ≈ 0.9994, spin purity ≈ 0.9964, n̄ ≈ 0.6206.

Runtime: Figs. 6 and 8 are sampled every 0.1 µs (about 3,300 protocol runs in total), so a full run takes
anywhere from several minutes to under an hour, depending on your machine. `--quick` skips the heatmap.

### Simulation frame

`USE_SECULAR_RESONANT_FRAME = True` (default; used in the paper) simulates in the secular, hyperfine-resonant
rotating frame. The transverse hyperfine term is dropped because it is suppressed by the 2.87 GHz zero-field
splitting. The secular term A∥SzIz is removed exactly by its interaction picture, and the SWAP drive is
resonant with the hyperfine-shifted transition. Setting it to `False` restores the first notebook version,
which keeps both terms in the rotating frame (final fidelity ≈ 0.30 in the single run).

## Using the model

```python
import nv_memory as nv

out = nv.run_memory_protocol(
    T_storage_us=10.0,      # storage time (µs)
    g_ph_MHz=0.02,          # spin–phonon coupling
    T_ph_K=5e-5,            # phonon-mode temperature
    Gamma1e_per_us=5e-4,    # electron relaxation rate
    Gamma_phie_per_us=5e-4, # electron dephasing rate
)
print(out["F_write"], out["F_final"], out["purity_final"])
```

The physical constants (D = 2870 MHz, B = 10 mT, A∥, A⊥, J_swap, phonon frequency, and so on) are
module-level variables in Section 5 of `nv_memory.py`. Units: time in µs, frequency in MHz.

## Scope and limitations

This is a **conceptual** model, as discussed in the paper. The parameters are illustrative and have not
been fitted to a specific experiment. The phonon bath is a single truncated mode. The temperature sweep
uses phenomenological rates. Final fidelities depend strongly on coherent phase accumulation during
storage, so absolute values should be read as model-level indicators.


## License

Code is released under the [MIT License](LICENSE).

## Contact

Manjith B C — manjithbc@iiitkottayam.ac.in
