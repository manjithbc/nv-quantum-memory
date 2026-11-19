"""
Reproduce all figures and the results table from the paper.

    python run_all.py            # full run (several minutes; the heatmap is the slowest)
    python run_all.py --quick    # skip the heatmap

Figures are written to figures/ (PNG + PDF) and numerical outputs to results/.
Figure numbers match the manuscript.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save figures instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402

import nv_memory as nv  # noqa: E402

FIG_DIR = "figures"
RES_DIR = "results"


def save_current(name):
    """Save the active matplotlib figure as PNG and PDF, then close it."""
    fig = plt.gcf()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR}/{name}.png")


def run_and_save(func, name):
    """Call one of the notebook's plotting functions, saving instead of showing."""
    original_show = plt.show
    plt.show = lambda *a, **k: save_current(name)
    try:
        return func()
    finally:
        plt.show = original_show


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slow heatmap (Fig. 8)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    summary = {"t_swap_us": nv.t_swap_us}
    print(f"SWAP time = {nv.t_swap_us} microseconds\n")

    print("Fig. 3-5: model, energy levels, pulse sequence")
    run_and_save(nv.plot_nv_memory_concept, "fig3_model_concept")
    run_and_save(nv.plot_energy_levels, "fig4_energy_levels")
    run_and_save(nv.plot_memory_pulse_sequence, "fig5_pulse_sequence")

    print("\nSection 5.1: single test run (Ts = 10 us, g_ph = 0.02 MHz)")
    test = nv.run_memory_protocol(T_storage_us=10.0, g_ph_MHz=0.02)
    single = {k: float(test[k]) for k in
              ("F_write", "F_final", "transfer_efficiency", "purity_final", "nbar")}
    for k, v in single.items():
        print(f"  {k:<20} = {v:.4f}")
    summary["single_run"] = single

    print("\nFig. 6: fidelity vs storage time")
    storage_times, storage_results = run_and_save(nv.plot_fidelity_vs_storage_time,
                                                  "fig6_fidelity_vs_storage_time")
    T2_eff = nv.estimate_T2_eff(storage_times, storage_results[0.01])
    print(f"  T2_eff (g_ph = 0.01 MHz) = {T2_eff:.4g} us")
    summary["T2_eff_us_gph_0.01"] = float(T2_eff)

    print("\nFig. 7: fidelity vs temperature")
    temps, temp_F, _ = run_and_save(nv.plot_fidelity_vs_temperature,
                                    "fig7_fidelity_vs_temperature")
    summary["temperature_fidelity_range"] = [float(min(temp_F)), float(max(temp_F))]

    if not args.quick:
        print("\nFig. 8: fidelity heatmap F(g_ph, Ts)  (slow)")
        run_and_save(nv.plot_fidelity_heatmap, "fig8_fidelity_heatmap")

    print("\nTable 1: summary of numerical results")
    df = nv.generate_results_table()
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RES_DIR, "table1_results.csv"), index=False)

    with open(os.path.join(RES_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Numerical outputs written to {RES_DIR}/")


if __name__ == "__main__":
    main()
