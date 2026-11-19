"""
Phonon-aware electron-to-nuclear spin quantum memory in NV centers.

Conceptual QuTiP (open-system, Lindblad) simulation accompanying the paper
"Phonon-Aware Electron-to-Nuclear Spin Quantum Memory in NV Centers"
by Manjith B C.

Hilbert space: electron spin-1 (3) x 13C nuclear spin-1/2 (2) x phonon mode (Nph).
Units: time in microseconds, frequencies in MHz (converted to rad/us via 2*pi).

The code is identical to the accompanying notebook (notebooks/nv_quantum_memory.ipynb),
split out as an importable module. Use run_all.py to regenerate every figure and the
results table from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qutip import (
    basis, qeye, tensor, jmat,
    sigmax, sigmay, sigmaz,
    destroy, thermal_dm, ket2dm,
    mesolve, expect
)

# ============================================================
# 1. Units and helper functions
# ============================================================

# Time unit: microsecond
# Frequency unit for Hamiltonians: MHz
# Since 1 MHz = 1 cycle/microsecond,
# angular frequency = 2*pi*f_MHz rad/microsecond.

twopi = 2 * np.pi

def wMHz(f_MHz):
    """
    Convert frequency in MHz to angular frequency in rad/us.
    This is used in the Hamiltonian because QuTiP uses hbar = 1.
    """
    return twopi * f_MHz


def thermal_nbar(T_K, f_MHz):
    """
    Average thermal phonon number:
    
        nbar(T) = 1 / (exp(h f / k_B T) - 1)
    
    Here f is the phonon frequency in MHz.
    """
    if T_K <= 0:
        return 0.0
    
    h = 6.62607015e-34      # Planck constant, J s
    kB = 1.380649e-23       # Boltzmann constant, J/K
    f_Hz = f_MHz * 1e6
    
    x = h * f_Hz / (kB * T_K)
    
    # Avoid overflow for very low temperature
    if x > 700:
        return 0.0
    
    return 1.0 / (np.exp(x) - 1.0)


# ============================================================
# 2. Hilbert space dimensions
# ============================================================

# Electron spin of NV- center:
# S = 1, so dimension = 3.
# QuTiP spin-1 basis ordering:
# basis(3, 0) -> m_s = +1
# basis(3, 1) -> m_s = 0
# basis(3, 2) -> m_s = -1

Ne = 3

# Nuclear spin:
# For a nearby 13C nuclear spin, I = 1/2, so dimension = 2.
Nn = 2

# Phonon mode:
# We truncate the harmonic oscillator to Nph Fock states.
# Larger Nph gives better accuracy but increases computation time.
Nph = 8

# Basis states for electron spin
e_p1 = basis(Ne, 0)   # |m_s = +1>
e_0  = basis(Ne, 1)   # |m_s = 0>
e_m1 = basis(Ne, 2)   # |m_s = -1>

# Basis states for nuclear spin
n_0 = basis(Nn, 0)    # nuclear |0>
n_1 = basis(Nn, 1)    # nuclear |1>

# Identity operators
Ie = qeye(Ne)
In = qeye(Nn)
Iph = qeye(Nph)


# ============================================================
# 3. Tensor embedding functions
# ============================================================

def E(op_e):
    """
    Embed an electron-spin operator into the full Hilbert space:
    
        electron ⊗ nuclear ⊗ phonon
    """
    return tensor(op_e, In, Iph)


def N(op_n):
    """
    Embed a nuclear-spin operator into the full Hilbert space.
    """
    return tensor(Ie, op_n, Iph)


def P(op_ph):
    """
    Embed a phonon operator into the full Hilbert space.
    """
    return tensor(Ie, In, op_ph)


def EN(op_e, op_n):
    """
    Embed an electron-nuclear product operator.
    """
    return tensor(op_e, op_n, Iph)


def EPH(op_e, op_ph):
    """
    Embed an electron-phonon product operator.
    """
    return tensor(op_e, In, op_ph)


# ============================================================
# 4. Spin and phonon operators
# ============================================================

# Electron spin-1 operators
Sx0 = jmat(1, 'x')
Sy0 = jmat(1, 'y')
Sz0 = jmat(1, 'z')

# Nuclear spin-1/2 operators
Ix0 = 0.5 * sigmax()
Iy0 = 0.5 * sigmay()
Iz0 = 0.5 * sigmaz()

# Phonon annihilation operator
a0 = destroy(Nph)

# Embedded phonon operator
a = P(a0)

# Electron-state projectors
P_ep1_0 = e_p1 * e_p1.dag()
P_e0_0  = e_0  * e_0.dag()
P_em1_0 = e_m1 * e_m1.dag()

# Nuclear-state projectors
P_n0_0 = n_0 * n_0.dag()
P_n1_0 = n_1 * n_1.dag()

# Electron qubit subspace:
# We use |m_s = 0> and |m_s = -1> as the electron qubit.
Z_e0 = P_e0_0 - P_em1_0

# Nuclear qubit Z operator
Z_n0 = P_n0_0 - P_n1_0

# Electron relaxation operator:
# |-1> -> |0>
e_relax0 = e_0 * e_m1.dag()

# Nuclear relaxation operator:
# |1> -> |0>
n_relax0 = n_0 * n_1.dag()

# Operators for SWAP-like electron-to-nuclear transfer
# e_down maps electron |-1> -> |0>
# n_up maps nuclear |0> -> |1>
e_down0 = e_0 * e_m1.dag()
e_up0   = e_m1 * e_0.dag()

n_up0   = n_1 * n_0.dag()
n_down0 = n_0 * n_1.dag()


# ============================================================
# 5. Physical and simulation parameters
# ============================================================

# NV zero-field splitting
D_MHz = 2870.0

# Electron gyromagnetic ratio
gamma_e_MHz_per_T = 28000.0

# 13C nuclear gyromagnetic ratio
gamma_n_MHz_per_T = 10.705

# Static magnetic field
Bz_T = 0.010       # 10 mT

# Hyperfine coupling parameters
# These are example values for a conceptual simulation.
# Strongly coupled 13C spins can have much larger values.
A_para_MHz = 0.20
A_perp_MHz = 0.05

# Effective phonon mode frequency
# This is a simplified single-mode phonon model.
omega_m_MHz = 1.0

# Effective electron-phonon coupling strength
g_ph_default_MHz = 0.01

# Effective SWAP coupling.
# This represents a controlled microwave/RF-assisted electron-nuclear
# state-transfer operation in a rotating frame.
J_swap_MHz = 0.20

# Simulation frame (see build_hamiltonians):
#   True  -> secular approximation + hyperfine-resonant SWAP drive (used in the paper)
#   False -> original treatment that keeps A_perp and the A_par/2 drive detuning
#            in the rotating frame (reproduces the first notebook version)
USE_SECULAR_RESONANT_FRAME = True

# SWAP time for Hamiltonian:
# H_swap = 2*pi*J ( |e0,n1><em1,n0| + h.c. )
# Complete transfer occurs at t = 1 / (4J)
t_swap_us = 1.0 / (4.0 * J_swap_MHz)



# ============================================================
# 6. Hamiltonian construction
# ============================================================

def build_hamiltonians(g_ph_MHz=g_ph_default_MHz):
    """
    Build different Hamiltonian components.
    
    Full laboratory-frame Hamiltonian:
    
        H = H_e + H_n + H_hf + H_ph + H_e-ph + H_drive
    
    For the actual memory protocol, we use a rotating-frame effective model:
    
        H_memory = H_ph + H_e-ph + H_swap   (secular, hyperfine-resonant frame;
                                             H_hf is added back if
                                             USE_SECULAR_RESONANT_FRAME = False)
    
    This avoids numerically unnecessary fast oscillations from the 2.87 GHz
    zero-field splitting.
    """
    
    # ----------------------------
    # Electron Hamiltonian
    # H_e = D Sz^2 + gamma_e Bz Sz
    # ----------------------------
    H_e = (
        wMHz(D_MHz) * E(Sz0 ** 2)
        + wMHz(gamma_e_MHz_per_T * Bz_T) * E(Sz0)
    )
    
    # ----------------------------
    # Nuclear Hamiltonian
    # H_n = - gamma_n Bz Iz
    # ----------------------------
    H_n = -wMHz(gamma_n_MHz_per_T * Bz_T) * N(Iz0)
    
    # ----------------------------
    # Hyperfine Hamiltonian
    # H_hf = A_parallel Sz Iz + A_perp(Sx Ix + Sy Iy)
    # ----------------------------
    H_hf = (
        wMHz(A_para_MHz) * EN(Sz0, Iz0)
        + wMHz(A_perp_MHz) * (EN(Sx0, Ix0) + EN(Sy0, Iy0))
    )
    
    # ----------------------------
    # Phonon Hamiltonian
    # H_ph = omega_m a^dagger a
    # ----------------------------
    H_ph = wMHz(omega_m_MHz) * P(a0.dag() * a0)
    
    # ----------------------------
    # Electron-phonon Hamiltonian
    # H_e-ph = g_ph Sz (a + a^dagger)
    # ----------------------------
    H_eph = wMHz(g_ph_MHz) * EPH(Sz0, a0 + a0.dag())
    
    # ----------------------------
    # Effective write/read drive Hamiltonian
    # This implements a SWAP-like transfer:
    #
    # |m_s = -1>_e |0>_n  <-->  |m_s = 0>_e |1>_n
    # ----------------------------
    H_swap = wMHz(J_swap_MHz) * (
        tensor(e_down0, n_up0, Iph)
        + tensor(e_up0, n_down0, Iph)
    )
    
    # Full Hamiltonian, mostly for conceptual completeness
    H_full_static = H_e + H_n + H_hf + H_ph + H_eph

    # Rotating-frame base Hamiltonian used during storage
    if USE_SECULAR_RESONANT_FRAME:
        # Secular approximation + resonant (hyperfine-calibrated) drive.
        # (i) The transverse hyperfine term A_perp (Sx Ix + Sy Iy) connects levels
        #     split by the zero-field splitting D ~ 2.87 GHz >> A_perp, so in the
        #     electron rotating frame it is non-secular and is dropped.
        # (ii) The secular term A_par Sz Iz commutes with H_ph, H_e-ph and with all
        #     dephasing operators, and only rotates the phases of the relaxation
        #     operators (which leaves the Lindblad dissipator unchanged). Moving to
        #     its interaction picture therefore removes it exactly, and the MW/RF
        #     SWAP drive, calibrated to the hyperfine-shifted transition, becomes
        #     resonant.
        H_base = H_ph + H_eph
    else:
        # Original (v0) treatment: both hyperfine terms kept in the rotating frame
        # without D and with an uncompensated drive detuning A_par/2.
        H_base = H_hf + H_ph + H_eph
    
    # Hamiltonian used during write and read operations
    H_write_read = H_base + H_swap
    
    return H_full_static, H_base, H_write_read


# ============================================================
# 7. Lindblad collapse operators
# ============================================================

def make_collapse_ops(
    Gamma1e_per_us=0.0005,
    Gamma_phie_per_us=0.0005,
    Gamma1n_per_us=0.00001,
    Gamma_phin_per_us=0.00001,
    kappa_per_us=0.001,
    T_ph_K=5e-5,
    include_phonon_bath=True
):
    """
    Build Lindblad collapse operators.
    
    Collapse operators:
    
        sqrt(Gamma1e)     * electron relaxation
        sqrt(Gamma_phie/2)* electron dephasing
        sqrt(Gamma1n)     * nuclear relaxation
        sqrt(Gamma_phin/2)* nuclear dephasing
        sqrt(kappa(nbar+1))* a
        sqrt(kappa nbar)   * a^dagger
    
    Rates are given in 1/microsecond.
    """
    
    c_ops = []
    
    # Electron relaxation: |-1> -> |0>
    if Gamma1e_per_us > 0:
        c_ops.append(np.sqrt(Gamma1e_per_us) * E(e_relax0))
    
    # Electron pure dephasing
    if Gamma_phie_per_us > 0:
        c_ops.append(np.sqrt(Gamma_phie_per_us / 2.0) * E(Z_e0))
    
    # Nuclear relaxation: |1> -> |0>
    if Gamma1n_per_us > 0:
        c_ops.append(np.sqrt(Gamma1n_per_us) * N(n_relax0))
    
    # Nuclear pure dephasing
    if Gamma_phin_per_us > 0:
        c_ops.append(np.sqrt(Gamma_phin_per_us / 2.0) * N(Z_n0))
    
    # Thermal phonon occupation
    nbar = thermal_nbar(T_ph_K, omega_m_MHz)
    
    # Phonon damping and thermal excitation
    if include_phonon_bath and kappa_per_us > 0:
        c_ops.append(np.sqrt(kappa_per_us * (nbar + 1.0)) * a)
        
        if nbar > 0:
            c_ops.append(np.sqrt(kappa_per_us * nbar) * a.dag())
    
    return c_ops, nbar


# ============================================================
# 8. Initial state preparation
# ============================================================

def make_initial_state(
    alpha=1 / np.sqrt(2),
    beta=1 / np.sqrt(2),
    T_ph_K=5e-5
):
    """
    Prepare the initial state:
    
        |Psi(0)> = (alpha |0>_e + beta |-1>_e)
                  ⊗ |0>_n
                  ⊗ rho_ph
    
    The phonon mode is initialized in a thermal state.
    """
    
    psi_e = (alpha * e_0 + beta * e_m1).unit()
    psi_n = n_0
    
    nbar = thermal_nbar(T_ph_K, omega_m_MHz)
    rho_ph = thermal_dm(Nph, nbar)
    
    rho0 = tensor(ket2dm(psi_e), ket2dm(psi_n), rho_ph)
    
    return rho0, psi_e, psi_n, nbar


# ============================================================
# 9. Fidelity and metric functions
# ============================================================

def reduced_spin_state(rho):
    """
    Trace out the phonon subsystem.
    Returns electron+nuclear density matrix.
    """
    return rho.ptrace([0, 1])


def fidelity_with_pure_spin_target(rho, psi_target_spin):
    """
    Fidelity with a pure target state in the electron+nuclear spin space.
    
        F = <psi_target| rho_spin |psi_target>
    """
    rho_spin = reduced_spin_state(rho)
    P_target = ket2dm(psi_target_spin)
    F = expect(P_target, rho_spin)
    return float(np.real(F))


def spin_purity(rho):
    """
    Purity of the electron+nuclear reduced density matrix:
    
        purity = Tr(rho_spin^2)
    """
    rho_spin = reduced_spin_state(rho)
    purity = (rho_spin * rho_spin).tr()
    return float(np.real(purity))


def target_after_write(alpha, beta):
    """
    Ideal target state after write operation.
    
    Because the exchange Hamiltonian introduces a known -i phase:
    
        beta |e=-1,n=0> -> -i beta |e=0,n=1>
    
    the target after write is:
    
        |e=0> ⊗ (alpha |n=0> - i beta |n=1>)
    """
    psi_n_target = (alpha * n_0 - 1j * beta * n_1).unit()
    return tensor(e_0, psi_n_target)


def target_after_read(alpha, beta):
    """
    Ideal target after write-store-read, after phase correction:
    
        (alpha |0>_e + beta |-1>_e) ⊗ |0>_n
    """
    psi_e_target = (alpha * e_0 + beta * e_m1).unit()
    return tensor(psi_e_target, n_0)


# Electron phase correction after read.
# The double SWAP gives a known minus sign to the |-1> component.
# This unitary corrects that phase.
U_Ze0 = P_ep1_0 + P_e0_0 - P_em1_0
U_Ze = E(U_Ze0)

def apply_read_phase_correction(rho):
    """
    Apply a simple electron Z phase correction after the read operation.
    """
    return U_Ze * rho * U_Ze.dag()


# Population operator for successful write transfer:
# |e=0, n=1>
P_e0n1 = tensor(P_e0_0, P_n1_0, Iph)


# ============================================================
# 10. Evolution function
# ============================================================

def evolve_segment(rho0, H, duration_us, c_ops, steps=100):
    """
    Evolve the density matrix for one segment of the protocol.
    
    Uses QuTiP mesolve:
    
        d rho / dt = -i[H, rho] + Lindblad terms
    """
    
    if duration_us <= 0:
        return rho0, None
    
    tlist = np.linspace(0, duration_us, steps)
    
    result = mesolve(
        H,
        rho0,
        tlist,
        c_ops=c_ops,
        e_ops=[],
        options={
            "store_states": True,
            "nsteps": 20000,
            "progress_bar": ""
        }
    )
    
    return result.states[-1], result


# ============================================================
# 11. Complete write-store-read memory protocol
# ============================================================

def run_memory_protocol(
    T_storage_us=5.0,
    g_ph_MHz=0.01,
    T_ph_K=5e-5,
    Gamma1e_per_us=0.0005,
    Gamma_phie_per_us=0.0005,
    Gamma1n_per_us=0.00001,
    Gamma_phin_per_us=0.00001,
    kappa_per_us=0.001,
    include_phonon_bath=True,
    alpha=1 / np.sqrt(2),
    beta=1 / np.sqrt(2)
):
    """
    Run the full memory protocol:
    
        1. Initialize electron superposition and nuclear |0>
        2. Write electron state to nuclear spin
        3. Store for T_storage_us
        4. Read nuclear state back to electron spin
        5. Apply known phase correction
        6. Calculate fidelity and purity
    """
    
    # Build Hamiltonians
    H_full_static, H_base, H_write_read = build_hamiltonians(g_ph_MHz=g_ph_MHz)
    
    # Build collapse operators
    c_ops, nbar = make_collapse_ops(
        Gamma1e_per_us=Gamma1e_per_us,
        Gamma_phie_per_us=Gamma_phie_per_us,
        Gamma1n_per_us=Gamma1n_per_us,
        Gamma_phin_per_us=Gamma_phin_per_us,
        kappa_per_us=kappa_per_us,
        T_ph_K=T_ph_K,
        include_phonon_bath=include_phonon_bath
    )
    
    # Initial state
    rho0, psi_e_initial, psi_n_initial, nbar_init = make_initial_state(
        alpha=alpha,
        beta=beta,
        T_ph_K=T_ph_K
    )
    
    # ----------------------------
    # Write operation
    # ----------------------------
    rho_write, result_write = evolve_segment(
        rho0,
        H_write_read,
        t_swap_us,
        c_ops,
        steps=100
    )
    
    F_write = fidelity_with_pure_spin_target(
        rho_write,
        target_after_write(alpha, beta)
    )
    
    P_transfer = float(np.real(expect(P_e0n1, rho_write)))
    
    # Normalize transfer probability by ideal beta population
    if abs(beta) > 1e-12:
        transfer_efficiency = P_transfer / (abs(beta) ** 2)
    else:
        transfer_efficiency = np.nan
    
    # ----------------------------
    # Storage operation
    # ----------------------------
    storage_steps = max(20, int(20 * T_storage_us))
    
    rho_store, result_store = evolve_segment(
        rho_write,
        H_base,
        T_storage_us,
        c_ops,
        steps=storage_steps
    )
    
    # ----------------------------
    # Read operation
    # ----------------------------
    rho_read, result_read = evolve_segment(
        rho_store,
        H_write_read,
        t_swap_us,
        c_ops,
        steps=100
    )
    
    # Correct known phase from two SWAP-like operations
    rho_read_corrected = apply_read_phase_correction(rho_read)
    
    F_final = fidelity_with_pure_spin_target(
        rho_read_corrected,
        target_after_read(alpha, beta)
    )
    
    purity_final = spin_purity(rho_read_corrected)
    
    return {
        "rho_final": rho_read_corrected,
        "F_write": F_write,
        "F_final": F_final,
        "purity_final": purity_final,
        "P_transfer": P_transfer,
        "transfer_efficiency": transfer_efficiency,
        "nbar": nbar,
        "T_storage_us": T_storage_us,
        "g_ph_MHz": g_ph_MHz,
        "T_ph_K": T_ph_K
    }


# ============================================================
# 12. Figure 1: Conceptual NV memory diagram
# ============================================================

def plot_nv_memory_concept():
    """
    Simple conceptual diagram:
    electron spin coupled to nuclear spin and phonon bath.
    """
    
    plt.figure(figsize=(8, 4))
    
    # Positions
    electron_pos = (0.25, 0.5)
    nuclear_pos = (0.65, 0.5)
    phonon_pos = (0.45, 0.15)
    
    # Draw nodes
    plt.scatter(*electron_pos, s=1800)
    plt.scatter(*nuclear_pos, s=1800)
    plt.scatter(*phonon_pos, s=1200)
    
    # Labels
    plt.text(*electron_pos, "Electron\nspin", ha="center", va="center")
    plt.text(*nuclear_pos, "Nuclear\nspin", ha="center", va="center")
    plt.text(*phonon_pos, "Phonon\nbath", ha="center", va="center")
    
    # Couplings
    plt.annotate(
        "",
        xy=nuclear_pos,
        xytext=electron_pos,
        arrowprops=dict(arrowstyle="<->", lw=2)
    )
    plt.text(0.45, 0.58, "Hyperfine\ncoupling", ha="center")
    
    plt.annotate(
        "",
        xy=phonon_pos,
        xytext=electron_pos,
        arrowprops=dict(arrowstyle="<->", lw=2)
    )
    plt.text(0.28, 0.28, "Spin-phonon\ncoupling", ha="center")
    
    plt.xlim(0, 0.9)
    plt.ylim(0, 0.8)
    plt.axis("off")
    plt.title("Phonon-aware electron-to-nuclear spin quantum memory")
    plt.show()


# ============================================================
# 13. Figure 2: NV electron-spin energy levels versus magnetic field
# ============================================================

def plot_energy_levels():
    """
    Plot electron spin energy levels:
    
        H_e = D Sz^2 + gamma_e Bz Sz
    
    This shows the zero-field splitting and Zeeman splitting.
    """
    
    B_values_T = np.linspace(0, 0.02, 150)   # 0 to 20 mT
    energies_MHz = []
    
    for B in B_values_T:
        H_electron = (
            wMHz(D_MHz) * (Sz0 ** 2)
            + wMHz(gamma_e_MHz_per_T * B) * Sz0
        )
        
        eigs = H_electron.eigenenergies()
        
        # Convert angular frequency back to MHz
        eigs_MHz = eigs / twopi
        energies_MHz.append(eigs_MHz)
    
    energies_MHz = np.array(energies_MHz)
    
    plt.figure(figsize=(7, 5))
    
    for i in range(3):
        plt.plot(B_values_T * 1e3, energies_MHz[:, i], label=f"Level {i+1}")
    
    plt.xlabel("Magnetic field Bz (mT)")
    plt.ylabel("Energy / h (MHz)")
    plt.title("NV electron-spin energy levels")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 14. Figure 3: Memory pulse sequence
# ============================================================

def plot_memory_pulse_sequence():
    """
    Draw a simple pulse sequence:
    
        Initialize -> Write -> Store -> Read -> Measure
    """
    
    stages = ["Initialize", "Write", "Store", "Read", "Measure"]
    durations = [0.5, t_swap_us, 5.0, t_swap_us, 0.5]
    
    start_times = np.cumsum([0] + durations[:-1])
    
    plt.figure(figsize=(9, 2.5))
    
    for start, duration, label in zip(start_times, durations, stages):
        plt.broken_barh([(start, duration)], (0, 1))
        plt.text(start + duration / 2, 0.5, label, ha="center", va="center")
    
    plt.xlabel("Time (microseconds)")
    plt.yticks([])
    plt.title("Write–store–read memory protocol")
    plt.xlim(0, sum(durations))
    plt.show()


# ============================================================
# 15. Figure 4: Fidelity versus storage time
# ============================================================

def plot_fidelity_vs_storage_time():
    """
    Study how memory fidelity changes with storage time
    for different phonon coupling strengths.
    """
    
    # 0.1 us resolution: resolves the coherent oscillation at the phonon-mode
    # period 1/omega_m (= 1 us), which a coarse grid would alias.
    storage_times = np.linspace(0, 25, 251)
    g_values = [0.0, 0.01, 0.03, 0.05]
    
    all_results = {}
    
    plt.figure(figsize=(9, 5))
    
    for g in g_values:
        fidelities = []
        
        for Ts in storage_times:
            out = run_memory_protocol(
                T_storage_us=Ts,
                g_ph_MHz=g,
                T_ph_K=5e-5,
                Gamma1e_per_us=0.0005,
                Gamma_phie_per_us=0.0005,
                Gamma1n_per_us=0.00001,
                Gamma_phin_per_us=0.00001,
                kappa_per_us=0.001,
                include_phonon_bath=True
            )
            
            fidelities.append(out["F_final"])
        
        all_results[g] = fidelities
        
        plt.plot(
            storage_times,
            fidelities,
            linewidth=1.2,
            label=f"g_ph = {g} MHz"
        )
    
    plt.xlabel("Storage time Ts (microseconds)")
    plt.ylabel("Final memory fidelity")
    plt.title("Memory fidelity versus storage time")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.grid(True)
    plt.show()
    
    return storage_times, all_results


# ============================================================
# 16. Estimate effective T2 from fidelity decay
# ============================================================

def exponential_decay(t, F_inf, A, T2):
    """
    Simple empirical decay model:
    
        F(t) = F_inf + A exp(-t/T2)
    """
    return F_inf + A * np.exp(-t / T2)


def estimate_T2_eff(storage_times, fidelities):
    """
    Estimate effective coherence time from fidelity curve.
    This is not a rigorous experimental T2 extraction;
    it is a useful simulation metric.
    """
    
    storage_times = np.array(storage_times)
    fidelities = np.array(fidelities)
    
    try:
        p0 = [min(fidelities), max(fidelities) - min(fidelities), 10.0]
        params, _ = curve_fit(
            exponential_decay,
            storage_times,
            fidelities,
            p0=p0,
            bounds=([0, 0, 0.001], [1, 1, 1e6])
        )
        
        F_inf, A, T2_eff = params
        return T2_eff
    
    except Exception:
        return np.nan


# ============================================================
# 17. Effective temperature-dependent rate model
# ============================================================

def effective_electron_rates_from_temperature(
    T_K,
    f_ref_MHz=2870.0,
    Gamma1e_base=0.0002,
    Gamma1e_extra_300K=0.004,
    Gamma_phie_base=0.0002,
    Gamma_phie_extra_300K=0.006
):
    """
    Phenomenological temperature-dependent electron relaxation/dephasing model.
    
    This is used for room-temperature sweeps.
    
    Why?
    Explicitly simulating a room-temperature phonon bath may require a very
    large phonon Hilbert space. Instead, we let phonons modify the effective
    Lindblad rates.
    
    The scaling is:
    
        rate(T) = rate_base + rate_extra_at_300K * nbar(T)/nbar(300K)
    
    This is a simplified model and should be fitted to experiment for a
    real quantitative paper.
    """
    
    nT = thermal_nbar(T_K, f_ref_MHz)
    n300 = thermal_nbar(300.0, f_ref_MHz)
    
    if n300 == 0:
        scale = 0
    else:
        scale = nT / n300
    
    Gamma1e = Gamma1e_base + Gamma1e_extra_300K * scale
    Gamma_phie = Gamma_phie_base + Gamma_phie_extra_300K * scale
    
    return Gamma1e, Gamma_phie, nT


# ============================================================
# 18. Figure 5: Fidelity versus temperature
# ============================================================

def plot_fidelity_vs_temperature():
    """
    Plot memory fidelity versus physical temperature.
    
    This uses the effective temperature-dependent Lindblad-rate model,
    rather than explicitly storing thousands of thermal phonon Fock states.
    """
    
    temperatures = np.linspace(1, 300, 25)
    fidelities = []
    nbar_values = []
    
    for T in temperatures:
        Gamma1e_eff, Gamma_phie_eff, nT = effective_electron_rates_from_temperature(T)
        
        out = run_memory_protocol(
            T_storage_us=10.0,
            g_ph_MHz=0.0,
            T_ph_K=5e-5,
            Gamma1e_per_us=Gamma1e_eff,
            Gamma_phie_per_us=Gamma_phie_eff,
            Gamma1n_per_us=0.00001,
            Gamma_phin_per_us=0.00001,
            kappa_per_us=0.0,
            include_phonon_bath=False
        )
        
        fidelities.append(out["F_final"])
        nbar_values.append(nT)
    
    plt.figure(figsize=(7, 5))
    plt.plot(temperatures, fidelities, marker="o")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Final memory fidelity")
    plt.title("Memory fidelity versus temperature")
    plt.grid(True)
    plt.show()
    
    return temperatures, fidelities, nbar_values


# ============================================================
# 19. Figure 6: Fidelity heatmap F(g_ph, Ts)
# ============================================================

def plot_fidelity_heatmap():
    """
    Heatmap of final memory fidelity versus:
    
        spin-phonon coupling strength g_ph
        storage time Ts
    """
    
    g_values = np.linspace(0.0, 0.06, 9)
    storage_times = np.linspace(0, 25, 251)   # 0.1 us resolution (see Fig. 6)
    
    F_map = np.zeros((len(g_values), len(storage_times)))
    
    for i, g in enumerate(g_values):
        for j, Ts in enumerate(storage_times):
            out = run_memory_protocol(
                T_storage_us=Ts,
                g_ph_MHz=g,
                T_ph_K=5e-5,
                Gamma1e_per_us=0.0005,
                Gamma_phie_per_us=0.0005,
                Gamma1n_per_us=0.00001,
                Gamma_phin_per_us=0.00001,
                kappa_per_us=0.001,
                include_phonon_bath=True
            )
            
            F_map[i, j] = out["F_final"]
    
    plt.figure(figsize=(8, 5))
    
    plt.imshow(
        F_map,
        aspect="auto",
        origin="lower",
        extent=[
            storage_times[0],
            storage_times[-1],
            g_values[0],
            g_values[-1]
        ]
    )
    
    plt.colorbar(label="Final memory fidelity")
    plt.xlabel("Storage time Ts (microseconds)")
    plt.ylabel("Spin-phonon coupling g_ph (MHz)")
    plt.title("Fidelity heatmap: F(g_ph, Ts)")
    plt.show()
    
    return g_values, storage_times, F_map


# ============================================================
# 20. Result table
# ============================================================

def generate_results_table():
    """
    Generate a small result table for the paper.
    """
    
    cases = [
        {
            "Case": "Ideal memory",
            "g_ph_MHz": 0.0,
            "T_storage_us": 2.0,
            "Gamma1e": 0.0,
            "Gamma_phie": 0.0,
            "kappa": 0.0
        },
        {
            "Case": "Weak phonon coupling",
            "g_ph_MHz": 0.01,
            "T_storage_us": 10.0,
            "Gamma1e": 0.0005,
            "Gamma_phie": 0.0005,
            "kappa": 0.001
        },
        {
            "Case": "Strong phonon coupling",
            "g_ph_MHz": 0.05,
            "T_storage_us": 10.0,
            "Gamma1e": 0.0005,
            "Gamma_phie": 0.0005,
            "kappa": 0.001
        },
        {
            "Case": "Long storage",
            "g_ph_MHz": 0.01,
            "T_storage_us": 30.0,
            "Gamma1e": 0.0005,
            "Gamma_phie": 0.0005,
            "kappa": 0.001
        }
    ]
    
    rows = []
    
    for case in cases:
        out = run_memory_protocol(
            T_storage_us=case["T_storage_us"],
            g_ph_MHz=case["g_ph_MHz"],
            T_ph_K=5e-5,
            Gamma1e_per_us=case["Gamma1e"],
            Gamma_phie_per_us=case["Gamma_phie"],
            Gamma1n_per_us=0.00001,
            Gamma_phin_per_us=0.00001,
            kappa_per_us=case["kappa"],
            include_phonon_bath=True
        )
        
        rows.append({
            "Case": case["Case"],
            "g_ph (MHz)": case["g_ph_MHz"],
            "Storage time (us)": case["T_storage_us"],
            "Write fidelity": out["F_write"],
            "Final fidelity": out["F_final"],
            "Transfer efficiency": out["transfer_efficiency"],
            "Spin purity": out["purity_final"]
        })
    
    df = pd.DataFrame(rows)
    return df
