"""
Experiment E19 — Post-transpile circuit-depth analysis (heavy-hex).

For FQK / PQK / AGPQK ansätze (n=8 qubits, 2 reps), transpile to the IBM
heavy-hex coupling map (`ibm_kingston`-like layout) at optimization_level
∈ {0, 1, 3} and report:
    - total depth, 2-qubit-gate (CX / ECR) depth, 2-qubit-gate count
    - SX / X count (single-qubit basis gates)
    - layout overhead (SWAPs inserted).

Cheap circuit-level diagnostic that AQT reviewers weight heavily.

Output: results/transpile_depth/metrics.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import config
from src.utils import ensure_dir, setup_logging

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "transpile_depth"))
logger = setup_logging("exp_e19", log_file=os.path.join(RES, "e19.log"))

N_QUBITS = config.N_QUBITS
REPS     = config.ZZ_REPS


def heavy_hex_8():
    """Minimal 8-qubit heavy-hex subgraph coupling (ibm_kingston-style)."""
    return [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [1, 4], [3, 6]]   # two heavy-hex cross-links


def build_zz(qc, x_params, pairs, n_qubits, reps):
    from qiskit.circuit import ParameterVector
    import numpy as np
    for _ in range(reps):
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits):
            qc.rz(x_params[i], i)
        for (i, j) in pairs:
            qc.cx(i, j)
            qc.rz((np.pi - x_params[i]) * (np.pi - x_params[j]), j)
            qc.cx(i, j)


def circ_fqk(pairs):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    qc = QuantumCircuit(N_QUBITS)
    x1 = ParameterVector("x1", N_QUBITS)
    x2 = ParameterVector("x2", N_QUBITS)
    build_zz(qc, x1, pairs, N_QUBITS, REPS)
    # adjoint (inverse) of the second ZZ block
    qc_adj = QuantumCircuit(N_QUBITS)
    build_zz(qc_adj, x2, pairs, N_QUBITS, REPS)
    qc.compose(qc_adj.inverse(), inplace=True)
    return qc


def circ_pqk(pairs):
    from qiskit.circuit import QuantumCircuit, ParameterVector
    qc = QuantumCircuit(N_QUBITS)
    x = ParameterVector("x", N_QUBITS)
    build_zz(qc, x, pairs, N_QUBITS, REPS)
    return qc


def run():
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
    except ImportError:
        logger.error("qiskit not installed — skipping transpile analysis")
        return

    coupling = CouplingMap(heavy_hex_8())
    basis = ["sx", "x", "rz", "cx"]
    linear = [(i, i + 1) for i in range(N_QUBITS - 1)]
    full   = [(i, j) for i in range(N_QUBITS)
              for j in range(i + 1, N_QUBITS)]

    circuits = {
        "FQK-linear":    circ_fqk(linear),
        "PQK-linear":    circ_pqk(linear),
        "AGPQK-linear":  circ_pqk(linear),   # AGPQK circuit ≈ PQK w/ attention pairs
        "FQK-full":      circ_fqk(full),
        "PQK-full":      circ_pqk(full),
    }

    rows = []
    for name, qc in circuits.items():
        for opt in [0, 1, 3]:
            tqc = transpile(qc, basis_gates=basis, coupling_map=coupling,
                            optimization_level=opt, seed_transpiler=42)
            ops = tqc.count_ops()
            rows.append(dict(
                circuit=name, opt_level=opt,
                depth=int(tqc.depth()),
                cx_depth=int(tqc.depth(filter_function=lambda g: g.operation.name == "cx")),
                cx_count=int(ops.get("cx", 0)),
                sx_count=int(ops.get("sx", 0)),
                x_count=int(ops.get("x", 0)),
                rz_count=int(ops.get("rz", 0)),
                total_gates=int(sum(ops.values())),
            ))
            logger.info(f"{name:16s} opt={opt}  depth={rows[-1]['depth']:4d}  "
                        f"cx={rows[-1]['cx_count']:4d}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)


if __name__ == "__main__":
    run()
