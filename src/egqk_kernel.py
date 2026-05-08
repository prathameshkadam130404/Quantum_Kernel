import pennylane as qml
import numpy as np
from tqdm import tqdm

N_QUBITS_DATA = 8

def build_egqk_circuit(q_device, selected_pairs):
    """
    Builds the PennyLane QNode natively evaluating the Entanglement-Gated Quantum 
    circuit (EGQK). By mimicking conditional tree hierarchies using Phase gating (CR_z),
    it generates states uniquely optimized for extracting pure Quantum Fidelities.
    """
    
    @qml.qnode(q_device)
    def egqk_encode(x):
        # Layer 1: Construct unconditional feature space base probability
        for i in range(N_QUBITS_DATA):
            qml.Hadamard(wires=i)
            # R_Y controls continuous amplitude probabilities based on feature magnitude
            qml.RY(x[i], wires=i)
            
        # Layer 2: Entanglement-Gated Thresholds (The Random Forest equivalent mapping)
        # Every SAR feature maps explicitly conditional on its Optical feature counterpart.
        for (q_idx, q_jdx) in selected_pairs:
            # Conditional Phase Rotation: Modifies phase strictly if parent index is sufficiently activated
            qml.CRZ(x[q_jdx], wires=[q_idx, q_jdx])
            # Bidirectional interaction logic to guarantee symmetric coupling
            qml.CRZ(x[q_idx], wires=[q_jdx, q_idx])
            
        # Optional depth-wrap constraint: 
        # Tying local neighbors sequentially solidifies the graph without overfitting 
        for i in range(N_QUBITS_DATA):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS_DATA])
            qml.RZ(x[i] * x[(i + 1) % N_QUBITS_DATA], wires=(i + 1) % N_QUBITS_DATA)

        # Output the pure, complete geometric representation vector
        # (This avoids classical 1-local geometric tracing!)
        return qml.state()
        
    return egqk_encode

def extract_egqk_fidelities(X_train, X_test, q_device, selected_pairs):
    """
    Simulates the Physical Inversion overlapping matrices by mathematically
    compiling the inner products of the state vectors. 
    Returns mathematically exact Fidelity matrices K_train and K_test.
    """
    egqk_encode = build_egqk_circuit(q_device, selected_pairs)
    
    n_tr = len(X_train)
    n_test = len(X_test)
    
    # 1. State compilation
    print(f"Pre-compiling fundamental quantum amplitudes for {n_tr + n_test} points...")
    states_tr = np.zeros((n_tr, 2**N_QUBITS_DATA), dtype=np.complex128)
    for i in tqdm(range(n_tr), desc="Training Vectors", unit="state"):
        states_tr[i] = egqk_encode(X_train[i])
        
    states_te = np.zeros((n_test, 2**N_QUBITS_DATA), dtype=np.complex128)
    for i in tqdm(range(n_test), desc="Test Vectors", unit="state"):
        states_te[i] = egqk_encode(X_test[i])
        
    # 2. Pure Fidelity evaluation (Inversion Overlap)
    print("Executing Native Quantum Inversion Mapping...")
    K_train = np.zeros((n_tr, n_tr))
    K_test = np.zeros((n_test, n_tr))
    
    for i in tqdm(range(n_tr), desc="K_train Matrix", unit="row"):
        # Vectorized probability magnitude overlap |<psi_i | psi_j>|^2
        K_train[i] = np.abs(np.dot(states_tr, np.conj(states_tr[i]))) ** 2
        
    for i in tqdm(range(n_test), desc="K_test Matrix", unit="row"):
        K_test[i] = np.abs(np.dot(states_tr, np.conj(states_te[i]))) ** 2
        
    return K_train, K_test
