"""
Q-SHIELD: Quantum AI-Based Drug-Abuse Risk Mapping & Prevention
Safe prototype using synthetic, area-level risk data.
"""

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Operator, Statevector

AREAS = ["Area A", "Area B", "Area C", "Area D"]
RISKS = np.array([0.82, 0.35, 0.91, 0.67])
BUDGET = 2
PENALTY = 2.0

def cost(bits):
    x = np.array(bits, dtype=int)
    return -float(np.dot(RISKS, x)) + PENALTY * (int(x.sum()) - BUDGET) ** 2

def build_cost_operator():
    n = len(AREAS)
    dim = 2 ** n
    diagonal = []
    for state in range(dim):
        bits = [(state >> i) & 1 for i in range(n)]
        diagonal.append(cost(bits))
    return Operator(np.diag(diagonal))

def run_qaoa_grid():
    op = build_cost_operator()
    circuit = QAOAAnsatz(cost_operator=op, reps=1, flatten=True)
    best = None

    for gamma in np.linspace(0, 2*np.pi, 9):
        for beta in np.linspace(0, np.pi, 9):
            bound = circuit.assign_parameters([gamma, beta])
            state = Statevector.from_instruction(bound)
            probabilities = state.probabilities()

            energy = 0.0
            for i, p in enumerate(probabilities):
                bits = [(i >> j) & 1 for j in range(len(AREAS))]
                energy += p * cost(bits)

            if best is None or energy < best["energy"]:
                best = {"energy": float(energy), "gamma": float(gamma),
                        "beta": float(beta), "probabilities": probabilities}

    valid = []
    for i, p in enumerate(best["probabilities"]):
        bits = [(i >> j) & 1 for j in range(len(AREAS))]
        if sum(bits) == BUDGET:
            valid.append((float(p), i, bits))

    _, index, bits = max(valid, key=lambda t: t[0])
    selected = [AREAS[i] for i, b in enumerate(bits) if b]

    return {
        "selected_areas": selected,
        "bitstring": "".join(map(str, bits)),
        "energy": best["energy"],
        "gamma": best["gamma"],
        "beta": best["beta"],
        "circuit": circuit,
    }

if __name__ == "__main__":
    result = run_qaoa_grid()
    print("Area risk scores:")
    for area, risk in zip(AREAS, RISKS):
        print(f"  {area}: {risk:.2f}")
    print("\nQAOA priority areas:", ", ".join(result["selected_areas"]))
    print("Bitstring:", result["bitstring"])
    print("Energy:", round(result["energy"], 4))
