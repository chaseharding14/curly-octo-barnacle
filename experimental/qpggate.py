import cirq
import numpy as np
import matplotlib.pyplot as plt

class QPGGate(cirq.Gate):
    def __init__(self):
        super().__init__()

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["QPG"] * self.num_qubits()

    def __str__(self):
        return "QPGGate"

# Define qubits and the circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),  # Apply a Hadamard gate to q0
    cirq.H(q1),  # Apply a Hadamard gate to q1
    QPGGate()(q0, q1),  # Apply the custom QPGGate
    cirq.measure(q0, key='q0'),  # Measure q0
    cirq.measure(q1, key='q1')   # Measure q1
)

# Print the circuit
print("Circuit:")
print(circuit)

# Simulate the circuit
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)

# Output the result
print("\nSimulation Results:")
print(result)

# Visualize the measurement results for both qubits
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

for i, q in enumerate(['q0', 'q1']):
    q_results = result.measurements[q]
    q_count_0 = np.sum(q_results == 0)
    q_count_1 = np.sum(q_results == 1)

    labels = ['0', '1']
    counts = [q_count_0, q_count_1]

    ax = ax0 if i == 0 else ax1
    ax.bar(labels, counts, color=['blue', 'orange'])
    ax.set_xlabel('Measurement Outcome')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Measurement Distribution for {q}')

plt.tight_layout()
plt.show()
