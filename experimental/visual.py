print("Running visual.py...")
import cirq
print("Cirq imported successfully...")

import cirq

# Create the qubits
q1, q2 = cirq.LineQubit.range(2)

# Build the circuit
circuit = cirq.Circuit(
    cirq.H(q1),
    cirq.CNOT(q1, q2),
    cirq.measure(q1, key='m1'),
    cirq.measure(q2, key='m2')
)

# Print the circuit diagram
print(circuit)
