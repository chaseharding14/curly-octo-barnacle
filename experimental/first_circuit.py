import cirq

# Step 1: Create qubits
qubit_1 = cirq.NamedQubit("q1")
qubit_2 = cirq.NamedQubit("q2")

# Step 2: Define the circuit
circuit = cirq.Circuit(
    cirq.H(qubit_1),                 # Apply Hadamard to q1
    cirq.CNOT(qubit_1, qubit_2),     # Apply CNOT with q1 as control and q2 as target
    cirq.measure(qubit_1, key='m1'), # Measure q1
    cirq.measure(qubit_2, key='m2')  # Measure q2
)

# Step 3: Simulate the circuit
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=10)

# Output the results
print("Circuit:")
print(circuit)
print("\nResults:")
print(result)