import cirq
import numpy as np
import matplotlib.pyplot as plt

# Custom QPGUniqueGate definition (already covered)
class QPGUniqueGate(cirq.Gate):
    def __init__(self):
        super().__init__()

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [(1 + 1j)/np.sqrt(2), (1 - 1j)/np.sqrt(2)],
            [(1 - 1j)/np.sqrt(2), (1 + 1j)/np.sqrt(2)]
        ], dtype=np.complex128)

    def __repr__(self):
        return "QPGUniqueGate"

# Define the new algorithm
class QPGSearchPhaseShift:
    def __init__(self, target_value):
        self.target_value = target_value  # The search target value

    def create_circuit(self):
        q0 = cirq.LineQubit(0)
        q1 = cirq.LineQubit(1)

        # Construct the quantum circuit
        circuit = cirq.Circuit(
            QPGUniqueGate()(q0),  # Apply the unique gate on q0
            QPGUniqueGate()(q1),  # Apply the unique gate on q1
            cirq.H(q0),           # Apply Hadamard on q0 to create superposition
            cirq.X(q0)**self.target_value,  # Target value affects the state
            cirq.measure(q0, q1, key='result')  # Measure the qubits
        )

        return circuit

    def run(self, repetitions=100):
        circuit = self.create_circuit()

        # Run the simulation using Cirq's simulator
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=repetitions)
        return result

# Main section to execute the algorithm
def main():
    # Define and run the search algorithm with a unique target value (e.g., 1)
    search_algorithm = QPGSearchPhaseShift(target_value=1)
    result = search_algorithm.run(repetitions=100)

    # Output the result
    print(result)

    # Plot the results as a histogram
    counts = result.histogram(key='result')
    plt.bar(counts.keys(), counts.values(), color='blue')
    plt.xlabel('Measurement Outcome')
    plt.ylabel('Count')
    plt.title('Quantum Search Phase Shift Results')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
