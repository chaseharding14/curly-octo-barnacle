import cirq
import numpy as np
import matplotlib.pyplot as plt

# Define a custom quantum gate, for example, a random gate
class RandomGate(cirq.Gate):
    def __init__(self, theta):
        self.theta = theta

    # Implementing the abstract methods
    def _num_qubits_(self):
        return 1  # The gate operates on a single qubit

    def _qid_shape_(self):
        return (2,)  # Qubits have two possible states (0 and 1)

    def num_qubits(self):
        return 1  # The gate operates on one qubit

    def __str__(self):
        return f"RandomGate({self.theta})"

    def _unitary_(self):
        # The gate applies a rotation by an angle theta
        return np.array([
            [np.cos(self.theta / 2), -np.sin(self.theta / 2)],
            [np.sin(self.theta / 2), np.cos(self.theta / 2)]
        ])

# Create a qubit
q1 = cirq.NamedQubit('q1')

# Create the random gate with a randomly chosen angle (between 0 and 2*pi)
theta = np.random.uniform(0, 2 * np.pi)
random_gate = RandomGate(theta)

# Create a circuit and add operations
circuit = cirq.Circuit(
    cirq.H(q1),  # Hadamard gate to initialize the qubit
    random_gate(q1),  # Apply the custom random gate
    cirq.measure(q1, key='m1')
)

# Simulate the circuit
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)

# Print the results
print("Measurement results:")
print(result)

# Count the occurrences of 0 and 1
zeros_count = np.sum(result.measurements['m1'] == 0)
ones_count = np.sum(result.measurements['m1'] == 1)

print(f"Number of 0's: {zeros_count}")
print(f"Number of 1's: {ones_count}")

# Visualize the result in a histogram
fig, ax = plt.subplots()
ax.hist(result.measurements['m1'], bins=np.arange(2) - 0.5, edgecolor='black')
ax.set_xlabel('Measurement Outcome')
ax.set_ylabel('Frequency')
ax.set_title('Measurement Results')
plt.show()
