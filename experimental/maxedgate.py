from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorSampler

class EnhancedPredictiveModelingGate(QuantumCircuit):
    def __init__(self, num_qubits: int, data_vector: np.ndarray):
        super().__init__(num_qubits)
        self.data_vector = data_vector

    def apply_predictive_model(self):
        # Encode data
        self.append(ZZFeatureMap(self.num_qubits, reps=2, entanglement='linear'), range(self.num_qubits))
        
        # Variational layer
        self.append(RealAmplitudes(self.num_qubits, reps=3, entanglement='full'), range(self.num_qubits))
        
        # Custom operations
        for i in range(self.num_qubits - 1):
            self.cry(np.pi * self.data_vector[i], i, i + 1)
        
        for i in range(self.num_qubits):
            self.rz(np.pi * self.data_vector[i], i)

        # Measurement
        self.measure_all()

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MaxAbsScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create backend for simulation
backend = Aer.get_backend('qasm_simulator')

# Define VQC with StatevectorSampler
num_qubits = train_features.shape[1]  # Use all features
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits, reps=3, entanglement='full')
sampler = StatevectorSampler()  # Create an instance of StatevectorSampler

vqc = VQC(feature_map=feature_map,
          ansatz=ansatz,
          optimizer=COBYLA(maxiter=100),
          sampler=sampler)  # Use StatevectorSampler directly

# Train the model
vqc.fit(train_features, train_labels)

# Make predictions
train_predictions = vqc.predict(train_features)
test_predictions = vqc.predict(test_features)

# Calculate accuracies
train_accuracy = np.sum(train_predictions == train_labels) / len(train_labels)
test_accuracy = np.sum(test_predictions == test_labels) / len(test_labels)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Visualize results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(train_labels)), train_labels, c='blue', label='Actual')
plt.scatter(range(len(train_predictions)), train_predictions, c='red', marker='x', label='Predicted')
plt.title('Training Data: Actual vs Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(test_labels)), test_labels, c='blue', label='Actual')
plt.scatter(range(len(test_predictions)), test_predictions, c='red', marker='x', label='Predicted')
plt.title('Test Data: Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()

# Demonstrate custom circuit on a sample
sample_data = test_features[0]
custom_circuit = EnhancedPredictiveModelingGate(num_qubits, sample_data)
custom_circuit.apply_predictive_model()

# Simulate custom circuit using StatevectorSampler
transpiled_circuit = transpile(custom_circuit, backend)
result = sampler.run([(transpiled_circuit)], shots=1024)  # Run with shots parameter if needed

# Extract measurement outcomes from result (if applicable)
counts = result.get_counts(transpiled_circuit)

# Plot custom circuit results
plt.figure(figsize=(10, 5))
plt.bar(counts.keys(), counts.values())
plt.title('Custom Circuit Measurement Outcomes')
plt.xlabel('Measurement Outcome')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()
