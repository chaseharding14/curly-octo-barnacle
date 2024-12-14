import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Use AerSimulator for simulation
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA  # Correct optimizer import
from qiskit_machine_learning.algorithms import VQC  # Correct VQC import

# Define the quantum circuit for predictive modeling
class EnhancedPredictiveModelingGate(QuantumCircuit):
    def __init__(self, num_qubits: int, data_vector: np.ndarray):
        super().__init__(num_qubits)
        self.data_vector = data_vector

    def apply_predictive_model(self):
        # Encode data
        self.append(ZZFeatureMap(self.num_qubits, reps=2, entanglement='linear'), range(self.num_qubits))
        
        # Variational layer
        self.append(RealAmplitudes(self.num_qubits, reps=3, entanglement='full'), range(self.num_qubits))
        
        # Custom operations based on input data_vector
        for i in range(self.num_qubits - 1):
            self.cry(np.pi * self.data_vector[i], i, i + 1)
        
        for i in range(self.num_qubits):
            self.rz(np.pi * self.data_vector[i], i)

        # Measurement
        self.measure_all()

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target

# Use only a subset of features to reduce complexity (e.g., first 4 features)
X = X[:, :4]  

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MaxAbsScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create backend for simulation using Aer simulator
backend = AerSimulator()  # Using AerSimulator directly

# Define VQC with Sampler
num_qubits = train_features.shape[1]  # Use reduced number of features (4)
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')

# Hyperparameter tuning: Experiment with different repetitions (reps)
best_accuracy = 0
best_reps = 0

for reps in range(1, 5):  # Try different numbers of repetitions from 1 to 4
    ansatz = RealAmplitudes(num_qubits, reps=reps + 1)  # Increase reps for complexity
    vqc = VQC(feature_map=feature_map,
              ansatz=ansatz,
              optimizer=COBYLA(maxiter=100))

    # Train the model using cross-validation to evaluate performance more robustly
    kf = KFold(n_splits=5)  # Use 5-fold cross-validation

    accuracies = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Normalize features for each fold
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Fit the model on the training set
        vqc.fit(X_train[:, :num_qubits], y_train)

        # Make predictions on the validation set
        val_predictions = vqc.predict(X_val[:, :num_qubits])
        
        # Calculate accuracy for this fold and store it
        accuracy = np.sum(val_predictions == y_val) / len(y_val)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    print(f"Reps: {reps + 1}, Mean Cross-Validation Accuracy: {mean_accuracy:.2f}")

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_reps = reps + 1  # Store best reps including additional complexity

print(f"Best Reps: {best_reps}, Best Mean Accuracy: {best_accuracy:.2f}")

# Final training on full training set with the best found hyperparameters
final_ansatz = RealAmplitudes(num_qubits, reps=best_reps)
vqc_final = VQC(feature_map=feature_map,
                ansatz=final_ansatz,
                optimizer=COBYLA(maxiter=100))

vqc_final.fit(train_features[:, :num_qubits], train_labels)
test_predictions = vqc_final.predict(test_features[:, :num_qubits])

# Calculate additional metrics on the test set
print("Classification Report:")
print(classification_report(test_labels, test_predictions))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Demonstrate custom circuit on a sample from the test set
sample_data = test_features[0]
custom_circuit = EnhancedPredictiveModelingGate(num_qubits, sample_data)
custom_circuit.apply_predictive_model()

# Simulate custom circuit using StatevectorSampler on a simulator (Aer backend)
transpiled_circuit = transpile(custom_circuit, backend)
result = sampler.run(transpiled_circuit, shots=100)  # Reduced shots for faster execution

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
