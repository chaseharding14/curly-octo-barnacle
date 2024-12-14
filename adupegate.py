from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorSampler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

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

# Use only a subset of features to reduce complexity (e.g., first 4 features)
X = X[:, :4]  

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MaxAbsScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create backend for simulation
backend = Aer.get_backend('aer_simulator')  # Using statevector simulator

# Define VQC with StatevectorSampler
num_qubits = train_features.shape[1]  # Use reduced number of features (4)
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits, reps=3, entanglement='full')
sampler = StatevectorSampler()  # Create an instance of StatevectorSampler

vqc = VQC(feature_map=feature_map,
          ansatz=ansatz,
          optimizer=COBYLA(maxiter=100),
          sampler=sampler)  # Use StatevectorSampler directly

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

print(f"Cross-Validation Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.2f}, Std Dev: {np.std(accuracies):.2f}")

# Final training on full training set and evaluate on test set
vqc.fit(train_features[:, :num_qubits], train_labels)
test_predictions = vqc.predict(test_features[:, :num_qubits])

# Calculate additional metrics on the test set
print("Classification Report:")
print(classification_report(test_labels, test_predictions))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Analysis using permutation importance
result = permutation_importance(vqc, train_features[:, :num_qubits], train_labels, n_repeats=10)

# Display feature importance
importance_df = pd.DataFrame({'Feature': data.feature_names[:num_qubits], 'Importance': result.importances_mean})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# Demonstrate custom circuit on a sample from the test set
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
