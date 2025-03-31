from azure.quantum import Workspace
from azure.identity import InteractiveBrowserCredential  # Import for active login
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN  # Updated import
from sklearn.datasets import make_classification  # Import for generating synthetic dataset
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler  # Import Sampler for backend execution
from qiskit.circuit import ParameterVector  # Import ParameterVector for parameterized circuits
from azure.quantum.qiskit import AzureQuantumProvider  # Import AzureQuantumProvider

def build_workspace(subscription_id, resource_group, workspace_name, location):
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        name=workspace_name,
        location=location
    )
    return workspace

def build_workspace_with_token(subscription_id, resource_group, workspace_name, location, tenant_id):
    # Use InteractiveBrowserCredential with tenant_id for active login
    credential = InteractiveBrowserCredential(tenant_id=tenant_id)
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        name=workspace_name,
        location=location,
        credential=credential  # Pass the credential for authentication
    )
    return workspace

if __name__ == "__main__":
    # Step 1: Connect to Azure Quantum Workspace with active login
    ws = build_workspace_with_token(
        "e4420dbb-ea34-41cf-b047-230c73836759",  # subscription_id
        "AzureQuantum",  # resource_group
        "hsunq",  # workspace_name
        "japaneast",  # location
        "16b3c013-d300-468d-ac64-7eda0820b6d3"  # tenant_id
    )

    # Step 1.1: List available providers
    provider = AzureQuantumProvider(workspace=ws)  # Create a provider from the workspace
    print("Available providers:")
    for backend in provider.backends():
        print(f"- {backend.name()}")

    # Step 2: Prepare QML Dataset
    feature_dim = 2
    training_features, training_labels = make_classification(
        n_samples=30, n_features=feature_dim, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    training_features = training_features[:20]  # Use the first 20 samples for training
    training_labels = training_labels[:20]  # Use the first 20 labels for training

    # Step 3: Define Quantum Circuit and QNN
    feature_map = QuantumCircuit(feature_dim, feature_dim)  # Add classical registers
    feature_params = ParameterVector("x", length=feature_dim)  # Parameterize the feature map
    for i in range(feature_dim):
        feature_map.h(i)
        feature_map.rx(feature_params[i], i)  # Use parameterized rotation gates
    feature_map.barrier()

    ansatz = QuantumCircuit(feature_dim, feature_dim)  # Add classical registers
    ansatz_params = ParameterVector("θ", length=feature_dim)  # Parameterize the ansatz
    for i in range(feature_dim):
        ansatz.rx(ansatz_params[i], i)  # Use parameterized rotation gates

    circuit = feature_map.compose(ansatz)
    circuit.measure(range(feature_dim), range(feature_dim))  # Add measurement operations

    qnn = EstimatorQNN(circuit=circuit, input_params=feature_params, weight_params=ansatz_params)

    # Step 4: Create and Train VQC
    sampler = Sampler()  # Create a Sampler instance
    vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz)  # Use sampler instead of estimator
    vqc.fit(training_features, training_labels)

    # Step 5: Submit Job to Azure Quantum
    backend = provider.get_backend("ionq.simulator")  # Replace with your target backend
    job = backend.run(circuit)  # Submit the circuit directly
    print(f"Job ID: {job.id()}")
    print(f"Job Status: {job.status()}")
