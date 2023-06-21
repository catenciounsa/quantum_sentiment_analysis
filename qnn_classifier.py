"""
This is a sentiment analysis system that uses
the embeddings from BERT.

It uses a Quantum Classifier.

@author Carlos E. Atencio-Torres
@email catencio@unsa.edu.pe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sentence_transformers import SentenceTransformer 

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2, RealAmplitudes
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

####################################
# Constants
####################################
num_qubits = 7
num_shots = 128
seed_state = 42
algorithm_globals.random_seed = 42

reps = 2

####################################
# Reading the information
####################################
csv_file = "IMDB-Dataset.csv"
df = pd.read_csv( csv_file )

df["label"].replace("positive",0, inplace=True)
df["label"].replace("negative",1, inplace=True)
df["label"] = df["label"].astype(int)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode( df["review"].to_numpy() )
labels = df["label"].to_numpy()

print("Embeddings loaded", datetime.now())

####################################
# Reducing its dimensionality
# (1) PCA - method
####################################
pca = PCA(n_components=num_qubits, random_state=seed_state)
X_pca = pca.fit_transform(embeddings)

####################################
# Defining the quantum classifier
####################################

information = []
def callback_func(weight, value):
    if len(information)%50 == 0:
        print("Run", len(information), "at", datetime.now()," weight:",weight," value:",value)
    information.append((weight, value))
    

def quantum_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_state)

    qc = QuantumCircuit(num_qubits)
    feature_map = ZZFeatureMap(num_qubits)
    ansatz = RealAmplitudes(num_qubits)
    qc.compose( feature_map, inplace=True )
    qc.compose( ansatz, inplace=True )

    estimator_qnn = EstimatorQNN(
        circuit=qc, input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    estimator_qnn.forward( X[0, :], algorithm_globals.random.random(estimator_qnn.num_weights))

    # construct neural network classifier
    estimator_classifier = NeuralNetworkClassifier(
        estimator_qnn, optimizer=COBYLA(maxiter=60), callback=callback_func
    )

    print("####################################")
    print("Parameters:")
    print("num_qubits:", num_qubits)
    print("Feature_map:", type(feature_map))
    print("Feature map - layers/num_parameters:", feature_map.num_layers, "/", feature_map.num_parameters)
    print("Feature map - parameters:", "reps=", reps)

    
    estimator_classifier.fit(X_train, y_train)

    score = estimator_classifier.score(X_test, y_test)
    print("Test score:",score, "at", datetime.now())

    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(information)), information)
    plt.savefig( "qnn_classifier.png" )
    


print("####################################")
print("Running classifer at", datetime.now())

quantum_classifier(X_pca, labels)
