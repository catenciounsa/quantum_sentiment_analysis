"""
This is a sentiment analysis system that uses
the embeddings from BERT.

It uses a Quantum Kernel Classifier

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
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit import Aer, execute

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

    feature_map = ZZFeatureMap(num_qubits, reps=reps, entanglement='linear')
    adhoc_kernel = FidelityQuantumKernel(feature_map=feature_map)

    print("####################################")
    print("Parameters:")
    print("num_qubits:", num_qubits)
    print("Feature_map:", type(feature_map))
    print("Feature map - layers/num_parameters:", feature_map.num_layers, "/", feature_map.num_parameters)
    print("Feature map - parameters:", "reps=", reps)
    print("qkernel:", type(adhoc_kernel))
    
    qsvc = QSVC(quantum_kernel=adhoc_kernel)

    #training
    qsvc.fit(X_train, y_train)

    #testing
    score = qsvc.score(X_test, y_test)
    print("QuantumKernel classification test score:", score, "at", datetime.now())

print("####################################")
print("Running classifer at", datetime.now())
quantum_classifier(X_pca, labels)
