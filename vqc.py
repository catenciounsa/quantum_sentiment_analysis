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

from sentence_transformers import SentenceTransformer

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, EfficientSU2, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM, SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit import Aer, execute

####################################
# Constants
####################################
num_qubits = 7
seed_state = 42
lr = 0.01
maxiter=10
reps=3

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

####################################
# Reducing its dimensionality
# (1) t-SNE - method
####################################
#tsne = TSNE(n_components=num_qubits)
#X_tsne = tsne.fit_transform(embeddings)

####################################
# Reducing its dimensionality
# (1) PCA
####################################
pca = PCA(n_components=num_qubits)
X_pca = pca.fit_transform(embeddings)

####################################
# Defining the quantum classifier
####################################

information = []
def callback_func(weight, value):
    print("Run", len(information), "at", datetime.now()," weight:",weight," value:",value)
    information.append((weight, value))
    

def quantum_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_state)

    #feature_map = ZZFeatureMap(num_qubits)
    feature_map = ZFeatureMap(num_qubits, reps=1)
    #ansatz = EfficientSU2( num_qubits=num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)
    #optimizer = L_BFGS_B(maxiter=10)
    #optimizer = ADAM(maxiter=10, lr=lr)
    optimizer = SPSA(maxiter=maxiter) #No need to put learning_rate because it would need perturbation

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_func
    )

    print("####################################")
    print("Parameters:")
    print("num_qubits:", num_qubits)
    print("LR:", lr)
    print("optimizer-iterations:", maxiter)
    print("Feature_map:", type(feature_map))
    print("Feature map - layers/num_parameters:", feature_map.num_layers, "/", feature_map.num_parameters)
    print("Ansatz reps:", reps)
    print("ansatz:", type(ansatz))
    print("optimizer:", type(optimizer))
    print("####################################")


    vqc.fit(X_train, y_train)
    score = vqc.score(X_test, y_test)
    print("VQC classification test score:", score, "at", datetime.now())

####################################
# Let us run both: TSNE & PCA
####################################
# TSNE presented problems. It throw an error indicating that TSNE 
# only can work with dimenssion lower than 4
#print("####################################")
#print("TSNES starting at ", datetime.now())
#quantum_classifier(X_tsne, labels)
print("####################################")
print("Running classifer at", datetime.now())
quantum_classifier(X_pca, labels)
