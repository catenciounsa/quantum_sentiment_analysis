"""
This is a sentiment analysis system that uses
the embeddings from BERT.

It uses Classici Classifiers.

@author Carlos E. Atencio-Torres
@email catencio@unsa.edu.pe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit import Aer, execute
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC

####################################
# Constants
####################################
# number of qubits is equal to the number of features
num_qubits = 7

seed_state = 42

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
print( embeddings[:5] )

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
pca = PCA(n_components=num_qubits, random_state=seed_state)
X_pca = pca.fit_transform(embeddings)

####################################
# Defining the quantum classifier
####################################

def classic_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_state)

    clf = RandomForestClassifier(max_depth=300, random_state=seed_state)
    print("####################################")
    print("Parameters:")
    print("num_features:", num_qubits)
    print("classifier:", type(clf))

    #training
    clf.fit(X_train, y_train)

    #testing
    score = clf.score(X_test, y_test)
    print("RandomForest classification test score:", score, "at", datetime.now())

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
classic_classifier(X_pca, labels)

