import csv
import pandas as pd

# Load data
train = pd.read_csv('../data/datosEntrenamiento.csv')
test = pd.read_csv('../data/datosPrueba.csv')
validation = pd.read_csv('../data/datosValidacion.csv')