#librerias
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
from sklearn.preprocessing import OneHotEncoder


#cargamos los datos (limpios)
data = pd.read_csv('Data/Processed_files/rrhh_train_limpio_final.csv')
            
#cargamos el mejor modelo.
with open('models/cat_best.pkl', 'rb') as file:
    cat_best = pickle.load(file)

#Separamos los datos.
X = data.drop('target', axis=1)
y = data['target']

#Pasamos las variables categoricas a númericas con el One Hot Encoder

encoder = OneHotEncoder(handle_unknown='ignore') 
X_encoded = encoder.fit_transform(X)

#Entrenamos el modelo

cat_best.fit(X_encoded, y)

#guardamos el modelo con los datos ya entrenados.
with open('models/cat_best_entrenado.pkl', 'wb') as file:
    pickle.dump(cat_best, file)

