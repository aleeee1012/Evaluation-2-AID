# Softmac Regression's Training :

import numpy  as np
import pandas as pd
from utility  import softmax

# ------ Configuracion y preparación de datos ---------

def load_train_config():
    # Leer parámetros
    config = pd.read_csv("FUENTES/conf_train.csv", header=None).values.flatten()
    max_iter = int(config[0])
    learning_rate = float(config[1])
    train_percentage = float(config[2]) / 100.0  # Convertir a fracción (ej. 70 -> 0.7)
    
    # Parametro beta para descenso gradiente
    beta = 0.9
    
    print("Configuración de entrenamiento cargada:")
    print(f" - Máximo de iteraciones: {max_iter}")
    print(f" - Tasa de aprendizaje: {learning_rate}")
    print(f" - Porcentaje de entrenamiento: {train_percentage*100}%")
    
    return max_iter, learning_rate, train_percentage, beta

def prepare_data(train_percentage):
    # Cargar los datasets generados por ppr.py
    print("\nCargando dClases.csv y dLabel.csv...")
    X = pd.read_csv("dClases.csv", header=None).values
    Y = pd.read_csv("dLabel.csv", header=None).values

    # Re-ordenar aleatoriamente las muestras
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    X = X[indices]
    Y = Y[indices]

    # Normalizar el dataset de Características
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1 # Evitar división por cero
    X_norm = (X - mean) / std
    
    # Guardar media y std para usar en tst.py
    np.savetxt('norm_params.csv', np.vstack((mean, std)), delimiter=',')

    # Dividir el dataset para Training/Testing
    split_idx = int(num_samples * train_percentage)
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Crear y guardar los archivos CSV correspondientes
    pd.DataFrame(X_train).to_csv('dtrn.csv', index=False, header=False)
    pd.DataFrame(Y_train).to_csv('dtrn_label.csv', index=False, header=False)
    pd.DataFrame(X_test).to_csv('dtst.csv', index=False, header=False)
    pd.DataFrame(Y_test).to_csv('dtst_label.csv', index=False, header=False)
    
    print(f"Datos divididos en {X_train.shape[0]} para entrenamiento y {X_test.shape[0]} para prueba.")

    return X_train, Y_train

# ------ Funciones Modelo Softmax ---------

def compute_cost(Y_hat, Y):
    m = Y.shape[0]
    # Se suma epsilon para evitar el log(0).
    cost = - (1 / m) * np.sum(Y * np.log(Y_hat + 1e-9))
    return cost

# -------- Funcion entrenamiento con mGD -----------

def train(X, Y, max_iter, learning_rate, beta):
    m, n_features = X.shape
    _, n_classes = Y.shape

    # Inicializar pesos (W) y sesgo/bias (b) con ceros
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))
    
    # Inicializar variables de momentum
    v_dW = np.zeros_like(W)
    v_db = np.zeros_like(b)

    cost_history = []

    for i in range(max_iter):
        # --- Forward Propagation) ---
        Z = np.dot(X, W) + b
        Y_hat = softmax(Z)

        # --- Cálculo del Costo ---
        cost = compute_cost(Y_hat, Y)
        cost_history.append(cost)

        # --- Backward Propagation ---
        # Cálculo de los gradientes
        dZ = Y_hat - Y
        dW = (1 / m) * np.dot(X.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        # --- Actualización con Momentum ---
        v_dW = (beta * v_dW) + ((1 - beta) * dW)
        v_db = (beta * v_db) + ((1 - beta) * db)

        # --- Actualización de Parámetros (Pesos y Sesgo) ---
        W -= learning_rate * v_dW
        b -= learning_rate * v_db

        # Imprimir el costo cada 100 iteraciones para ver el progreso
        if (i + 1) % 100 == 0:
            print(f"  Iteración {i+1}/{max_iter} -> Costo: {cost:.6f}")
    
    print("Entrenamiento completado.")
    
    # Combinar pesos y sesgo en una sola matriz para guardarlos
    final_weights = np.vstack((W, b))
    return final_weights, cost_history


# Beginning ...
def main():    
    # Cargar la configuración
    max_iter, learning_rate, train_percentage, beta = load_train_config()
    
    # Preparar los datos
    X_train, Y_train = prepare_data(train_percentage)
    
    # Entrenar el modelo
    weights, cost_vector = train(X_train, Y_train, max_iter, learning_rate, beta)
    
    # Guardar los resultados del modelo 
    pd.DataFrame(weights).to_csv('pesos.csv', index=False, header=False) # 
    pd.DataFrame(cost_vector).to_csv('costo.csv', index=False, header=False) # 
    print("\nArchivos 'pesos.csv' y 'costo.csv' guardados correctamente.")
       
if __name__ == '__main__':   
	 main()

