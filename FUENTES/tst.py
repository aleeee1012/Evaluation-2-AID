# Testing for Softmax Regresion
import numpy as np
import pandas as pd
from ppr import conf_entropy # Solo para ver los resultados mas ordenados

# ------ Cálculo de Métricas de Evaluación ---------

def calculate_metrics(Y_true, Y_pred, n_classes):
    # Matriz de Confusión
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Convierte las etiquetas one-hot a etiquetas de una sola columna (0, 1, 2, 3)
    y_true_labels = np.argmax(Y_true, axis=1)
    
    # Itera sobre cada predicción para llenar la matriz
    for i in range(len(y_true_labels)):
        true_label = y_true_labels[i]
        pred_label = Y_pred[i]
        conf_matrix[true_label, pred_label] += 1
        
    print("\nMatriz de Confusión:")
    print(conf_matrix)

    # F-scores
    f_scores = []
    # Itera sobre cada clase para calcular su F-score
    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        # Calcular Precisión y Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calcular F-score
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_scores.append(f_score)
    
    print("\nF1-Scores por clase:")
    for i, score in enumerate(f_scores):
        print(f"  - Clase #{i+1}: {score:.4f}")
        
    return conf_matrix, np.array(f_scores).reshape(1, -1)

# --------------- Softmax function -----------------
def softmax(z):
    # Se resta el máximo de z para estabilidad numérica y evitar 'overflow'
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0,keepdims=True))

def my_softmax(x,w):
    z= w@x
    return softmax(z) 

# Beginning ...
def main():
    print("Evaluación de Rendimiento...")
    
    # Cargar los datasets de testing y los pesos del modelo
    print("Cargando datos de prueba y pesos del modelo...")
    try:
        X_test = pd.read_csv('dtst.csv', header=None).values
        Y_test = pd.read_csv('dtst_label.csv', header=None).values

        # Cargar pesos y separar W (pesos) de b (sesgo)
        pesos_b = pd.read_csv('pesos.csv', header=None).values
        W = pesos_b[:-1, :] # Todas las filas excepto la última
        b = pesos_b[-1, :]  # La última fila
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}.")
        print("Asegúrate de haber ejecutado ppr.py y trn.py primero.")
        return

    # Obtener los valores estimados usando Regresión Softmax
    print("Realizando predicciones en el conjunto de prueba...")

    # Calcular scores
    Z = np.dot(X_test, W) + b

    # Aplicar softmax
    Y_hat = softmax(Z)

    # La predicción final es la clase con la probabilidad más alta
    Y_pred = np.argmax(Y_hat, axis=1)

    opt, lF, d, tau, c, Smax = conf_entropy()

    # Mostrar tipo de entropía antes de procesar
    print("--------------------------------------------------")
    print(f"Tipo de Entropía seleccionada: {opt}")
    print("--------------------------------------------------")

    # Calcular métricas de rendimiento
    n_classes = Y_test.shape[1]
    confusion_matrix, fscores = calculate_metrics(Y_test, Y_pred, n_classes)
    
    # Crear y guardar los archivos de resultados
    print("\nGuardando resultados en 'cmatriz.csv' y 'fscores.csv'...")
    pd.DataFrame(confusion_matrix).to_csv('cmatriz.csv', index=False, header=False)
    pd.DataFrame(fscores).to_csv('fscores.csv', index=False, header=False)
    
    print("\nEvaluación completada.")

if __name__ == '__main__':   
	 main()