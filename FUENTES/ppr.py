#----------------------------------------------
# Create Features by use Multi-scales Entropy
#----------------------------------------------

import pandas  as pd
import numpy   as np
from utility   import (multiscale_dispersion_entropy, improved_multiscale_dispersion_entropy, multiscale_permutation_entropy, improved_multiscale_permutation_entropy)

# Carga parametros del "Conf_ppr.csv"
def conf_entropy():
    config = pd.read_csv("FUENTES/conf_ppr.csv", header=None).values.flatten()
    opt_code = int(config[0])  # 1 = Dispersión, 2 = Disp-Mejorada, 3 = Permutación, 4 = Perm-Mejorada

    if opt_code == 1:
        opt = 'dispersion'
    elif opt_code == 2:
        opt = 'dispersion-mejorada'
    elif opt_code == 3:
        opt = 'permutación'
    elif opt_code == 4:
        opt = 'permutación-mejorada'
    else:
        raise ValueError(f"conf_ppr.csv: opción de entropía inválida ({opt_code}). Elija entre las siguientes opciones de Entropía Multi-escala: 1.- Dispersión | 2.- Dispersión Mejorada | 3.- Permutación | 4.- Permutación Mejorada")
    
    lF = int(config[1]) # Longitud del segmento
    d = int(config[2]) # Dimensión embebida
    tau = int(config[3]) # Factor de retardo embebido
    c = int(config[4]) # Número de clase de Entropía Dispersión
    Smax = int(config[5]) # Número Máximo de Escalas
    return opt, lF, d, tau, c, Smax

def load_data():
    df1 = pd.read_csv("G2/class1.csv", header=None).values
    df2 = pd.read_csv("G2/class2.csv", header=None).values
    df3 = pd.read_csv("G2/class3.csv", header=None).values
    df4 = pd.read_csv("G2/class4.csv", header=None).values

    # Calcular la derivada usando diferencias finitas 
    data1 = np.diff(df1, axis = 0)
    data2 = np.diff(df2, axis = 0)
    data3 = np.diff(df3, axis = 0)
    data4 = np.diff(df4, axis = 0)

    return data1, data2 , data3, data4

# Obtain Features by use Entropy    
def gets_features(opt, lF, d, tau, c, Smax, all_data):
    all_features = []
    all_labels = []
    
    # Mapeo a etiquetas 
    labels_map = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # Itera sobre cada set
    for class_idx, class_data in enumerate(all_data):
        print(f"\nProcesando Clase #{class_idx + 1}...")

        # Itera sobre cada muestra dentro de un archivo
        for sample_idx in range(class_data.shape[1]):
            sample_series = class_data[:, sample_idx]
            
            # Segmenta
            n_segments = len(sample_series) // lF
            
            # Procesa cada segmento
            for seg_idx in range(n_segments):
                segment = sample_series[seg_idx * lF : (seg_idx + 1) * lF]
                features_vector = []
                
                # Multi escala
                try:
                    if opt == 1: # Dispersión
                        features_vector = multiscale_dispersion_entropy(segment, Smax, m=d, tau=tau, c=c)
                    elif opt == 2: # Dispersión Mejorada
                        features_vector = improved_multiscale_dispersion_entropy(segment, Smax, m=d, tau=tau, c=c)
                    elif opt == 3: # Permutación
                        features_vector = multiscale_permutation_entropy(segment, Smax, m=d, tau=tau)
                    elif opt == 4: # Permutación Mejorada
                        features_vector = improved_multiscale_permutation_entropy(segment, Smax, m=d, tau=tau)
                    else:
                        raise ValueError(f"Opción de entropía inválida: {opt}")

                except ValueError as e:
                    print(f"Error al calcular entropía para un segmento: {e}.")
                    exit

                all_features.append(features_vector)
                all_labels.append(labels_map[class_idx])

    F = pd.DataFrame(all_features)
    L = pd.DataFrame(all_labels)

    return F, L

# Beginning ...
def main():
    # Cargar Configuración
    opt, lF, d, tau, c, Smax = conf_entropy()

    # Cargar data
    data1, data2 , data3, data4 = load_data()

    # Lista para datas
    all_data = [data1, data2, data3, data4]

    F, L = gets_features(opt, lF, d, tau, c, Smax, all_data)

    # Guarda archivos (clase y label)
    F.to_csv('dClases.csv', index=False, header=False)
    L.to_csv('dLabel.csv', index=False, header=False)


if __name__ == '__main__':   
	 main()

