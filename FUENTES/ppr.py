#----------------------------------------------
# Create Features by use Multi-scales Entropy
#----------------------------------------------

import pandas  as pd
import numpy   as np
from utility   import *

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
        raise ValueError(f"conf_ppr.csv: opción de entropía inválida ({opt_code}). Debe ser 1, 2, 3 o 4.")
    
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
def gets_features():
    
    return(F)

# Beginning ...
def main():
    opt, lF, d, tau, c, Smax = conf_entropy()
    data1, data2 , data3, data4 = load_data()
    x = gets_features()

if __name__ == '__main__':   
	 main()

