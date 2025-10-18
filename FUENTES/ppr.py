#----------------------------------------------
# Create Features by use Multi-scales Entropy
#----------------------------------------------

import pandas  as pd
import numpy   as np
from utility   import *

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
    data1, data2 , data3, data4 = load_data()
    x = gets_features()

if __name__ == '__main__':   
	 main()

