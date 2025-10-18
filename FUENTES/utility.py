# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

# Residual-Dispersion Entropy
def entropy_dispersion(x, d, tau, c):
    x = np.asarray(x)
    n = len(x)

    if n < (d - 1) * tau + 1:
        raise ValueError("La serie es demasiado corta para los parámetros dados.")
    
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))# Paso 1
    embeddings = []# Paso 2

    for i in range(n - (d - 1) * tau):
        window = x_norm[i : i + tau * d : tau]
        embeddings.append(window)

    #y = [np.round(c * emb + 0.5).astype(int) for emb in embeddings]# Paso 3
    # El uso de np.clip limita al rango [1, c].
    y = [np.clip(np.round(c * emb + 0.5), 1, c).astype(int) for emb in embeddings]
    pattern = []# Paso 4

    for s in y:
        base_k = sum((s[j] - 1) * (c ** (d - j - 1)) for j in range(d))
        pattern.append(base_k)

    df = pd.Series(pattern)# Paso 5
    p = df.value_counts(normalize = True)# Paso 6
    entr = - np.sum(p * np.log2(p))# Paso 7
    r = c ** d# Paso 8
    n_enter = entr / np.log2(r)
    return(n_enter)

# -------------------------------------------------------------------------------------

# Entropía Dispersión Multiescala (MDE)
def multiscale_dispersion_entropy(x, m, tau, c, Smax):
    x = np.asarray(x)
    n = len(x)
    features = []

    # Paso 1: escala t = 1

    de = entropy_dispersion(x, m, tau, c)
    features.append(de)

    # Paso 2: escalas t = 2 hasta Smax
    for i in range(2, Smax + 1):
        entropias = []

        for k in range(i):
            subSerie = x[k:]
            t = len(subSerie) // i

            if t == 0:
                continue # Evita subseries vacías (division por cero)

            promedio = [np.mean(subSerie[j * i : (j + 1) * i]) for j in range(t)]

            # Calcular Entropía de Dispersión
            de_Promedio = entropy_dispersion(promedio, m, tau, c)
            entropias.append(de_Promedio)
        
        if entropias:
            features.append(np.mean(entropias))
        else:
            features.append(0)  # Para indicar que no se pudo calcular

    return np.array(features)

# -------------------------------------------------------------------------------------

# Entropía Dispersión Multiescala Mejorada (eMDE)
def improved_multiscale_dispersion_entropy(x, m, tau, c, Smax):
    x = np.asarray(x)
    emde = []

    # Paso 1: escala tsu = 1
    try:
        ent = entropy_dispersion(x, m, tau, c)
    except ValueError:
        ent = 0

    emde.append(ent)

    # Paso 2: escalas t = 2 hasta Smax
    for i in range(2, Smax + 1):
        entropias = []

        for k in range(i):
            subSerie = x[k:]
            Tt = len(subSerie) // i

            if t == 0:
                continue  # Evita subseries vacías

            # Promediar ventanas
            promedio = [np.mean(subSerie[j * i : (j + 1) * i]) for j in range(t)]

            # Calcular entropía de dispersión
            try:
                ent_k = entropy_dispersion(promedio, m, tau, c)
            except ValueError:
                ent_k = 0

            entropias.append(ent_k)

        # Calcular entropía promedio
        if entropias:
            emde.append(np.mean(entropias))
        else:
            emde.append(0)

    return np.array(emde)

# -------------------------------------------------------------------------------------

# Permutation Entropy
def entropy_permuta(x, m, tau):
    x = np.asarray(x)
    n = len(x)

    if n < (m - 1) * tau + 1:
        raise ValueError("La serie es demasiado corta para los parámetros dados.")
    
    patterns = []# Paso 1

    for i in range(n - (m - 1) * tau):
        window = x[i : i + tau * m : tau] # 2a
        window = window + 1e-10 * np.random.rand(m)
        pattern = tuple(np.argsort(window)) # 2b 
        patterns.append(pattern)
    
    df = pd.Series(patterns)# Paso 3
    freq = df.value_counts(normalize=True)
    entr = -np.sum(freq * np.log2(freq))# Paso 4
    n_entr = entr / np.log2(np.prod(np.arange(1, m + 1)))  # 5a
    return(n_entr)

# -------------------------------------------------------------------------------------

# Entropía permutación Multiescala (MPE)
def multiscale_permutation_entropy(x, m , tau, Smax):
    x = np.asarray(x)
    n = len(x)
    mpe = []

    for i in range(1, Smax + 1 ):
        t = n // i

        if t == 0:
            mpe.append(0)
            continue # Evita subseries vacías (division por cero)

        # Paso 1: Promediar ventanas
        y = [np.mean(x[j * i: (j + 1) * i]) for j in range(t)]

        # Paso 2 : Calcular Entropía de Permutación
        try:
            ent = entropy_permuta(y, m, tau)
        except ValueError:
            ent = 0

        mpe.append(ent)

    return np.array(mpe)

# -------------------------------------------------------------------------------------

# Entropía Permutación Multiescala Mejorada (eMPE)
def improved_multiscale_permutation_entropy(x, m, tau, Smax):
    x = np.asarray(x)
    empe = []

    # Paso 1: escala tau = 1
    try:
        ent = entropy_permuta(x, m, tau)
    except ValueError:
        ent = 0

    empe.append(ent)

    # Paso 2: escalas t = 2 hasta Smax
    for i in range(2, Smax + 1):
        entropias = []

        for k in range(i):
            subSerie = x[k:]
            t = len(subSerie) // i

            if t == 0:
                continue  # Evita subseries vacías

            # Promediar ventanas
            promedio = [np.mean(subSerie[j * i : (j + 1) * i]) for j in range(t)]

            # Calcular entropía de permutación
            try:
                ent_k = entropy_permuta(promedio, m, tau)
            except ValueError:
                ent_k = 0

            entropias.append(ent_k)

        # Calcular entropía promedio
        if entropias:
            empe.append(np.mean(entropias))
        else:
            empe.append(0)

    return np.array(empe)
