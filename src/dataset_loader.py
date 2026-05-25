import pickle
import numpy as np

def cargar_radio_ml(path_archivo):
    """
    Carga el dataset RadioML 2016.10a y lo prepara para el entrenamiento.
    """
    print(f"Cargando dataset desde {path_archivo}...")
    
    # El dataset original fue empaquetado en Python 2, por lo que usamos latin1
    with open(path_archivo, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    secuencias_iq = []
    etiquetas_mod = []
    valores_snr = []
    
    # Recorrer el diccionario extraído
    for (mod, snr), muestras in df.items():
        for muestra in muestras:
            secuencias_iq.append(muestra)
            etiquetas_mod.append(mod)
            valores_snr.append(snr)
            
    # Convertir a arreglos de NumPy
    X = np.array(secuencias_iq)         # Forma original: (N, 2, 128)
    y_mod = np.array(etiquetas_mod)     # Tipo de modulación (String)
    y_snr = np.array(valores_snr)       # Valores de SNR (Int/Float)
    
    print("Dataset cargado con éxito.")
    print(f"Forma de X (Muestras I/Q): {X.shape}")
    print(f"Total de clases encontradas: {len(np.unique(Y_mod))}")
    
    return X, y_mod, y_snr
