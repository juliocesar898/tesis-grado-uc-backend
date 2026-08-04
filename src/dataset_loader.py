import h5py
import numpy as np


def cargar_radio_ml_2018(path_archivo, num_samples=None):
    """
    Carga el dataset RadioML 2018.01A (archivo HDF5) de forma veloz y eficiente.
    Utiliza muestreo por saltos (strided slicing) en lugar de índices aleatorios dispersos
    para evitar cuellos de botella de lectura en disco.
    """
    print(f"Cargando dataset RadioML 2018 desde {path_archivo}...")
    with h5py.File(path_archivo, "r") as f:
        # Los datasets dentro del HDF5 oficial son 'X', 'Y' y 'Z'
        X_ds = f["X"]
        Y_ds = f["Y"]
        Z_ds = f["Z"]

        total_muestras = f["X"].shape[0]
        print(f"Total de muestras detectadas: {total_muestras}")
        print(f"Forma original de X: {X_ds.shape} (N, 1024, 2)")
        print(f"Forma original de Y: {Y_ds.shape} (N, 24)")

        if num_samples is not None and num_samples < total_muestras:
            # Calcular el paso (stride) dinámico
            step = total_muestras // num_samples
            print(f"Submuestreando {num_samples} registros usando un paso de zancada {step}...")
            
            # El rebanado con paso (step slice) carga bloques enteros de manera secuencial 
            # y es órdenes de magnitud más rápido (toma segundos)
            X = X_ds[::step][:num_samples]
            Y = Y_ds[::step][:num_samples]
            Z = Z_ds[::step][:num_samples]
        else:
            X = X_ds[:]
            Y = Y_ds[:]
            Z = Z_ds[:]

    print("Dataset cargado exitosamente.")
    print(f"Forma final X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}")
    return X, Y, Z