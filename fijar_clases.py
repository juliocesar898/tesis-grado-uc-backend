import numpy as np

# Las 11 modulaciones oficiales del dataset RML2016.10a en orden alfabético
clases = np.array(
    [
        "8PSK",
        "AM-DSB",
        "AM-SSB",
        "BPSK",
        "CPFSK",
        "GFSK",
        "PAM4",
        "QAM16",
        "QAM64",
        "QPSK",
        "WBFM",
    ]
)

# Guardar directamente en tu carpeta de modelos
np.save("models/clases_modulacion.npy", clases)
print("¡Archivo clases_modulacion.npy creado con éxito!")
