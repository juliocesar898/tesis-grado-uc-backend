import numpy as np
from pathlib import Path

# Las 24 clases oficiales de RadioML 2018.01A en su orden exacto de índice
clases = np.array([
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
])

# Crear la carpeta de modelos si no existe de manera agnóstica a SO
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

# Guardar directamente en tu carpeta de modelos
classes_path = models_dir / "clases_modulacion.npy"
np.save(classes_path, clases)
print(f"¡Archivo {classes_path} creado con éxito con 24 clases de RadioML 2018!")