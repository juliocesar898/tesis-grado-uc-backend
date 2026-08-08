import os
import sys
import numpy as np
import logging
import gc  # Liberador de RAM explícito
from pathlib import Path

# -- PROTECCIÓN FÍSICA PARA CPU (EVITAR PANTALLA AZUL / BSOD) --
# Limitar hilos a nivel de bibliotecas de álgebra lineal antes de importar TF
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Asegurar que el path reconozca el módulo 'src' desde la raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importación segura de TensorFlow configurando límites de subprocesos
import tensorflow as tf
total_cores = os.cpu_count() or 4
usar_cores = max(1, total_cores - 2) # Dejar siempre 2 núcleos libres para el sistema OS
tf.config.threading.set_intra_op_parallelism_threads(usar_cores)
tf.config.threading.set_inter_op_parallelism_threads(2)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

from src.dataset_loader import cargar_radio_ml_2018
from src.processing import spectrogram_stft
from src.model import build_amc_cnn, save_model

# Las 24 clases oficiales del dataset RadioML 2018.01A en su respectivo orden de índice
CLASES_2018 = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prepare_dataset(dataset_path: str, num_samples: int = 250000):
    """
    Carga el dataset RadioML 2018.01A (HDF5) y construye las entradas duales
    sincronizadas (1D IQ + 2D STFT).
    """
    logging.info(">> Cargando el dataset RadioML 2018.01A...")
    X_raw, Y_raw, Z_raw = cargar_radio_ml_2018(dataset_path, num_samples=num_samples)

    num_samples_loaded = len(X_raw)
    logging.info(f">> Procesando {num_samples_loaded} ráfagas con Sincronización CFO...")

    # Forzar arrays de salida optimizados
    X_1d = np.zeros_like(X_raw)
    X_2d_list = []

    # Importar sinc localmente
    from src.processing import sincronizar_cfo_rf, normalize_iq

    logging.info("Generando espectrograma usando STFT configurado para AMC.")
    for i in range(num_samples_loaded):
        # 1. Reconstruir señal compleja original de 1024 muestras
        iq_burst_complex = X_raw[i, :, 0] + 1j * X_raw[i, :, 1]
        
        # 2. Sincronizar CFO para detener rotaciones
        iq_complex_sinc = sincronizar_cfo_rf(iq_burst_complex)
        
        # 3. Guardar señal IQ 1D limpia y normalizada
        X_1d[i] = normalize_iq(iq_complex_sinc, sincronizar=False)
        
        # 4. Generar STFT alineada con la señal ya sincronizada
        stft = spectrogram_stft(X_1d[i], sincronizar=False)
        X_2d_list.append(stft)

    X_2d = np.array(X_2d_list)

    if X_2d.ndim == 3:
        X_2d = np.expand_dims(X_2d, axis=-1)

    classes = CLASES_2018
    return X_1d, X_2d, Y_raw, classes

def run_training(dataset_path: str, epochs: int = 40, batch_size: int = 128):
    setup_logging()
    
    if not os.path.exists(dataset_path):
        logging.error(f"No se encontró el archivo del dataset en: {dataset_path}")
        logging.error("Asegúrate de colocar 'GOLD_XYZ_OSC.0001_1024.hdf5' en la ruta especificada.")
        return

    # 1. Preparación de datos
    X_1d, X_2d, Y, classes = prepare_dataset(dataset_path, num_samples=250000)

    # 2. Dividir en conjuntos de Entrenamiento y Validación (80/20)
    X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(
        X_1d, X_2d, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
    )

    # -- OPTIMIZACIÓN DE MEMORIA RAM --
    # Eliminar referencias a variables pesadas que ya fueron duplicadas por `train_test_split`
    del X_1d
    del X_2d
    del Y
    gc.collect()  # Invocar inmediatamente el recolector de basura

    # 3. Construir Modelo con Shapes Alineados (1024, 2) y (32, 65, 1) para 24 Clases
    num_classes = len(classes)
    model = build_amc_cnn(
        input_shape_1d=(1024, 2),
        input_shape_2d=(32, 65, 1),
        num_classes=num_classes
    )
    model.summary()

    # 4. Configurar Callbacks para Entrenamiento Robusto
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Iniciar Entrenamiento
    logging.info(">> Iniciando el entrenamiento del modelo híbrido DeepSignal para 1024-IQ...")
    history = model.fit(
        x=[X1_train, X2_train],
        y=Y_train,
        validation_data=([X1_val, X2_val], Y_val),
        epochs=epochs,
        batch_size=batch_size,     # Default modificado a 128 para mitigar estrés en CPU
        callbacks=callbacks,
        verbose=1
    )

    # 6. Guardar Modelo Final y Clases
    # Guardamos como 'clasificador_sdr.h5' para que sobreescriba el modelo usado por el Server FastAPI
    save_model(model, "clasificador_sdr.h5")
    
    # Guardar el array de clases oficiales para usarlo en la API de inferencia
    classes_path = Path(__file__).resolve().parent.parent / "models" / "clases_modulacion.npy"
    np.save(classes_path, np.array(classes))

    logging.info(">> ¡Entrenamiento completado!")
    logging.info(">> Modelo guardado en 'models/clasificador_sdr.h5'")
    logging.info(f">> Clases guardadas en '{classes_path}'")


if __name__ == "__main__":
    # Ruta predeterminada al HDF5 de RadioML 2018.01A
    DATASET_FILE = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    # Lote reducido de 256 a 128 para estabilizar la temperatura del procesador y RAM del sistema
    run_training(DATASET_FILE, epochs=40, batch_size=128)