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
from src.processing import (
    spectrogram_stft, 
    sincronizar_cfo_rf, 
    normalize_iq, 
    calcular_cumulantes_hoc
)
from src.model import build_amc_cnn, save_model

# Las 24 clases oficiales completas del dataset RadioML 2018.01A en su respectivo orden de índice
# Sincronizadas rigurosamente con capture.py para evitar desalineación de etiquetas
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


def prepare_dataset(dataset_path: str, num_samples: int = 400000):
    """
    Carga el dataset RadioML 2018.01A (HDF5) y construye las entradas triples
    sincronizadas (1D IQ + 2D STFT + 1D Cumulantes HOC).
    """
    logging.info(">> Cargando el dataset RadioML 2018.01A...")
    X_raw, Y_raw, Z_raw = cargar_radio_ml_2018(dataset_path, num_samples=num_samples)

    num_samples_loaded = len(X_raw)
    logging.info(f">> Procesando {num_samples_loaded} ráfagas con Sincronización CFO y HOC...")

    # Forzar arrays de salida optimizados
    X_1d = np.zeros_like(X_raw)
    X_2d_list = []
    X_stats_list = []  # Lista para consolidar los descriptores estadísticos

    logging.info("Generando espectrogramas (STFT) y cumulantes (HOC) configurados para AMC de triple entrada.")
    for i in range(num_samples_loaded):
        # 1. Reconstruir señal compleja original de 1024 muestras
        iq_burst_complex = X_raw[i, :, 0] + 1j * X_raw[i, :, 1]
        
        # 2. Sincronizar CFO para detener rotaciones
        iq_complex_sinc = sincronizar_cfo_rf(iq_burst_complex)
        
        # 3. Guardar señal IQ 1D limpia y normalizada por varianza
        X_1d[i] = normalize_iq(iq_complex_sinc, sincronizar=False)
        
        # 4. Generar STFT alineada con la señal ya sincronizada
        stft = spectrogram_stft(X_1d[i], sincronizar=False)
        X_2d_list.append(stft)

        # 5. Inyectar firma estadística de cumulantes de orden superior (HOC)
        stats = calcular_cumulantes_hoc(X_1d[i])
        X_stats_list.append(stats)

    X_2d = np.array(X_2d_list)
    if X_2d.ndim == 3:
        X_2d = np.expand_dims(X_2d, axis=-1)

    X_stats = np.array(X_stats_list)

    classes = CLASES_2018
    return X_1d, X_2d, X_stats, Y_raw, classes


def run_training(dataset_path: str, epochs: int = 40, batch_size: int = 128):
    setup_logging()
    
    if not os.path.exists(dataset_path):
        logging.error(f"No se encontró el archivo del dataset en: {dataset_path}")
        logging.error("Asegúrate de colocar 'GOLD_XYZ_OSC.0001_1024.hdf5' en la ruta especificada.")
        return

    # 1. Preparación de datos de entrada triple
    X_1d, X_2d, X_stats, Y, classes = prepare_dataset(dataset_path, num_samples=400000)

    # 2. Dividir en conjuntos de Entrenamiento y Validación (80/20) estratificado
    X1_train, X1_val, X2_train, X2_val, Xs_train, Xs_val, Y_train, Y_val = train_test_split(
        X_1d, X_2d, X_stats, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
    )

    # -- OPTIMIZACIÓN AGRESIVA DE MEMORIA RAM --
    # Eliminar referencias a variables pesadas que ya fueron duplicadas por train_test_split
    del X_1d
    del X_2d
    del X_stats
    del Y
    gc.collect()  # Invocar inmediatamente el recolector de basura

    # 3. Construir Modelo de Triple Entrada con Shapes Alineados:
    # 1D Temporal (1024, 2) | 2D Espectrograma (32, 65, 1) | 1D Cumulantes (8,) | 24 Clases
    num_classes = len(classes)
    model = build_amc_cnn(
        input_shape_1d=(1024, 2),
        input_shape_2d=(32, 65, 1),
        input_shape_stats=(8,),
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

    # 5. Iniciar Entrenamiento pasando las tres ramas de características como una lista
    logging.info(">> Iniciando el entrenamiento del modelo híbrido triple-input de DeepSignal para 1024-IQ...")
    history = model.fit(
        x=[X1_train, X2_train, Xs_train],
        y=Y_train,
        validation_data=([X1_val, X2_val, Xs_val], Y_val),
        epochs=epochs,
        batch_size=batch_size,     # Lote reducido para regular temperatura de hardware en Windows
        callbacks=callbacks,
        verbose=1
    )

    # 6. Guardar Modelo Final y Clases
    save_model(model, "clasificador_sdr.h5")
    
    # Guardar el array de clases oficiales para usarlo en la API de inferencia
    classes_path = Path(__file__).resolve().parent.parent / "models" / "clases_modulacion.npy"
    np.save(classes_path, np.array(classes))

    logging.info(">> ¡Entrenamiento con características estadísticas del modelo completado!")
    logging.info(">> Modelo guardado en 'models/clasificador_sdr.h5'")
    logging.info(f">> Clases guardadas en '{classes_path}'")


if __name__ == "__main__":
    # Ruta predeterminada al HDF5 de RadioML 2018.01A
    DATASET_FILE = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    run_training(DATASET_FILE, epochs=40, batch_size=128)