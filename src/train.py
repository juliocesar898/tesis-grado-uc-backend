import os
import sys
import numpy as np
import logging
from pathlib import Path

# Asegurar que el path reconozca el módulo 'src' desde la raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

from src.dataset_loader import cargar_radio_ml
from src.processing import spectrogram_stft
from src.model import build_amc_cnn, save_model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prepare_dataset(dataset_path: str):
    """
    Carga el dataset RadioML 2016.10A y construye las entradas duales (1D IQ + 2D STFT).
    """
    logging.info(">> Cargando el dataset RadioML 2016.10A...")
    X_raw, y_mod_raw, y_snr_raw = cargar_radio_ml(dataset_path)

    num_samples = len(X_raw)
    logging.info(f">> Procesando {num_samples} ráfagas para la arquitectura dual...")

    # 1. Transformar X_1D: (N, 2, 128) -> (N, 128, 2)
    X_1d = np.transpose(X_raw, (0, 2, 1))

    # 2. Generar X_2D (STFT Spectrograms): (N, 32, 9, 1)
    logging.info(">> Calculando espectrogramas STFT para la rama 2D...")
    X_2d_list = []
    
    for i in range(num_samples):
        iq_burst = X_1d[i]
        stft = spectrogram_stft(iq_burst)
        X_2d_list.append(stft)

    X_2d = np.array(X_2d_list)

    # Asegurar la 4ta dimensión del canal (N, 32, 9, 1) si viene como (N, 32, 9)
    if X_2d.ndim == 3:
        X_2d = np.expand_dims(X_2d, axis=-1)

    # 3. Codificar Etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_mod_raw)
    y_categorical = to_categorical(y_encoded)
    classes = list(encoder.classes_)

    logging.info(f">> Clases de modulación ({len(classes)}): {classes}")
    logging.info(f">> Forma final X_1D: {X_1d.shape}")
    logging.info(f">> Forma final X_2D: {X_2d.shape}")
    logging.info(f">> Forma final Y: {y_categorical.shape}")

    return X_1d, X_2d, y_categorical, classes


def run_training(dataset_path: str, epochs: int = 40, batch_size: int = 128):
    setup_logging()
    
    if not os.path.exists(dataset_path):
        logging.error(f"No se encontró el archivo del dataset en: {dataset_path}")
        logging.error("Asegúrate de colocar 'RML2016.10a_dict.pkl' en la ruta especificada.")
        return

    # 1. Preparación de datos
    X_1d, X_2d, Y, classes = prepare_dataset(dataset_path)

    # 2. Dividir en conjuntos de Entrenamiento y Validación (80/20)
    X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(
        X_1d, X_2d, Y, test_size=0.2, random_state=42, stratify=np.argmax(Y, axis=1)
    )

    # 3. Construir Modelo con Shapes Alineados (128, 2) y (32, 9, 1)
    num_classes = len(classes)
    model = build_amc_cnn(
        input_shape_1d=(128, 2),
        input_shape_2d=(32, 9, 1),  # <-- ✅ CORREGIDO A 32 BINS DE FRECUENCIA
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
    logging.info(">> Iniciando el entrenamiento del modelo híbrido DeepSignal...")
    history = model.fit(
        x=[X1_train, X2_train],
        y=Y_train,
        validation_data=([X1_val, X2_val], Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Guardar Modelo Final y Clases
    save_model(model, "clasificador_sdr.h5")
    
    # Guardar el array de clases para usarlo en la API de inferencia
    classes_path = Path(__file__).resolve().parent.parent / "models" / "clases_modulacion.npy"
    np.save(classes_path, np.array(classes))

    logging.info(">> ¡Entrenamiento completado!")
    logging.info(">> Modelo guardado en 'models/clasificador_sdr.h5'")
    logging.info(f">> Clases guardadas en '{classes_path}'")


if __name__ == "__main__":
    # Ruta predeterminada al pickle de RadioML 2016.10A
    DATASET_FILE = "dataset/RML2016.10a_dict.pkl"
    run_training(DATASET_FILE, epochs=40, batch_size=256)