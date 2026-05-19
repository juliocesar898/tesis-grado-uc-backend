"""
DeepSignal - Automatic Modulation Classification (AMC) Model
Arquitectura de CNN Híbrida para Procesamiento de Señales Digitales I/Q y Espectrogramas.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  # type: ignore
import logging
from pathlib import Path

# Configuración agnóstica al Sistema Operativo para la carpeta de modelos
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_pretrained_model(model_name="amc_model.h5"):
    """
    Carga un modelo preentrenado desde el directorio de modelos.
    """
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        logging.info(f"Cargando modelo existente desde: {model_path}")
        return models.load_model(model_path)
    return None


def save_model(model, model_name="amc_model.h5"):
    """
    Guarda el modelo entrenado en el directorio de modelos, creando
    la carpeta si no existe.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    model.save(model_path)
    logging.info(f"Modelo guardado exitosamente en: {model_path}")


def build_amc_cnn(input_shape_1d=(128, 2), input_shape_2d=(32, 8193, 1), num_classes=11, l2_reg=1e-4):
    """
    Construye una Red Neuronal Convolucional (CNN) Híbrida (Multi-Input) orientada a la
    clasificación de señales base I/Q.

    Args:
        input_shape_1d (tuple): Dimensiones de las muestras de entrada temporales.
        input_shape_2d (tuple): Dimensiones del espectrograma STFT de entrada.
        num_classes (int): Cantidad de modulaciones posibles a clasificar.
        l2_reg (float): Parámetro para la regularización L2 evitando overfitting.

    Returns:
        model (tf.keras.Model): Modelo Keras compilado.
    """
    # =========================================================================
    # RAMA 1: Procesamiento Temporal 1D (Señal cruda IQ)
    # =========================================================================
    input_1d = layers.Input(shape=input_shape_1d, name="iq_1d_input")
    
    x1 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", 
                       kernel_regularizer=regularizers.l2(l2_reg))(input_1d)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)

    x1 = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu", 
                       kernel_regularizer=regularizers.l2(l2_reg))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    
    flat_1d = layers.Flatten()(x1)

    # =========================================================================
    # RAMA 2: Procesamiento Frecuencial 2D (Espectrograma STFT)
    # =========================================================================
    input_2d = layers.Input(shape=input_shape_2d, name="stft_2d_input")
    
    x2 = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", 
                       kernel_regularizer=regularizers.l2(l2_reg))(input_2d)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

    x2 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", 
                       kernel_regularizer=regularizers.l2(l2_reg))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    
    flat_2d = layers.Flatten()(x2)

    # =========================================================================
    # FUSIÓN: Concatenación y Clasificación Final
    # =========================================================================
    merged = layers.concatenate([flat_1d, flat_2d])

    z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(merged)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.3)(z)

    # Salida categórica con Softmax
    output = layers.Dense(num_classes, activation="softmax", name="mod_prediction")(z)

    # Construir el modelo multi-input
    model = models.Model(inputs=[input_1d, input_2d], outputs=output, name="DeepSignal_Hybrid_AMC")

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Prueba de Humo y Arquitectura Base
    print("Inicializando Arquitectura Híbrida DeepSignal AMC...")

    # Ej: Asumiendo trama de 128 muestras y espectrograma de (32, 8193, 1)
    amc_model = build_amc_cnn(
        input_shape_1d=(128, 2), 
        input_shape_2d=(32, 8193, 1), 
        num_classes=11
    )

    amc_model.summary()
    print("✓ Modelo configurado y listo para ser entrenado iterativamente.")