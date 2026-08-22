"""
DeepSignal - Automatic Modulation Classification (AMC) Model
Arquitectura de CNN Híbrida Multi-Input con Inyección de Características Estadísticas (Cumulantes)
Optimizado con Mecanismos de Dual Global Pooling para reducción de Bias de Parámetros.
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


def build_amc_cnn(
    input_shape_1d=(1024, 2), 
    input_shape_2d=(32, 65, 1), 
    input_shape_stats=(8,), 
    num_classes=24, 
    l2_reg=1e-4
):
    """
    Construye una Red Neuronal Convolucional (CNN) Híbrida de entrada triple (Multi-Input) 
    optimizada mediante reducción de dimensiones vía Global Average/Max Pooling para
    evitar overfitting y asfixia por regularización L2.
    """
    # =========================================================================
    # RAMA 1: Procesamiento Temporal 1D (Señal cruda IQ)
    # =========================================================================
    input_1d = layers.Input(shape=input_shape_1d, name="iq_1d_input")

    x1 = layers.Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(input_1d)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)

    x1 = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)

    # --- REEMPLAZO FLATTEN POR DUAL GLOBAL POOLING 1D ---
    # Extrae el promedio de energía espacial y picos del vector temporal,
    # logrando representaciones robustas y ultraligeras.
    avg_pool_1d = layers.GlobalAveragePooling1D()(x1)
    max_pool_1d = layers.GlobalMaxPooling1D()(x1)
    concat_1d = layers.concatenate([avg_pool_1d, max_pool_1d]) # Dim: 256

    proj_1d = layers.Dense(
        128, 
        activation="relu", 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="proj_1d"
    )(concat_1d)
    proj_1d = layers.BatchNormalization()(proj_1d)
    proj_1d = layers.Dropout(0.2)(proj_1d)

    # =========================================================================
    # RAMA 2: Procesamiento Frecuencial 2D (Espectrograma STFT)
    # =========================================================================
    input_2d = layers.Input(shape=input_shape_2d, name="stft_2d_input")

    x2 = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(input_2d)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

    x2 = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

    # --- REEMPLAZO FLATTEN POR DUAL GLOBAL POOLING 2D ---
    avg_pool_2d = layers.GlobalAveragePooling2D()(x2)
    max_pool_2d = layers.GlobalMaxPooling2D()(x2)
    concat_2d = layers.concatenate([avg_pool_2d, max_pool_2d]) # Dim: 128

    proj_2d = layers.Dense(
        128, 
        activation="relu", 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="proj_2d"
    )(concat_2d)
    proj_2d = layers.BatchNormalization()(proj_2d)
    proj_2d = layers.Dropout(0.2)(proj_2d)

    # =========================================================================
    # RAMA 3: Inyección de Características Estadísticas (Cumulantes HOC)
    # =========================================================================
    input_stats = layers.Input(shape=input_shape_stats, name="stats_input")
    
    x3 = layers.Dense(
        32, 
        activation="relu", 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="proj_stats"
    )(input_stats)
    x3 = layers.BatchNormalization()(x3)

    # =========================================================================
    # FUSIÓN: Concatenación Triple Altamente Balanceada
    # =========================================================================
    # Fusión matemática equitativa y ligera frente a Overfitting:
    # 128 (Temporal) + 128 (Espectral) + 32 (Cumulantes) = 288 neuronas balanceadas.
    merged = layers.concatenate([proj_1d, proj_2d, x3], name="balanced_fusion")

    z = layers.Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
    )(merged)
    z = layers.Dropout(0.4)(z)

    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.3)(z)

    # Salida categórica con Softmax
    output = layers.Dense(num_classes, activation="softmax", name="mod_prediction")(z)

    # Construir el modelo triple-input
    model = models.Model(
        inputs=[input_1d, input_2d, input_stats], outputs=output, name="DeepSignal_Hybrid_AMC"
    )

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Prueba de Humo de la Red
    print("Inicializando Arquitectura de Entrada Triple DeepSignal AMC con Global Pooling...")

    amc_model = build_amc_cnn(
        input_shape_1d=(1024, 2), 
        input_shape_2d=(32, 65, 1), 
        input_shape_stats=(8,), 
        num_classes=24
    )

    amc_model.summary()
    print("✓ Modelo configurado, parametrizado y listo para re-entrenamiento.")