import numpy as np
from model import load_pretrained_model, build_amc_cnn, save_model

def get_prediction(iq_tensor, stft_tensor, logger):
    """Maneja el ruteo del modelo, pesos y clasificación de hardware usando CNN Híbrida"""
    logger.info(">> Fase 3: Predicción con CNN")
    model = load_pretrained_model("amc_model.h5")

    if model is None:
        logger.warning("No se halló modelo preentrenado. Inicializando y guardando...")
        # NOTA: Debes actualizar `build_amc_cnn` en model.py para que acepte estos dos parámetros
        # y construya una arquitectura multi-input (híbrida).
        model = build_amc_cnn(
            input_shape_1d=(128, 2), input_shape_2d=stft_tensor.shape, num_classes=11
        )
        save_model(model, "amc_model.h5")

    # 1. Preparar la ventana temporal (1D)
    # Extraemos una ventana de observación estructurada (Frame) de 128 muestras
    window_size = 128
    if len(iq_tensor) >= window_size:
        iq_frame = iq_tensor[:window_size, :]
    else:
        # Padding en caso anómalo de buffers demasiado cortos
        iq_frame = np.pad(iq_tensor, ((0, window_size - len(iq_tensor)), (0, 0)))

    # Convertimos a Batch -> (1, 128, 2)
    hq_batch = np.expand_dims(iq_frame, axis=0)

    # 2. Preparar el Espectrograma (2D)
    # Expandimos dimensiones para coincidir con la entrada Conv2D -> (1, Frecuencias, Tiempos, 1)
    stft_batch = np.expand_dims(stft_tensor, axis=0)

    # 3. Predicción multi-input (Lista de arrays)
    preds = model.predict([hq_batch, stft_batch], verbose=0)

    predicted_class_idx = np.argmax(preds[0])
    confidence = preds[0][predicted_class_idx]

    logger.info("=== RESULTADO DE CLASIFICACIÓN ===")
    logger.info(
        f"Clase predicha (índice): {predicted_class_idx}, Nivel de confianza: {confidence:.2%}"
    )
    logger.info("----------------------------------")