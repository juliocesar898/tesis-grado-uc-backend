import numpy as np
from model import load_pretrained_model, build_amc_cnn, save_model

def get_prediction(iq_tensor, stft_tensor, logger):
    """Maneja el ruteo del modelo, pesos y clasificación de hardware usando CNN Híbrida"""
    logger.info(">> Fase 3: Predicción con CNN")
    model = load_pretrained_model("amc_model.h5")

    if model is None:
        logger.warning("No se halló modelo preentrenado. Inicializando y guardando...")
        model = build_amc_cnn(
            input_shape_1d=(128, 2), input_shape_2d=(17, 9, 1), num_classes=11
        )
        save_model(model, "amc_model.h5")

    # 1. Preparar la ventana temporal (1D)
    window_size = 128
    if len(iq_tensor) >= window_size:
        iq_frame = iq_tensor[:window_size, :]
    else:
        iq_frame = np.pad(iq_tensor, ((0, window_size - len(iq_tensor)), (0, 0)))

    hq_batch = np.expand_dims(iq_frame, axis=0)  # (1, 128, 2)

    # 2. Preparar el Espectrograma (2D)
    stft_batch = np.expand_dims(stft_tensor, axis=0)  # (1, 17, 9, 1)

    # 3. Predicción multi-input (Lista de arrays)
    preds = model.predict([hq_batch, stft_batch], verbose=0)  # <-- Asegúrate de pasar ambas entradas

    predicted_class_idx = np.argmax(preds[0])
    confidence = preds[0][predicted_class_idx]

    logger.info("=== RESULTADO DE CLASIFICACIÓN ===")
    logger.info(
        f"Clase predicha (índice): {predicted_class_idx}, Nivel de confianza: {confidence:.2%}"
    )
    logger.info("----------------------------------")

    return predicted_class_idx, confidence