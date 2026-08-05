import numpy as np
from src.model import load_pretrained_model, build_amc_cnn, save_model
from src.processing import spectrogram_stft

def get_prediction(iq_tensor, stft_tensor, logger):
    """Maneja el ruteo del modelo, pesos y clasificación de hardware usando CNN Híbrida (Dataset 2018)"""
    logger.info(">> Fase 3: Predicción con CNN Híbrida")
    
    # IMPORTANTE: Cambiar el nombre del archivo de pesos si guardaste el nuevo modelo de 2018
    model = load_pretrained_model("clasificador_sdr.h5")

    if model is None:
        logger.warning("No se halló modelo preentrenado. Inicializando y guardando...")
        # Actualizado a 1024 muestras y 24 clases
        model = build_amc_cnn(
            input_shape_1d=(1024, 2), input_shape_2d=(32, 65, 1), num_classes=24
        )
        save_model(model, "clasificador_sdr.h5")

    # 1. Preparar la ventana temporal (1D) para 1024 muestras
    window_size = 1024
    if len(iq_tensor) >= window_size:
        iq_frame = iq_tensor[:window_size, :]
    else:
        iq_frame = np.pad(iq_tensor, ((0, window_size - len(iq_tensor)), (0, 0)))

    hq_batch = np.expand_dims(iq_frame, axis=0)  # (1, 1024, 2)

    # 2. Generar Espectrograma (2D) con la dimensión correcta para 1024
    # Si recibes iq_tensor plano, calculamos la STFT directamente o usamos el stft_tensor ajustado
    if stft_tensor is None or stft_tensor.shape != (32, 65, 1):
        stft_tensor = spectrogram_stft(iq_frame)

    stft_batch = np.expand_dims(stft_tensor, axis=0)  # (1, 32, 65, 1)

    # 3. Inferencia Dual
    preds = model.predict([hq_batch, stft_batch], verbose=0)
    probs = preds[0] if isinstance(preds, list) else preds

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    return predicted_idx, confidence