import numpy as np
from src.model import load_pretrained_model, build_amc_cnn, save_model
from src.processing import spectrogram_stft, calcular_cumulantes_hoc

# Variable de caché global para retener el modelo cargado en memoria RAM
_MODEL_CACHE = None

def get_prediction(iq_tensor, stft_tensor, logger):
    global _MODEL_CACHE
    logger.info(">> Fase 3: Predicción con CNN Híbrida Triple-Input")
    
    # Cargar usando caché global
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_pretrained_model("clasificador_sdr.h5")
        # (Si no existe, construir modelo triple)
        if _MODEL_CACHE is None:
            _MODEL_CACHE = build_amc_cnn(
                input_shape_1d=(1024, 2), 
                input_shape_2d=(32, 65, 1), 
                input_shape_stats=(8,), 
                num_classes=24
            )
            save_model(_MODEL_CACHE, "clasificador_sdr.h5")

    model = _MODEL_CACHE

    # 1. Preparar entrada 1D
    window_size = 1024
    if len(iq_tensor) >= window_size:
        iq_frame = iq_tensor[:window_size, :]
    else:
        iq_frame = np.pad(iq_tensor, ((0, window_size - len(iq_tensor)), (0, 0)))
    hq_batch = np.expand_dims(iq_frame, axis=0)

    # 2. Preparar entrada 2D
    if stft_tensor is None or stft_tensor.shape != (32, 65, 1):
        stft_tensor = spectrogram_stft(iq_frame)
    stft_batch = np.expand_dims(stft_tensor, axis=0)

    # 3. NUEVO: Preparar entrada estadística (Momento de orden superior)
    stats_vector = calcular_cumulantes_hoc(iq_frame)
    stats_batch = np.expand_dims(stats_vector, axis=0)

    # 4. Inferencia Triple
    preds = model.predict([hq_batch, stft_batch, stats_batch], verbose=0)
    probs = preds[0] if isinstance(preds, list) else preds

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    return predicted_idx, confidence