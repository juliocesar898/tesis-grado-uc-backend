import logging
import numpy as np
from capture import capture_iq_data
from processing import normalize_iq, spectrogram_stft
from model import load_pretrained_model, build_amc_cnn, save_model
from plot_signal import visualize_iq


def setup_logger():
    """Configura el logger estructurado con marcas de tiempo"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def main():
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("--- Iniciando Pipeline de Clasificación de Modulaciones AMC ---")

    # 1. Captura de Datos (Hardware o Mock)
    try:
        logger.info(">> Fase 1: Captura de Señal")
        raw_iq = capture_iq_data(num_samples=128)

        # --- LLAMADA A LA GRÁFICA ---
        logger.info("Generando gráficas...")
        visualize_iq(raw_iq)

    except Exception as e:
        logger.error(f"Falla crítica en captura de señal: {e}")
        return

    # 2. Preprocesamiento de la Señal
    try:
        logger.info(">> Fase 2: Preprocesamiento")
        iq_tensor = normalize_iq(raw_iq)
        logger.info(f"Tensor 1D I/Q normalizado con shape: {iq_tensor.shape}")

        # Transformación en Espectrograma por si usamos CNNs 2D alternas
        stft_tensor = spectrogram_stft(raw_iq)
        logger.info(f"Tensor 2D Espectrograma generado con shape: {stft_tensor.shape}")
    except Exception as e:
        logger.error(f"Falla crítica en preprocesamiento: {e}")
        return

    # 3. Predicción / Modelo
    try:
        logger.info(">> Fase 3: Predicción con CNN")
        model = load_pretrained_model("amc_model.h5")
        if model is None:
            logger.warning(
                "No se halló modelo preentrenado. Inicializando nuevo modelo y guardando..."
            )
            model = build_amc_cnn(input_shape=(128, 2), num_classes=11)
            save_model(model, "amc_model.h5")

        # Agregamos la dimensión de 'Batch' antes de predecir [Batch, Tiempo, Canales]
        # Ej: tensor shape (1, 128, 2)
        hq_batch = np.expand_dims(iq_tensor, axis=0)

        preds = model.predict(hq_batch, verbose=0)
        predicted_class_idx = np.argmax(preds[0])
        confidence = preds[0][predicted_class_idx]

        logger.info("=== RESULTADO DE CLASIFICACIÓN ===")
        logger.info(
            f"Clase predicha (índice): {predicted_class_idx}, Nivel de confianza: {confidence:.2%}"
        )
        logger.info("----------------------------------")
    except Exception as e:
        logger.error(f"Falla durante la compilación/predicción con el modelo: {e}")


if __name__ == "__main__":
    main()
