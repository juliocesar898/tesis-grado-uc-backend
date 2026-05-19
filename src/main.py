import logging
import numpy as np
import time
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


def execute_capture(logger):
    """Maneja la lógica de ingesta de datos RF"""
    logger.info(">> Fase 1: Captura de Señal")
    # Capturamos 1024 muestras (emulando un buffer real),
    # pero la red solo usará las primeras 128.
    return capture_iq_data(num_samples=1024)


def execute_processing(raw_iq, logger):
    """Maneja la normalización y transformaciones de los tensores"""
    logger.info(">> Fase 2: Preprocesamiento")
    iq_tensor = normalize_iq(raw_iq)
    logger.info(f"Tensor 1D I/Q normalizado con shape: {iq_tensor.shape}")

    stft_tensor = spectrogram_stft(raw_iq)
    logger.info(f"Tensor 2D Espectrograma generado con shape: {stft_tensor.shape}")
    return iq_tensor


def execute_prediction(iq_tensor, logger):
    """Maneja el ruteo del modelo, pesos y clasificación de hardware"""
    logger.info(">> Fase 3: Predicción con CNN")
    model = load_pretrained_model("amc_model.h5")

    if model is None:
        logger.warning("No se halló modelo preentrenado. Inicializando y guardando...")
        model = build_amc_cnn(input_shape=(128, 2), num_classes=11)
        save_model(model, "amc_model.h5")

    # Extraemos una ventana de observación estructurada (Frame) de 128 muestras
    # para emparejar con el input_shape original de la red neuronal.
    window_size = 128
    if len(iq_tensor) >= window_size:
        iq_frame = iq_tensor[:window_size, :]
    else:
        # Padding en caso anómalo de buffers demasiado cortos
        iq_frame = np.pad(iq_tensor, ((0, window_size - len(iq_tensor)), (0, 0)))

    # Pasamos 'iq_frame' al modelo en lugar del tensor completo
    # [Batch, Tiempo, Canales] -> quedará como (1, 128, 2)
    hq_batch = np.expand_dims(iq_frame, axis=0)
    preds = model.predict(hq_batch, verbose=0)

    predicted_class_idx = np.argmax(preds[0])
    confidence = preds[0][predicted_class_idx]

    logger.info("=== RESULTADO DE CLASIFICACIÓN ===")
    logger.info(
        f"Clase predicha (índice): {predicted_class_idx}, Nivel de confianza: {confidence:.2%}"
    )
    logger.info("----------------------------------")


def execute_visualization(raw_iq, logger):
    """Dispara los componentes de interfaz gráfica al finalizar el DSP"""
    logger.info("Generando plot grafico de la señal...")
    time.sleep(3)
    logger.info(
        "Generando gráfica post-predicción... (El script terminará al cerrar la ventana)"
    )
    visualize_iq(raw_iq)


def main():
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("--- Iniciando Pipeline de Clasificación de Modulaciones AMC ---")

    try:
        # Pipeline Lineal y Limpio
        raw_iq = execute_capture(logger)
        iq_tensor = execute_processing(raw_iq, logger)
        execute_prediction(iq_tensor, logger)
        execute_visualization(raw_iq, logger)

    except Exception as e:
        logger.error(f"Falla crítica abortando el pipeline: {e}")


if __name__ == "__main__":
    main()
