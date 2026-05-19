import logging
import numpy as np
import time
from capture import capture_iq_data
from processing import normalize_iq, spectrogram_stft
from model import load_pretrained_model, build_amc_cnn, save_model
from prediction import get_prediction
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

    stft_tensor = spectrogram_stft(raw_iq)
    return iq_tensor, stft_tensor


def execute_prediction(iq_tensor, stft_tensor, logger):
    """Maneja la lógica de predicción usando el modelo CNN híbrido"""
    get_prediction(iq_tensor, stft_tensor, logger)


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

        # Procesamos la señal para obtener ambos tensores: el temporal (1D) y el espectrograma (2D)
        iq_tensor, stft_tensor = execute_processing(raw_iq, logger)

        logger.info(
            "Tensores preparados para la predicción. Shape IQ: "
            f"{iq_tensor.shape}, Shape STFT: {stft_tensor.shape}"
        )

        # Pasamos ambos tensores al modelo --> TO DO
        # execute_prediction(iq_tensor, stft_tensor, logger)

        # Visualización final (opcional, pero útil para debugging y validación visual)
        execute_visualization(raw_iq, logger)

    except Exception as e:
        logger.error(f"Falla crítica abortando el pipeline: {e}")


if __name__ == "__main__":
    main()
