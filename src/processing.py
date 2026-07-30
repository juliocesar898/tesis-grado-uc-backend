import numpy as np
import logging
from scipy import signal


def normalize_iq(iq_data):
    """
    Normaliza los datos IQ crudos (arreglos complejos) a su máximo valor absoluto.
    Retorna un tensor [Tiempo, Canales] adecuado para la CNN.
    """
    logging.info("Normalizando datos IQ crudos.")
    iq_data = np.asarray(iq_data, dtype=complex)

    max_val = np.max(np.abs(iq_data))
    if max_val > 0:
        iq_data = iq_data / max_val

    iq_tensor = np.column_stack((np.real(iq_data), np.imag(iq_data)))

    # Validación de dimensiones: [Longitud_Muestras, 2 Canales (I/Q)]
    expected_shape = (len(iq_data), 2)
    if iq_tensor.shape != expected_shape:
        logging.error(
            f"Error de dimensiones en normalize_iq. Actual: {iq_tensor.shape}, Esperado: {expected_shape}"
        )
        raise ValueError("El tensor IQ normalizado no tiene las dimensiones correctas.")

    return iq_tensor


def spectrogram_stft(iq_data, fs=1.0):
    """
    Transforma los datos IQ en espectrogramas usando STFT.
    Apropiado para arquitecturas CNN 2D. Garantiza salida de shape (17, 9, 1).
    """
    logging.info("Generando espectrograma usando STFT configurado para AMC.")
    iq_data = np.asarray(iq_data)

    # 1. Combinar componentes I/Q si se recibe un tensor temporal (128, 2)
    if iq_data.ndim == 2 and iq_data.shape[1] == 2:
        iq_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
    else:
        iq_complex = iq_data.astype(complex)

    # 2. Configurar scipy.signal.stft para obtener dimensiones (17, 9) con ráfagas de 128 muestras
    f, t, Zxx = signal.stft(
        iq_complex,
        fs=fs,
        nperseg=32,
        noverlap=16,
        nfft=32,
        boundary="even",
        padded=True
    )
    espectrograma = np.abs(Zxx)

    # 3. Aplicar normalización Min-Max [0, 1]
    spec_min = np.min(espectrograma)
    spec_max = np.max(espectrograma)
    if spec_max > spec_min:
        espectrograma = (espectrograma - spec_min) / (spec_max - spec_min)
    else:
        espectrograma = np.zeros_like(espectrograma)

    # 4. Expandir dimensiones agregando el canal al final -> (17, 9, 1)
    espectrograma_tensor = np.expand_dims(espectrograma, axis=-1)

    # Validación estricta de dimensiones
    expected_shape = (32, 9, 1)
    if espectrograma_tensor.shape != expected_shape:
        logging.error(
            f"Error de dimensiones en spectrogram_stft. Actual: {espectrograma_tensor.shape}, Esperado: {expected_shape}"
        )
        raise ValueError(
            f"El espectrograma resultante de la STFT {espectrograma_tensor.shape} no coincide con el esperado {expected_shape}."
        )

    return espectrograma_tensor