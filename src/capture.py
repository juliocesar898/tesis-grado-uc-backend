import numpy as np
import logging

# Evaluamos la disponibilidad del hardware a nivel global
try:
    from rtlsdr import RtlSdr

    SDR_AVAILABLE = True
except ImportError:
    SDR_AVAILABLE = False
    logging.warning(
        "Librería 'rtlsdr' no encontrada. La captura se ejecutará en [MODO MOCK]."
    )


def capture_iq_data(num_samples=1024):
    """
    Intenta capturar datos del hardware (RTL-SDR / HackRF).
    Si falla, cae en modo Mock/Simulación generando señales sintéticas.
    """
    if SDR_AVAILABLE:
        try:
            logging.info("Intentando conectar con hardware SDR (RTL-SDR)...")
            sdr = RtlSdr()
            sdr.sample_rate = 2.048e6
            sdr.center_freq = 100e6
            sdr.gain = "auto"
            samples = sdr.read_samples(num_samples)
            sdr.close()
            logging.info("Datos capturados exitosamente desde el SDR estacionario.")
            return samples
        except Exception as e:
            logging.warning(f"Error accediendo al hardware SDR: {e}. Pasando a Mock.")

    logging.info("Generando datos IQ sintéticos con ruido (AWGN)...")

    # Simulación de Modulación PSK/QAM + Ruido Gaussiano
    t = np.linspace(0, 10, num_samples)
    f = 2.0  # f simulada
    rng = np.random.default_rng(seed=42)

    i_channel = np.cos(2 * np.pi * f * t) + rng.normal(0, 0.2, num_samples)
    q_channel = np.sin(2 * np.pi * f * t + np.pi / 4) + rng.normal(0, 0.2, num_samples)

    # Composición compleja explícita predecible (np.complex64 por eficiencia)
    iq_synthetic = np.array(i_channel + 1j * q_channel, dtype=np.complex64)
    return iq_synthetic
