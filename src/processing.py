import numpy as np
import logging
from scipy import signal

# --- AGRUPACIÓN DE CLASES POR FAMILIAS ---
FAMILIAS_MODULACION = {
    # Amplitud Analógica / Digital Básica
    "OOK": "ASK (Amplitud)",
    "4ASK": "ASK (Amplitud)",
    "8ASK": "ASK (Amplitud)",
    
    # Fase Digital (PSK)
    "BPSK": "PSK (Fase)",
    "QPSK": "PSK (Fase)",
    "8PSK": "PSK (Fase)",
    "16PSK": "PSK (Fase)",
    "32PSK": "PSK (Fase)",
    "OQPSK": "PSK (Fase)",
    
    # Amplitud-Fase Híbrida (APSK)
    "16APSK": "APSK (Híbrido)",
    "32APSK": "APSK (Híbrido)",
    "64APSK": "APSK (Híbrido)",
    "128APSK": "APSK (Híbrido)",
    
    # Amplitud en Cuadratura de Alta Densidad (QAM)
    "16QAM": "QAM (Alta Densidad)",
    "32QAM": "QAM (Alta Densidad)",
    "64QAM": "QAM (Alta Densidad)",
    "128QAM": "QAM (Alta Densidad)",
    "256QAM": "QAM (Alta Densidad)",
    
    # Analógicas Continentales (AM)
    "AM-SSB-WC": "AM (Modulación de Amplitud)",
    "AM-SSB-SC": "AM (Modulación de Amplitud)",
    "AM-DSB-WC": "AM (Modulación de Amplitud)",
    "AM-DSB-SC": "AM (Modulación de Amplitud)",
    
    # Continuas / Modulación de Frecuencia
    "FM": "FM / FSK (Frecuencia)",
    "GMSK": "FM / FSK (Frecuencia)"
}


def obtener_familia_modulacion(clase_predicha: str) -> str:
    """
    Mapea cualquier clase específica de RadioML 2018.01A a su categoría general core.
    """
    return FAMILIAS_MODULACION.get(clase_predicha, "Desconocida / Ruido")


def sincronizar_cfo_rf(iq_complex, fs=1.0):
    """
    Estima y compensa el Carrier Frequency Offset (CFO) usando elevación
    de potencia (M=4 para QPSK/QAM) y análisis espectral FFT.
    Evita que la rotación deforme las constelaciones en círculos concéntricos.
    """
    n = len(iq_complex)
    if n == 0:
        return iq_complex

    try:
        # 1. Elevar la señal a la 4ª potencia para eliminar modulaciones simétricas de fase
        iq_power4 = iq_complex ** 4

        # 2. Calcular la FFT del residuo para encontrar el pico espectral de rotación
        fft_vals = np.fft.fft(iq_power4)
        fft_freqs = np.fft.fftfreq(n, d=1.0 / fs)

        # Obtenemos el índice de la frecuencia con mayor energía (omitimos la componente DC)
        idx_pico = np.argmax(np.abs(fft_vals)[1:]) + 1
        frecuencia_estimada_rotada = fft_freqs[idx_pico]

        # 3. La frecuencia real de CFO es el pico espectral obtenido dividido por M (4)
        cfo_estimado = frecuencia_estimada_rotada / 4.0

        # 4. Generar el vector corrector exponencial e^(-j * 2 * pi * cfo * t)
        tiempo = np.arange(n)
        corrector_fase = np.exp(-1j * 2 * np.pi * cfo_estimado * tiempo)

        # 5. Aplicar la corrección por producto elemento a elemento
        iq_sincronizado = iq_complex * corrector_fase
        return iq_sincronizado
    except Exception as e:
        logging.error(f"Fallo en algoritmo Coarse CFO: {e}. Retornando señal original.")
        return iq_complex


def normalize_iq(iq_data, sincronizar=False):
    """
    Normaliza los datos IQ crudos (arreglos complejos) a su máximo valor absoluto.
    Opcionalmente aplica sincronización de portadora (CFO) antes del procesamiento de amplitud.
    Retorna un tensor [Tiempo, Canales] adecuado de tamaño (1024, 2) o (128, 2) para la CNN.
    """
    logging.info("Normalizando datos IQ crudos.")
    iq_data = np.asarray(iq_data, dtype=complex)

    # 1. Sincronización opcional de fase
    if sincronizar:
        logging.info("Sincronizando desajuste de frecuencia de ráfaga (CFO)...")
        iq_data = sincronizar_cfo_rf(iq_data)

    # 2. Normalización de amplitud por el valor absoluto máximo
    rms = np.sqrt(np.mean(np.square(np.abs(iq_data))))
    if rms > 0:
        iq_data = iq_data / rms

    iq_tensor = np.column_stack((np.real(iq_data), np.imag(iq_data)))

    # Validación de dimensiones: [Longitud_Muestras, 2 Canales (I/Q)]
    expected_shape = (len(iq_data), 2)
    if iq_tensor.shape != expected_shape:
        logging.error(
            f"Error de dimensiones en normalize_iq. Actual: {iq_tensor.shape}, Esperado: {expected_shape}"
        )
        raise ValueError("El tensor IQ normalizado no tiene las dimensiones correctas.")

    return iq_tensor


def spectrogram_stft(iq_data, fs=1.0, sincronizar=False):
    """
    Transforma los datos IQ en espectrogramas usando STFT.
    Apropiado para arquitecturas CNN 2D. 
    Soporta ráfagas de 128 muestras (32, 9, 1) y 1024 muestras (32, 65, 1).
    """
    iq_data = np.asarray(iq_data)

    # 1. Combinar componentes I/Q si se recibe un tensor temporal estructurado (N, 2)
    if iq_data.ndim == 2 and iq_data.shape[1] == 2:
        iq_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
    else:
        iq_complex = iq_data.astype(complex)

    # Sincronización opcional del dominio temporal previo al análisis espectral
    if sincronizar:
        iq_complex = sincronizar_cfo_rf(iq_complex, fs=fs)

    # 2. Configurar scipy.signal.stft para obtener ráfagas de 128 o 1024 muestras
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

    # 4. Expandir dimensiones agregando el canal al final -> (32, 9, 1) o (32, 65, 1)
    espectrograma_tensor = np.expand_dims(espectrograma, axis=-1)

    # 5. Validación dinámica de dimensiones (Retrocompatible)
    expected_shapes = [(32, 9, 1), (32, 65, 1)]
    if espectrograma_tensor.shape not in expected_shapes:
        logging.error(
            f"Error de dimensiones en spectrogram_stft. Actual: {espectrograma_tensor.shape}, Esperado: {expected_shapes}"
        )
        raise ValueError(
            f"El espectrograma resultante de la STFT {espectrograma_tensor.shape} no coincide con ningún tamaño esperado."
        )

    return espectrograma_tensor

def calcular_cumulantes_hoc(iq_data):
    """
    Calcula los cumulantes y momentos estadísticos de orden superior (HOC)
    del canal complejo I/Q de forma matemáticamente rigurosa.
    Retorna un array de tamaño (8,) con: [Media_Abs, Var, C20, C40, C42, C60, C63, C80]
    """
    # 1. Asegurar formato complex de una sola dimensión
    if iq_data.ndim == 2 and iq_data.shape[1] == 2:
        complex_signal = iq_data[:, 0] + 1j * iq_data[:, 1]
    else:
        complex_signal = np.asarray(iq_data, dtype=complex)
        
    n = len(complex_signal)
    if n == 0:
        return np.zeros(8, dtype=float)
        
    # Remover componente de offset DC
    sig = complex_signal - np.mean(complex_signal)
    
    # 2. Momentos Estadísticos Complejos Básicos Epq = E[s^p * (s*)^q]
    e_20 = np.mean(sig**2)
    e_21 = np.mean(np.abs(sig)**2)   # Varianza / Energía promedio
    
    e_40 = np.mean(sig**4)
    e_41 = np.mean((sig**3) * np.conj(sig))
    e_42 = np.mean(np.abs(sig)**4)
    
    e_60 = np.mean(sig**6)
    e_63 = np.mean(np.abs(sig)**6)
    
    e_80 = np.mean(sig**8)
    
    # 3. Cumulantes de Orden Superior (HOC) Complejos
    c_20 = e_20
    c_21 = e_21  
    
    c_40 = e_40 - 3 * (e_20**2)
    c_42 = e_42 - np.abs(e_20)**2 - 2 * (e_21**2)
    
    c_60 = e_60 - 15 * e_40 * e_20 + 30 * (e_20**3)
    
    # --- CORRECCIÓN DE FASE PARA C63 ---
    # Usamos np.conj(e_20) para simular E02 y np.abs(e_20)**2 para simular |E20|^2
    c_63 = (
        e_63 
        - 9 * e_41 * np.conj(e_20) 
        - 6 * e_42 * e_21 
        + 18 * (np.abs(e_20)**2) * e_21 
        + 12 * (e_21**3)
    )
    
    # c_80 es analíticamente correcto ya que no contiene variables conjugadas (s*)
    c_80 = e_80 - 28 * e_60 * e_20 - 35 * (e_40**2) + 420 * e_40 * (e_20**2) - 630 * (e_20**4)
    
    # 4. Formar vector de entrada para Keras (Extrayendo magnitudes reales)
    stats_vector = np.array([
        np.mean(np.abs(sig)),             # Amplitud Promedio (L1)
        float(np.real(c_21)),             # Energía Promedio (Real)
        float(np.abs(c_20)),              # |C20|
        float(np.abs(c_40)),              # |C40|
        float(np.real(c_42)),             # C42 (Siempre real por definición simétrica)
        float(np.abs(c_60)),              # |C60|
        float(np.real(c_63)),             # C63 (Siempre real por definición simétrica)
        float(np.abs(c_80))               # |C80|
    ], dtype=float)
    
    # Prevenir inconsistencias numéricas
    stats_vector = np.nan_to_num(stats_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return stats_vector