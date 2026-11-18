import numpy as np
import logging
from functools import lru_cache
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


def estimar_cfo_hz(iq_complex, fs=2000000.0):
    """Estima el desvío de portadora (CFO en Hz) usando elevación M=4 con zero-padding 4x."""
    n = len(iq_complex)
    if n == 0:
        return 0.0
    n_pad = n * 4
    fft_vals = np.fft.fft(iq_complex ** 4, n_pad)
    fft_freqs = np.fft.fftfreq(n_pad, d=1.0)
    idx_pico = int(np.argmax(np.abs(fft_vals)))
    if idx_pico <= 1 or idx_pico >= n_pad - 1:
        return 0.0
    return float((fft_freqs[idx_pico] / 4.0) * fs)


def sincronizar_cfo_rf(iq_complex, fs=1.0, remove_dc=True):
    """
    Estima y compensa el Carrier Frequency Offset (CFO) usando elevación
    de potencia (M=4 para QPSK/QAM) con FFT sobremuestreada (Zero-Padding 4x)
    y alineamiento fino de fase a los ejes canónicos de la constelación.
    No suprime el bin DC de la señal a potencia 4 para evitar inyectar
    rotación falsa cuando el TX y RX comparten oscilador (CFO ≈ 0 Hz).
    """
    n = len(iq_complex)
    if n == 0:
        return iq_complex

    try:
        # 1. Quitar DC solo si se solicita (evita sesgar ráfagas cortas aleatorias)
        iq_work = (iq_complex - np.mean(iq_complex)) if remove_dc else iq_complex

        # 2. Elevar a la 4ª potencia con Zero-Padding 4x para resolución sub-bin fina
        n_pad = n * 4
        iq_power4 = iq_work ** 4
        fft_vals = np.fft.fft(iq_power4, n_pad)
        fft_freqs = np.fft.fftfreq(n_pad, d=1.0)

        # 3. Buscar el pico global incluyendo bin 0 (CFO nulo cuando TX/RX comparten reloj)
        idx_pico = int(np.argmax(np.abs(fft_vals)))
        if idx_pico <= 1 or idx_pico >= n_pad - 1:
            iq_derotado = iq_work
        else:
            frecuencia_rotada = fft_freqs[idx_pico]
            cfo_estimado = frecuencia_rotada / 4.0
            tiempo = np.arange(n)
            iq_derotado = iq_work * np.exp(-1j * 2 * np.pi * cfo_estimado * tiempo)

        # 4. Alineamiento fino de fase (Fine Phase Lock a 45 grados para QPSK)
        phi_cuarto = np.angle(np.mean(iq_derotado ** 4))
        phi_correccion = (phi_cuarto - np.pi) / 4.0
        iq_alineado = iq_derotado * np.exp(-1j * phi_correccion)

        return iq_alineado
    except Exception as e:
        logging.error(f"Fallo en algoritmo Coarse CFO: {e}. Retornando señal original.")
        return iq_complex


@lru_cache(maxsize=16)
def generar_taps_rrc(sps=10, alpha=0.5, span=11):
    """Genera coeficientes normalizados de un filtro Root-Raised-Cosine (RRC)."""
    n_taps = span * sps
    t = np.arange(-n_taps // 2, n_taps // 2 + 1) / float(sps)
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + 4.0 * alpha / np.pi
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-6:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    norm = np.sqrt(np.sum(h ** 2))
    return (h / norm) if norm > 0 else h


def aplicar_filtro_acoplado_rrc(iq_complex, sps=10, alpha=0.5, span=11):
    """
    Aplica el filtro acoplado RRC en recepción (RRC_TX * RRC_RX = Coseno Alzado)
    para eliminar la Interferencia Intersimbólica (ISI) y abrir el diagrama de ojo.
    """
    if len(iq_complex) < sps * 4:
        return iq_complex
    taps = generar_taps_rrc(sps=sps, alpha=alpha, span=span)
    return np.convolve(iq_complex, taps, mode="same")


def extraer_simbolos_optimos(iq_complex, sps=10, osr=4):
    """
    Recuperación fraccional de reloj de símbolo (interpolación sinc 4x) y
    alineamiento fino de fase por orden de simetría (M=2 para BPSK horizontal,
    M=4 para QPSK/16QAM, M=8 para 8PSK) sobre los centros de símbolo limpios.
    """
    if len(iq_complex) < sps * 6:
        return iq_complex

    # 1. Sobremuestreo fraccional (evita desdoblamiento de puntos cuando 1024 no es múltiplo de sps)
    iq_up = signal.resample(iq_complex, len(iq_complex) * osr)
    step = sps * osr
    # Recortar extremos transitorios de convolución/FFT (2 símbolos por lado)
    margin = step * 2
    iq_mid = iq_up[margin:-margin] if len(iq_up) > margin * 3 else iq_up

    mejor_fase = max(range(step), key=lambda k: float(np.mean(np.abs(iq_mid[k::step]) ** 2)))
    iq_sym = iq_mid[mejor_fase::step]
    if len(iq_sym) == 0:
        return iq_complex

    # 2. Alinear fase estática sobre los símbolos limpios según simetría de constelación
    rms = np.sqrt(np.mean(np.abs(iq_sym) ** 2))
    if rms > 0:
        iq_u = iq_sym / rms
        e20 = np.mean(iq_u ** 2)
        e40 = np.mean(iq_u ** 4)
        if np.abs(e20) > 0.55:
            # BPSK / ASK -> alinear estrictamente horizontal en eje -I / +I (0° y 180°)
            phi_corr = np.angle(e20) / 2.0
        elif np.abs(e40) > 0.15:
            # QPSK / 16QAM -> alinear a cuadrantes canónicos (±45°, ±135°)
            phi_corr = (np.angle(e40) - np.pi) / 4.0
        else:
            # 8PSK -> alinear simetría octogonal M=8
            e80 = np.mean(iq_u ** 8)
            phi_corr = np.angle(e80) / 8.0 if np.abs(e80) > 0.05 else 0.0
        iq_sym = iq_sym * np.exp(-1j * phi_corr)

    return iq_sym


def recuperar_reloj_simbolo(iq_complex, sps=8):
    """
    Sincronizador de reloj de símbolo no lineal (Oerder-Meyr).
    Extrae el tono de energía de símbolo en Fs/SPS y ajusta el retardo fraccional
    para alinear el instante de muestreo óptimo (máxima apertura del ojo).
    """
    n = len(iq_complex)
    if n < sps * 4:
        return iq_complex

    try:
        mag_sq = np.abs(iq_complex) ** 2
        fft_mag = np.fft.fft(mag_sq)
        tone_bin = n // sps
        phase_rad = np.angle(fft_mag[tone_bin])
        tau = (-phase_rad / (2.0 * np.pi)) * sps
        shift_samples = int(np.round(tau)) % sps
        if shift_samples != 0:
            return np.roll(iq_complex, -shift_samples)
        return iq_complex
    except Exception as e:
        logging.debug(f"Aviso en sincronizador de reloj: {e}")
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
        padded=True,
        return_onesided=False
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