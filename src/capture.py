import numpy as np
import logging
import asyncio
import time
import threading
import random

from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep
from src.processing import (
    obtener_familia_modulacion,
    sincronizar_cfo_rf,
    recuperar_reloj_simbolo,
    calcular_cumulantes_hoc
)

class SDRWorker:
    def __init__(self):
        # Variables de control de la API y el Hilo (Thread)
        self.is_running = False
        self.thread = None
        self.main_loop = None  # Referencia al bucle principal de FastAPI para inyectar tareas asíncronas
        self.sdr_device = None  # Instancia física del ADALM-Pluto cargada bajo demanda
        
        self.config = {
            "mode": "simulation",        # Interruptor de procedencia de datos: 'simulation' o 'pluto'
            "pluto_ip": "ip:192.168.2.1", # URI IP del Pluto+ por USB RNDIS o Ethernet
            "pluto_freq_hz": 2480000000, # Frecuencia portadora central de sintonía (2.480 GHz - BLE CH 39)
            "pluto_sample_rate": 2000000, # Tasa de muestreo (2 MSPS)
            "pluto_gain_db": 30.0,       # Amplitud de ganancia de entrada Rx en dB
            
            "process_constellation": True,
            "process_psd": True,
            "run_amc_inference": True,
            "min_snr_db": 6.0,          # Filtro de SNR mínimo en dB para simulación
            "max_snr_db": None,         # Filtro de SNR máximo (None = Sin límite superior)
            "allowed_classes": None,    # Lista de clases permitidas (None = Todas)
            "class_hold_repeats": 4,    # Cuántas veces seguidas emitir la misma modulación en simulación
        }
        
        # Variables de control para el estado de retención (Hold Machine en modo simulación)
        self.current_class_name = None
        self.current_class_counter = 0

        # Variables para cargar el dataset RadioML
        self.dataset_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
        self.radio_ml_data = None

        # Suavizador temporal de probabilidades (Filtro Moving Average de ráfagas)
        from collections import deque
        self.probs_history = deque(maxlen=5)

        # Suavizado de video exponencial IIR para el analizador de espectro (PSD)
        self.psd_smooth = None

        # Retenedor temporal (Latch & Hold) de ráfagas para lectura humana estable
        self.latched_burst = None
        self.latched_burst_time = 0.0
        self.BURST_HOLD_SECONDS = 1.8

    def actualizar_configuracion(self, nueva_config: dict):
        """Actualiza las banderas de procesamiento bajo demanda y reinicia el estado de retención"""
        filtrada = {k: v for k, v in nueva_config.items() if v is not None}
        modo_antiguo = self.config.get("mode")
        self.config.update(filtrada)
        
        # Si se altera el modo o parámetros RF de Pluto, limpiar la instancia sdr_device para reinicialización limpia
        if modo_antiguo != self.config.get("mode") or any(k in filtrada for k in ["pluto_freq_hz", "pluto_sample_rate", "pluto_gain_db"]):
            if self.sdr_device is not None:
                try:
                    del self.sdr_device
                except Exception:
                    pass
                self.sdr_device = None
                
        # Al actualizar configuración desde la API, forzamos un cambio inmediato de modulación
        self.current_class_name = None
        self.current_class_counter = 0
        if hasattr(self, 'probs_history'):
            self.probs_history.clear()
        logging.info(f"[SDR WORKER] Configuración actualizada: {self.config}")

    def iniciar(self):
        """Lanza el bucle continuo en un hilo secundario protegido"""
        if not self.is_running:
            self.is_running = True
            
            # --- CAPTURAR EL EVENT LOOP DE FASTAPI ---
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.main_loop = None
                
            self.thread = threading.Thread(target=self._loop_procesamiento, daemon=True)
            self.thread.start()
            logging.info("[SDR WORKER] Hilo de captura INICIADO.")

    def detener(self):
        """Apaga el flujo continuo de forma segura"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.sdr_device is not None:
                try:
                    # Liberar el contexto IIO para no bloquear puertos TCP
                    del self.sdr_device
                    self.sdr_device = None
                    logging.info("[SDR WORKER] Contexto IIO liberado exitosamente.")
                except Exception as e:
                    logging.error(f"[SDR WORKER] Error destruyendo contexto IIO: {e}")
            logging.info("[SDR WORKER] Hilo de telemetría detenido.")

    def load_dataset_for_simulation(self):
        """Carga el dataset HDF5 en memoria de forma ultra rápida para simular ráfagas reales"""
        if self.radio_ml_data is None:
            logging.info(f"[SDR WORKER] Cargando {self.dataset_path} para simulación en RAM...")
            try:
                import h5py
                with h5py.File(self.dataset_path, 'r') as f:
                    total_muestras = f['X'].shape[0]
                    num_sim_samples = min(40000, total_muestras)
                    
                    step = total_muestras // num_sim_samples
                    self.X_sim = f['X'][::step][:num_sim_samples]
                    self.Y_sim = f['Y'][::step][:num_sim_samples]
                    self.Z_sim = f['Z'][::step][:num_sim_samples]
                    
                self.clases_2018 = [
                    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                    'FM', 'GMSK', 'OQPSK'
                ]
                self.radio_ml_data = True
                logging.info(f"[SDR WORKER] ¡Dataset 1024-IQ de simulación listo para tiempo real!")
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] No se pudo cargar el dataset HDF5: {e}")

    def _conectar_pluto_hardware(self) -> bool:
        """Inicializa pyadi-iio de forma controlada sin interrumpir el arranque del backend"""
        if self.sdr_device is not None:
            return True
        try:
            import adi
            ip_val = self.config.get("pluto_ip", "ip:192.168.2.1")
            fc = int(self.config.get("pluto_freq_hz", 915000000))
            fs = int(self.config.get("pluto_sample_rate", 1000000))
            gain = float(self.config.get("pluto_gain_db", 30.0))

            logging.info(f"[SDR WORKER] Conectado a Pluto+ SDR [{ip_val}] - Freq: {fc/1e6} MHz - Rate: {fs/1e6} MSPS")
            sdr = adi.Pluto(ip_val)
            sdr.rx_lo = fc
            sdr.sample_rate = fs
            sdr.rx_rf_bandwidth = fs
            sdr.gain_control_mode_chan0 = "manual"
            sdr.rx_hardwaregain_chan0 = int(gain)
            sdr.rx_enabled_channels = [0]
            sdr.rx_buffer_size = 1024  # Ffija un tamaño estricto de bloque de 1024 muestras complejas
            
            self.sdr_device = sdr
            return True
        except ImportError:
            logging.error("[SDR WORKER ERROR] La biblioteca 'pyadi-iio' no se encuentra instalada en el entorno virtual.")
            return False
        except Exception as e:
            logging.error(f"[SDR WORKER ERROR] Falla de enlace físico con Pluto+ SDR en {self.config.get('pluto_ip')}: {e}")
            return False

    def _loop_procesamiento(self):
        """Bucle en segundo plano con lógica selectiva agnóstica a la procedencia de datos"""
        self.load_dataset_for_simulation()

        while self.is_running:
            try:
                modo_activo = self.config.get("mode", "simulation")
                ventana_iq = None
                mod_name_generada = "En Vivo SDR"
                snr_generado = 15.0  # Asumimos una SNR nominal limpia en capturas físicas controladas

                if modo_activo == "pluto":
                    # --- CAPTURA MEDIANTE HW ADALM-PLUTO ---
                    if self._conectar_pluto_hardware() and self.sdr_device is not None:
                        try:
                            # rx() es una llamada bloqueante a nivel driver por puerto iio tcp
                            muestras = self.sdr_device.rx()
                            
                            # --- CORRECCIÓN 1: VALIDACIÓN MATEMÁTICA DEL FRAME IQ (SIN NP.PAD) ---
                            if muestras is None or len(muestras) < 1024:
                                logging.warning(
                                    f"[SDR WORKER] Frame incompleto o corrupto (recibidas {len(muestras) if muestras is not None else 0} muestras). "
                                    "Descartando ráfaga para proteger espectrograma STFT..."
                                )
                                # Evitamos sleep largo; vaciamos el búfer y pasamos al siguiente ciclo directamente
                                if modo_activo == "pluto":
                                    time.sleep(0.001)
                                continue
                            
                            ventana_iq = muestras[:1024]
                        except Exception as e:
                            logging.error(f"[SDR WORKER] Falla al leer datos complejos del búfer Pluto: {e}")
                            # Si se desabastece el enlace con la antena, forzamos reinvención del contexto
                            self.sdr_device = None
                            time.sleep(0.5)
                            continue
                    else:
                        # Dejar un lapso corto de descanso en caso de fallo crítico de comunicación física con el Pluto
                        time.sleep(1.0)
                        continue
                else:
                    # --- CAPTURA DE EMULACIÓN MEDIANTE DATASET (SIMULACIÓN ORIGINAL) ---
                    if hasattr(self, 'X_sim'):
                        min_snr_db = float(self.config.get("min_snr_db", 6.0))
                        max_snr_db = self.config.get("max_snr_db")
                        allowed_classes = self.config.get("allowed_classes")
                        class_hold_repeats = int(self.config.get("class_hold_repeats", 4))
                        
                        pool_clases = allowed_classes if allowed_classes else self.clases_2018
                        
                        if (self.current_class_name is None or 
                            self.current_class_counter >= class_hold_repeats or 
                            self.current_class_name not in pool_clases):
                            
                            self.current_class_name = random.choice(pool_clases)
                            self.current_class_counter = 0
                            logging.info(
                                f"[SDR WORKER] >>> CAMBIO DE CANAL: Modulación fijada a '{self.current_class_name}' "
                                f"durante {class_hold_repeats} ráfagas consecutivas."
                            )
                        
                        encontrado = False
                        intentos = 0
                        
                        while not encontrado and self.is_running and intentos < 1000:
                            idx = random.randint(0, len(self.X_sim) - 1)
                            snr_generado = float(self.Z_sim[idx][0])
                            
                            class_idx = np.argmax(self.Y_sim[idx])
                            mod_name_generada = self.clases_2018[class_idx]
                            
                            cumple_clase = (mod_name_generada == self.current_class_name)
                            cumple_snr = snr_generado >= min_snr_db
                            if cumple_snr and max_snr_db is not None:
                                cumple_snr = snr_generado <= float(max_snr_db)
                                
                            if cumple_clase and cumple_snr:
                                encontrado = True
                                iq_sample = self.X_sim[idx]
                                self.current_class_counter += 1
                                
                            intentos += 1
                        
                        if not encontrado:
                            logging.warning(
                                f"[SDR WORKER] No se hallaron muestras de '{self.current_class_name}' "
                                f"con SNR >= {min_snr_db}dB. Reseteando retenedor..."
                            )
                            self.current_class_name = None
                            time.sleep(0.5)
                            continue

                        # Convertir del dataset a complejo nativo de 1024 muestras
                        ventana_iq = iq_sample[:, 0] + 1j * iq_sample[:, 1]

                # --- 5. INFERENCIA Y CLASIFICACIÓN DE ALTA VELOCIDAD ---
                if ventana_iq is not None:
                    # 1. Eliminar offset DC analógico de los mezcladores I/Q del SDR
                    ventana_iq = ventana_iq - np.mean(ventana_iq)

                    # 2. Sincronización y Normalización para la constelación web
                    iq_sinc = sincronizar_cfo_rf(ventana_iq)
                    iq_sinc = recuperar_reloj_simbolo(iq_sinc, sps=8)
                    rms_val = np.sqrt(np.mean(np.abs(iq_sinc)**2))
                    if rms_val > 0:
                        iq_norm = iq_sinc / rms_val
                    else:
                        iq_norm = iq_sinc

                    predicted_class = "Sin Inferencia"
                    confidence = 0.0
                    top_candidates = []
                    
                    if self.config.get("run_amc_inference", True):
                        inst_class, inst_conf, raw_probs = model_dep.predict(ventana_iq)
                        self.probs_history.append(raw_probs)

                        # Promedio temporal de las últimas N ráfagas (Filtro Moving Average)
                        avg_probs = np.mean(self.probs_history, axis=0)
                        best_idx = int(np.argmax(avg_probs))
                        confidence = float(avg_probs[best_idx])
                        
                        if model_dep.classes is not None and len(model_dep.classes) > best_idx:
                            predicted_class = str(model_dep.classes[best_idx])
                        else:
                            predicted_class = inst_class

                        # Obtener top 3 clases candidatas
                        top_indices = np.argsort(avg_probs)[::-1][:3]
                        for tidx in top_indices:
                            c_name = str(model_dep.classes[tidx]) if model_dep.classes is not None and len(model_dep.classes) > tidx else f"Clase_{tidx}"
                            top_candidates.append({
                                "modulation": c_name,
                                "probability": round(float(avg_probs[tidx]), 3)
                            })

                        logging.info(
                            f"[IA - {modo_activo.upper()}] "
                            f"Inst: {inst_class:9} | Suavizado: {predicted_class:9} ({confidence*100:5.1f}%)"
                        )

                    # --- CÁLCULO DE PSD (DENSIDAD ESPECTRAL) CON MEDICIÓN DE PICO ---
                    psd_web = []
                    peak_offset_hz = 0
                    peak_freq_hz = int(self.config.get("pluto_freq_hz", 2480000000))
                    peak_power_db = 0.0

                    if self.config.get("process_psd", True):
                        try:
                            N = len(ventana_iq)
                            win = np.blackman(N)
                            fft_res = np.fft.fftshift(np.fft.fft(ventana_iq * win))
                            mag_sq = np.abs(fft_res) ** 2
                            psd_raw = 10 * np.log10(mag_sq + 1e-12)

                            # Submuestrear a 128 bins uniformemente
                            step_psd = max(1, N // 128)
                            psd_sub = psd_raw[::step_psd][:128]

                            # 1. Filtro de Video IIR Exponencial para estabilizar el espectrograma (Anti-titileo)
                            if self.psd_smooth is None or len(self.psd_smooth) != len(psd_sub):
                                self.psd_smooth = psd_sub.copy()
                            else:
                                alpha_v = 0.35
                                self.psd_smooth = alpha_v * psd_sub + (1.0 - alpha_v) * self.psd_smooth

                            psd_work = self.psd_smooth

                            # 2. Análisis de energía y detección de ráfagas (Squelch Dinámico)
                            p_median = float(np.median(psd_work))
                            idx_peak = int(np.argmax(psd_work))
                            peak_power_db = float(psd_work[idx_peak])
                            snr_est_db = round(peak_power_db - p_median, 1)

                            # Ráfaga presente si supera el piso de ruido por >= 6.5 dB (o en simulación sintética)
                            instant_burst = bool(snr_est_db >= 6.5) or (modo_activo == "simulation")

                            fs_cur = float(self.config.get("pluto_sample_rate", 2000000))
                            freq_step = fs_cur / 128.0
                            fc_cur = int(self.config.get("pluto_freq_hz", 2480000000))

                            if instant_burst:
                                peak_offset_hz = int((idx_peak - 64) * freq_step)
                                peak_freq_hz = fc_cur + peak_offset_hz
                                threshold_db = peak_power_db - 10.0
                                bins_above = np.where(psd_work >= threshold_db)[0]
                                bw_est_hz = int(len(bins_above) * freq_step) if len(bins_above) > 0 else int(freq_step)
                            else:
                                peak_offset_hz = 0
                                peak_freq_hz = fc_cur
                                bw_est_hz = 0

                            # Escala relativa normalizada para frontend con cota de piso calibrada
                            p_min = float(np.min(psd_work))
                            p_max = float(np.max(psd_work))
                            dyn_range = p_max - p_min
                            effective_span = max(30.0, dyn_range)
                            psd_norm = ((psd_work - p_min) / effective_span) * 95.0 + 5.0
                            psd_web = np.clip(psd_norm, 5.0, 100.0).tolist()
                        except Exception as e_psd:
                            logging.error(f"[SDR WORKER] Error calculando PSD: {e_psd}")
                            psd_web = [0.0] * 128
                            bw_est_hz = 0
                            instant_burst = False
                            p_median = -60.0
                            snr_est_db = 0.0
                            idx_peak = 64
                            fc_cur = int(self.config.get("pluto_freq_hz", 2480000000))
                            peak_freq_hz = fc_cur
                            peak_offset_hz = 0
                            peak_power_db = -60.0

                    # --- 5.2 EXTRACCIÓN DE PARÁMETROS CIENTÍFICOS Y TELEMETRÍA SIGINT ---
                    try:
                        pwr = np.abs(ventana_iq) ** 2
                        mean_pwr = float(np.mean(pwr)) + 1e-12
                        max_pwr = float(np.max(pwr))
                        papr_db = round(float(10.0 * np.log10(max_pwr / mean_pwr)), 2)

                        hoc_stats = calcular_cumulantes_hoc(ventana_iq)
                        c20_val = round(float(hoc_stats[2]), 4)
                        c40_val = round(float(hoc_stats[3]), 4)
                        rms_power_db = round(float(10.0 * np.log10(mean_pwr)), 1)
                    except Exception as e_telem:
                        logging.debug(f"[SDR WORKER] Error calculando telemetría: {e_telem}")
                        papr_db = 0.0
                        c20_val = 0.0
                        c40_val = 0.0
                        rms_power_db = 0.0

                    # --- 5.3 LATCH & HOLD: RETENCIÓN TEMPORAL DE RÁFAGA (LECTURA HUMANA ESTABLE) ---
                    now = time.time()
                    if instant_burst:
                        self.latched_burst = {
                            "papr_db": papr_db,
                            "c40": c40_val,
                            "c20": c20_val,
                            "freq_offset_hz": peak_offset_hz,
                            "occupied_bw_hz": bw_est_hz,
                            "rms_power_db": rms_power_db,
                            "snr_est_db": snr_est_db,
                            "peak_freq_hz": peak_freq_hz,
                            "peak_power_db": round(peak_power_db, 1),
                            "peak_offset_hz": peak_offset_hz,
                            "peak_bin": idx_peak,
                            "predicted_class": predicted_class,
                            "confidence": float(confidence),
                            "family": obtener_familia_modulacion(predicted_class),
                            "top_candidates": top_candidates
                        }
                        self.latched_burst_time = now

                    is_holding = (now - self.latched_burst_time) < self.BURST_HOLD_SECONDS
                    has_active_signal = instant_burst or is_holding

                    if has_active_signal and self.latched_burst is not None:
                        display_telem = {
                            "papr_db": self.latched_burst["papr_db"],
                            "c40": self.latched_burst["c40"],
                            "c20": self.latched_burst["c20"],
                            "freq_offset_hz": self.latched_burst["freq_offset_hz"],
                            "occupied_bw_hz": self.latched_burst["occupied_bw_hz"],
                            "rms_power_db": self.latched_burst["rms_power_db"],
                            "snr_est_db": self.latched_burst["snr_est_db"],
                            "burst_detected": True,
                            "is_holding": is_holding and not instant_burst
                        }
                        display_spectrum = {
                            "has_signal": True,
                            "peak_offset_hz": self.latched_burst["peak_offset_hz"],
                            "peak_freq_hz": self.latched_burst["peak_freq_hz"],
                            "peak_power_db": self.latched_burst["peak_power_db"],
                            "bw_hz": int(self.config.get("pluto_sample_rate", 2000000)),
                            "occupied_bw_hz": self.latched_burst["occupied_bw_hz"],
                            "peak_bin": self.latched_burst.get("peak_bin", idx_peak)
                        }
                    else:
                        display_telem = {
                            "papr_db": None,
                            "c40": None,
                            "c20": None,
                            "freq_offset_hz": 0,
                            "occupied_bw_hz": None,
                            "rms_power_db": rms_power_db,
                            "snr_est_db": snr_est_db,
                            "burst_detected": False,
                            "is_holding": False
                        }
                        display_spectrum = {
                            "has_signal": False,
                            "peak_offset_hz": 0,
                            "peak_freq_hz": fc_cur,
                            "peak_power_db": round(p_median, 1),
                            "bw_hz": int(self.config.get("pluto_sample_rate", 2000000)),
                            "occupied_bw_hz": 0,
                            "peak_bin": 64
                        }

                    # Puntos de constelación NORMALIZADOS para renderizado en cuadrantes
                    i_pts = iq_norm.real[:256].tolist() if self.config.get("process_constellation", True) else []
                    q_pts = iq_norm.imag[:256].tolist() if self.config.get("process_constellation", True) else []

                    # --- 6. PROPAGAR PAYLOAD DSP VÍA WEBSOCKET ---
                    payload = {
                        "modulation": mod_name_generada,      
                        "predicted_class": self.latched_burst["predicted_class"] if (has_active_signal and self.latched_burst) else predicted_class,   
                        "confidence": self.latched_burst["confidence"] if (has_active_signal and self.latched_burst) else float(confidence),       
                        "snr": self.latched_burst["snr_est_db"] if (has_active_signal and self.latched_burst) else float(snr_generado),
                        "mode": modo_activo,
                        "center_freq_hz": int(self.config.get("pluto_freq_hz", 2480000000)),
                        "sample_rate": int(self.config.get("pluto_sample_rate", 2000000)),
                        "spectrum": display_spectrum,
                        "classification": {
                            "modulation": self.latched_burst["predicted_class"] if (has_active_signal and self.latched_burst) else predicted_class,
                            "probability": self.latched_burst["confidence"] if (has_active_signal and self.latched_burst) else float(confidence),
                            "family": self.latched_burst["family"] if (has_active_signal and self.latched_burst) else obtener_familia_modulacion(predicted_class),
                            "top_candidates": self.latched_burst["top_candidates"] if (has_active_signal and self.latched_burst) else top_candidates
                        },
                        "telemetry": display_telem,
                        "psd": psd_web,
                        "constellation": {
                            "i": i_pts,
                            "q": q_pts
                        },
                        "data": {
                            "i": i_pts,
                            "q": q_pts
                        }
                    }
                    
                    if self.main_loop and self.main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            stream_manager.broadcast(payload), 
                            self.main_loop
                        )
                        
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo inesperado en el núcleo de captura: {e}")
            
            # --- CORRECCIÓN 2: CONTROL DE CADENCIA Y FLUIDEZ ESTABLE (~16-18 FPS) ---
            if modo_activo == "pluto":
                infer_on = bool(self.config.get("run_amc_inference", False))
                # 55 ms de pausa da una cadencia de ~16 FPS: el Waterfall avanza relajado y legible (~9 seg de historia)
                time.sleep(0.025 if infer_on else 0.055)
            else:
                # Retardo pausado para no saturar al cliente Web / Frontend con ráfagas sintéticas
                time.sleep(1.2)

# Instancia global para ser inyectada e importada en el backend
sdr_worker = SDRWorker()