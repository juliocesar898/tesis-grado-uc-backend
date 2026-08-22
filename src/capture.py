import numpy as np
import logging
import asyncio
import time
import threading
import random

from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep

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
            "pluto_freq_hz": 915000000,  # Frecuencia portadora central de sintonía
            "pluto_sample_rate": 1000000, # Tasa de muestreo (1 MSPS)
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

    def actualizar_configuracion(self, nueva_config: dict):
        """Actualiza las banderas de procesamiento bajo demanda y reinicia el estado de retención"""
        self.config.update(nueva_config)
        # Al actualizar configuración desde la API, forzamos un cambio inmediato de modulación
        self.current_class_name = None
        self.current_class_counter = 0
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
                if ventana_iq is not None and self.config.get("run_amc_inference"):
                    predicted_class, confidence = model_dep.predict(ventana_iq)
                    
                    logging.info(
                        f"[IA - {modo_activo.upper()}] "
                        f"Real: {mod_name_generada:11} "
                        f"| Predicho: {predicted_class:11} ({confidence*100:6.2f}%)"
                    )

                    # --- 6. PROPAGAR PAYLOAD DSP VÍA WEBSOCKET ---
                    payload = {
                        "modulation": mod_name_generada,      
                        "predicted_class": predicted_class,   
                        "confidence": float(confidence),       
                        "snr": float(snr_generado),                  
                        "data": {
                            "i": ventana_iq.real.tolist(),
                            "q": ventana_iq.imag.tolist()
                        }
                    }
                    
                    if self.main_loop and self.main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            stream_manager.broadcast(payload), 
                            self.main_loop
                        )
                        
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo inesperado en el núcleo de captura: {e}")
            
            # --- CORRECCIÓN 2: RETARDO DINÁMICO CONTRA OVERFLOW DE BÚFER ---
            if modo_activo == "pluto":
                # Como self.sdr_device.rx() bloquea el hilo por ~1 milisegundo (1 MSPS),
                # dejamos un respiro mínimo de CPU para vaciar el flujo rápido y robusto.
                time.sleep(0.01)
            else:
                # Retardo pausado para no saturar al cliente Web / Frontend con ráfagas sintéticas
                time.sleep(1.2)

# Instancia global para ser inyectada e importada en el backend
sdr_worker = SDRWorker()