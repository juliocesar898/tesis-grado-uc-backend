import numpy as np
import logging
import asyncio
import time
import threading
import random
import pickle
import json

from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep
from src.processing import normalize_iq

class SDRWorker:
    def __init__(self):
        # Variables de control de la API y el Hilo (Thread)
        self.is_running = False
        self.thread = None
        self.main_loop = None  # Referencia al bucle principal de FastAPI para inyectar tareas asíncronas
        self.config = {
            "process_constellation": True,
            "process_psd": True,
            "run_amc_inference": True,
            "min_snr_db": 6.0,          # Filtro de SNR mínimo en dB
            "max_snr_db": None,         # Filtro de SNR máximo (None = Sin límite superior)
            "allowed_classes": None,    # Lista de clases permitidas (None = Todas)
            "class_hold_repeats": 4,    # Cuántas veces seguidas emitir la misma modulación
        }
        
        # Variables de control para el estado de retención (Hold Machine)
        self.current_class_name = None
        self.current_class_counter = 0

        # Variables para cargar el dataset RadioML
        self.dataset_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
        self.radio_ml_data = None
        self.radio_ml_keys = None

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
            # ------------------------------------------------
                
            self.thread = threading.Thread(target=self._loop_procesamiento, daemon=True)
            self.thread.start()
            logging.info("[SDR WORKER] Hilo de captura INICIADO.")

    def detener(self):
        """Apaga el flujo continuo de forma segura"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            logging.info("[SDR WORKER] Hilo de telemetría detenido.")

    def load_dataset_for_simulation(self):
        """Carga el dataset HDF5 en memoria de forma ultra rápida para simular ráfagas reales"""
        if self.radio_ml_data is None:
            logging.info(f"[SDR WORKER] Cargando {self.dataset_path} para simulación en RAM...")
            try:
                import h5py
                with h5py.File(self.dataset_path, 'r') as f:
                    total_muestras = f['X'].shape[0]
                    # Extraer unas 40,000 muestras representativas para simular
                    num_sim_samples = min(40000, total_muestras)
                    
                    # Usamos muestreo secuencial por saltos (strided slicing) - Tarda 1 segundo
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
                self.radio_ml_data = True  # Flag para indicar carga exitosa
                logging.info(f"[SDR WORKER] ¡Dataset 1024-IQ de simulación listo para tiempo real!")
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] No se pudo cargar el dataset HDF5: {e}")

    def _loop_procesamiento(self):
        """El bucle real en segundo plano con control de persistencia de modulación"""
        self.load_dataset_for_simulation()

        while self.is_running:
            try:
                if hasattr(self, 'X_sim'):
                    # 1. Recuperar filtros dinámicos desde el almacenamiento de configuración de la API
                    min_snr_db = float(self.config.get("min_snr_db", 6.0))
                    max_snr_db = self.config.get("max_snr_db")
                    allowed_classes = self.config.get("allowed_classes")
                    class_hold_repeats = int(self.config.get("class_hold_repeats", 4))
                    
                    # Obtener la bolsa de clases sobre las cuales podemos iterar
                    pool_clases = allowed_classes if allowed_classes else self.clases_2018
                    
                    # 2. Controladores de cambio de clase (Máquina de estados de retención)
                    if (self.current_class_name is None or 
                        self.current_class_counter >= class_hold_repeats or 
                        self.current_class_name not in pool_clases):
                        
                        # Elegir un nuevo objetivo al azar entre las permitidas
                        self.current_class_name = random.choice(pool_clases)
                        self.current_class_counter = 0
                        logging.info(
                            f"[SDR WORKER] >>> CAMBIO DE CANAL: Modulación fijada a '{self.current_class_name}' "
                            f"durante {class_hold_repeats} ráfagas consecutivas."
                        )
                    
                    encontrado = False
                    intentos = 0
                    
                    # 3. Buscar una ráfaga que cumpla estrictamente con la clase fijada y el rango de SNR
                    while not encontrado and self.is_running and intentos < 1000:
                        idx = random.randint(0, len(self.X_sim) - 1)
                        snr_generado = float(self.Z_sim[idx][0])
                        
                        # Obtener etiqueta real
                        class_idx = np.argmax(self.Y_sim[idx])
                        mod_name_generada = self.clases_2018[class_idx]
                        
                        # Comprobar si coincide con la modulación objetivo retenida
                        cumple_clase = (mod_name_generada == self.current_class_name)
                        
                        # Comprobar SNR
                        cumple_snr = snr_generado >= min_snr_db
                        if cumple_snr and max_snr_db is not None:
                            cumple_snr = snr_generado <= float(max_snr_db)
                            
                        if cumple_clase and cumple_snr:
                            encontrado = True
                            iq_sample = self.X_sim[idx]
                            y_one_hot = self.Y_sim[idx]
                            self.current_class_counter += 1  # Registrar ráfaga emitida
                            
                        intentos += 1
                    
                    if not encontrado:
                        # Si no hay coincidencias con esa clase y SNR, reseteamos el hold para no asfixiar el ciclo
                        logging.warning(
                            f"[SDR WORKER] No se hallaron muestras de '{self.current_class_name}' "
                            f"con SNR >= {min_snr_db}dB. Reseteando retenedor..."
                        )
                        self.current_class_name = None
                        time.sleep(1.0)
                        continue

                    # 4. Convertir muestras a representación compleja
                    ventana_iq = iq_sample[:, 0] + 1j * iq_sample[:, 1]
                    
                    # 5. Inferencia Inteligente
                    if self.config.get("run_amc_inference"):
                        predicted_class, confidence = model_dep.predict(ventana_iq)
                        
                        logging.info(
                            f"[IA CONSOLA] [{self.current_class_counter}/{class_hold_repeats}] "
                            f"Real: {mod_name_generada:10} (SNR: {snr_generado:4.1f}dB) "
                            f"| Predicho: {predicted_class:10} ({confidence*100:6.2f}%)"
                        )

                        # --- CONSTRUCCIÓN DEL PAYLOAD (SIn pre-serialización JSON) ---
                        payload = {
                            "modulation": mod_name_generada,      
                            "predicted_class": predicted_class,   
                            "confidence": float(confidence),       
                            "snr": snr_generado,                  
                            "data": {
                                "i": ventana_iq.real.tolist(),
                                "q": ventana_iq.imag.tolist()
                            }
                        }
                        
                        # --- ENVIAR AL LOG DE TRANSMISORES WEBSOCKET (THREAD-SAFE) ---
                        if self.main_loop and self.main_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                stream_manager.broadcast(payload), 
                                self.main_loop
                            )
                        else:
                            logging.error("[WS ERROR] No se encontró el Event Loop principal de FastAPI.")
                        
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo en loop de captura: {e}")
            
            time.sleep(1.5)

# Instancia global para ser importada en el router
sdr_worker = SDRWorker()