import numpy as np
import logging
import asyncio
import time
import threading
import random
import pickle

from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep
from src.processing import normalize_iq  # <-- IMPORTACIÓN CORREGIDA

class SDRWorker:
    def __init__(self):
        # Variables de control de la API y el Hilo (Thread)
        self.is_running = False
        self.thread = None
        self.config = {
            "process_constellation": True,
            "process_psd": True,
            "run_amc_inference": True,
            "min_snr_db": 6.0,  # <-- Filtro configurable de SNR mínimo (por defecto 6.0 dB)
        }
        
        # Variables para cargar el dataset RadioML
        self.dataset_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
        self.radio_ml_data = None
        self.radio_ml_keys = None

    def actualizar_configuracion(self, nueva_config: dict):
        """Actualiza las banderas de procesamiento bajo demanda desde HTTP"""
        self.config.update(nueva_config)
        logging.info(f"[SDR WORKER] Configuración actualizada: {self.config}")

    def iniciar(self):
        """Lanza el bucle continuo en un hilo secundario protegido"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._loop_procesamiento, daemon=True)
            self.thread.start()
            logging.info("[SDR WORKER] Hilo de telemetría encendido (Modo Dataset Real).")

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
                    # Queremos extraer unas 40,000 muestras representativas para simular
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
        """El bucle real en segundo plano adaptado a muestras de longitud 1024"""
        self.load_dataset_for_simulation()

        while self.is_running:
            try:
                if hasattr(self, 'X_sim'):
                    # Umbral actual desde la configuración del worker
                    min_snr_db = float(self.config.get("min_snr_db", 6.0))
                    
                    encontrado = False
                    intentos = 0
                    
                    # 1. Buscar ráfaga aleatoria que cumpla el umbral de SNR mínimo
                    # Se incluye un límite de intentos para evitar bucles infinitos
                    while not encontrado and self.is_running and intentos < 1000:
                        idx = random.randint(0, len(self.X_sim) - 1)
                        snr_generado = float(self.Z_sim[idx][0])  # SNR en dB del dataset
                        
                        if snr_generado >= min_snr_db:
                            encontrado = True
                            iq_sample = self.X_sim[idx]       # Forma: (1024, 2)
                            y_one_hot = self.Y_sim[idx]       # Forma: (24,)
                        intentos += 1
                    
                    if not encontrado:
                        # Si no se encuentra una muestra adecuada, espera e intenta de nuevo
                        time.sleep(1.5)
                        continue

                    # Extraer etiqueta real
                    class_idx = np.argmax(y_one_hot)
                    mod_name_generada = self.clases_2018[class_idx]
                    
                    # 2. Generar tensor complejo IQ
                    ventana_iq = iq_sample[:, 0] + 1j * iq_sample[:, 1]
                    
                    # 3. Inferencia
                    if self.config.get("run_amc_inference"):
                        predicted_class, confidence = model_dep.predict(ventana_iq)
                        
                        logging.info(
                            f"[IA CONSOLA] Real: {mod_name_generada:10} (SNR: {snr_generado:4.1f}dB) "
                            f"| Predicho: {predicted_class:10} ({confidence*100:6.2f}%)"
                        )
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo en loop de captura: {e}")
            
            time.sleep(1.5)

# Instancia global para ser importada en el router
sdr_worker = SDRWorker()