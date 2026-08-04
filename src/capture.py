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
        }
        
        # Variables para cargar el dataset RadioML
        self.dataset_path = "dataset/RML2016.10a_dict.pkl"
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
        """Carga el dataset en memoria solo la primera vez que se inicia el loop"""
        if self.radio_ml_data is None:
            logging.info(f"[SDR WORKER] Cargando {self.dataset_path} en RAM...")
            try:
                with open(self.dataset_path, 'rb') as f:
                    self.radio_ml_data = pickle.load(f, encoding='latin1')
                    
                    # FILTRO SNR > 0: Guardamos solo las llaves que tienen SNR positivo
                    self.radio_ml_keys = [k for k in self.radio_ml_data.keys() if k[1] > 0]
                    
                logging.info(f"[SDR WORKER] ¡Dataset cargado con éxito! Usando solo señales con SNR > 0.")
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] No se pudo cargar el dataset: {e}")


    def _loop_procesamiento(self):
        """El bucle real que corre en el hilo secundario generando ráfagas"""
        self.load_dataset_for_simulation()

        # Variables para mantener la señal unos segundos
        current_key = None
        last_change_time = 0
        segundos_por_modulacion = 6.0  # Tiempo que durará cada señal antes de cambiar

        while self.is_running:
            try:
                # Verificar que el dataset esté cargado
                if self.radio_ml_data and self.radio_ml_keys:
                    
                    # 1. Cambiar la Modulación y SNR solo si ya pasó el tiempo establecido
                    tiempo_actual = time.time()
                    if current_key is None or (tiempo_actual - last_change_time) > segundos_por_modulacion:
                        current_key = random.choice(self.radio_ml_keys)
                        last_change_time = tiempo_actual
                        logging.info("-" * 50)
                        logging.info(f"[SDR WORKER] Cambiando sintonía a nueva señal...")
                    
                    # Limpiar el string
                    mod_name_generada = current_key[0].decode('utf-8') if isinstance(current_key[0], bytes) else current_key[0]
                    snr_generado = current_key[1]
                    
                    # 2. Extraer una ráfaga aleatoria DENTRO de la modulación actual
                    muestras = self.radio_ml_data[current_key]
                    idx = random.randint(0, len(muestras) - 1)
                    iq_sample = muestras[idx]  # Forma: (2, 128)
                    
                    # 3. Convertir a Array Complejo (I + jQ)
                    ventana_iq = iq_sample[0] + 1j * iq_sample[1]
                    
                    # 4. Inferencia con la IA (Solo si está activa en la config)
                    if self.config.get("run_amc_inference"):
                        predicted_class, confidence = model_dep.predict(ventana_iq)
                        
                        # Formatear bonito para la consola
                        logging.info(f"[IA CONSOLA] Real: {mod_name_generada:8} (SNR: {snr_generado:3}dB) | Predicho: {predicted_class:8} ({confidence*100:6.2f}%)")
                        
                    # (Más adelante aquí enviaremos los datos al WebSocket)
                    
            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo en loop de captura: {e}")
            
            # 5. Esperar 1.5 segundos entre cada escaneo de la misma señal
            time.sleep(1.5)

# Instancia global para ser importada en el router
sdr_worker = SDRWorker()