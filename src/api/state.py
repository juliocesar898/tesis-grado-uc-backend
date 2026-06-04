import threading
import time
import numpy as np
from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep

# import hackrf  # (O la librería de enlace que uses para el SDR)


class SDRWorker:
    def __init__(self):
        self.is_running = False
        self.thread = None
        # Valores por defecto modificables en tiempo de ejecución por la API
        self.config = {
            "process_constellation": True,
            "process_psd": True,
            "run_amc_inference": True,
        }

    def actualizar_configuracion(self, nueva_config: dict):
        self.config.update(nueva_config)

    def iniciar(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._loop_captura, daemon=True)
            self.thread.start()

    def detener(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _loop_captura(self):
        """Bucle infinito corriendo en un hilo secundario para no bloquear FastAPI"""
        print("[SDR WORKER] Hilo de captura encendido y escuchando el HackRF One.")

        while self.is_running:
            # --- SIMULACIÓN O LECTURA REAL DEL HACKRF ---
            # Simulamos un bloque temporal de tamaño (128, 2)
            muestras_iq = np.random.normal(0, 1, (128, 2))

            payload = {}

            # Botón / Filtro 1: Densidad de Potencia Espectral
            if self.config["process_psd"]:
                # Aquí aplicarías tu FFT de src/processing.py
                fft_calculada = np.abs(
                    np.fft.fft(muestras_iq[:, 0] + 1j * muestras_iq[:, 1])
                ).tolist()
                payload["psd"] = fft_calculada

            # Botón / Filtro 2: Constelación (Coordenadas I/Q)
            if self.config["process_constellation"]:
                payload["constellation"] = {
                    "i": muestras_iq[:, 0].tolist(),
                    "q": muestras_iq[:, 1].tolist(),
                }

            # Botón / Filtro 3: Identificación Automática (IA)
            if self.config["run_amc_inference"] and model_dep.model is not None:
                # Ejecuta inferencia ciega en la RAM usando el singleton cargado en lifespan
                mod, conf = model_dep.predict(muestras_iq)
                payload["classification"] = {"modulation": mod, "probability": conf}

            # Si el payload contiene datos y hay clientes conectados en el WebSocket, hacemos Broadcast
            if payload and stream_manager.active_connections:
                import asyncio

                # Forzar el envío asíncrono seguro desde el hilo secundario
                asyncio.run(stream_manager.broadcast(payload))

            # Control de tasa de muestreo para el streaming (ej: 20 actualizaciones por segundo)
            time.sleep(0.05)


# Instancia única global para el Backend
sdr_worker = SDRWorker()
