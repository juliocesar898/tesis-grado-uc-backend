import numpy as np
import logging
import asyncio
import time
import threading  # <-- Esencial para que no se congele la API al escanear
from src.api.websocket_manager import stream_manager
from src.api.dependencies import model_dep

class SDRWorker:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.config = {
            "process_constellation": True,
            "process_psd": True,
            "run_amc_inference": True,
        }

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
            logging.info("[SDR WORKER] Hilo de telemetría RF encendido exitosamente.")

    def detener(self):
        """Apaga el flujo continuo de forma segura"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
            logging.info("[SDR WORKER] Hilo de telemetría RF destruido limpiamente.")

    def _loop_procesamiento(self):
        """Bucle continuo que genera señales sintéticas con modulaciones aleatorias"""
        logging.info(
            "[SDR WORKER] Loop encendido. Generando ráfagas RF sintéticas variables..."
        )

        # Lista de tipos de modulación que vamos a simular matemáticamente
        tipos_modulacion = ["BPSK", "QPSK", "8PSK", "QAM16"]
        rng = np.random.default_rng()

        while self.is_running:
            try:
                # 1. Seleccionar aleatoriamente un esquema de modulación para esta ráfaga
                mod_actual = rng.choice(tipos_modulacion)

                # Generamos un flujo de bits aleatorios (símbolos)
                num_símbolos = 128

                if mod_actual == "BPSK":
                    # 2 fases: 0 y pi
                    fases = rng.choice([0, np.pi], size=num_símbolos)
                    i_vals = np.cos(fases)
                    q_vals = np.sin(fases)

                elif mod_actual == "QPSK":
                    # 4 fases ordenadas en los cuadrantes
                    fases = rng.choice([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4], size=num_símbolos)
                    i_vals = np.cos(fases)
                    q_vals = np.sin(fases)

                elif mod_actual == "8PSK":
                    # 8 fases distribuidas simétricamente en el círculo unitario
                    fases = rng.choice([k * np.pi / 4 for k in range(8)], size=num_símbolos)
                    i_vals = np.cos(fases)
                    q_vals = np.sin(fases)

                elif mod_actual == "QAM16":
                    # Amplitudes en rejilla cuadrática de 4x4 (-3, -1, 1, 3)
                    i_vals = rng.choice([-3, -1, 1, 3], size=num_símbolos) / 3.0
                    q_vals = rng.choice([-3, -1, 1, 3], size=num_símbolos) / 3.0

                # 2. Añadir Ruido Blanco Gaussiano (AWGN) para simular el aire real
                snr_lineal = 10 ** (15 / 10)  # Simula una SNR saludable de 15 dB
                potencia_ruido = 1.0 / (2.0 * snr_lineal)

                i_vals += rng.normal(0, np.sqrt(potencia_ruido), num_símbolos)
                q_vals += rng.normal(0, np.sqrt(potencia_ruido), num_símbolos)

                # Empaquetamos en la matriz shape (128, 2) que espera estrictamente tu CNN
                ventana_iq = np.stack((i_vals, q_vals), axis=-1)

                payload = {}

                # Filtro Constelación
                if self.config["process_constellation"]:
                    payload["constellation"] = {
                        "i": ventana_iq[:, 0].tolist(),
                        "q": ventana_iq[:, 1].tolist()
                    }

                # Filtro Espectro (Densidad de Potencia Espectral via FFT)
                if self.config["process_psd"]:
                    complex_signal = ventana_iq[:, 0] + 1j * ventana_iq[:, 1]
                    fft_vals = np.abs(np.fft.fftshift(np.fft.fft(complex_signal)))
                    psd_db = (20 * np.log10(fft_vals + 1e-6)).tolist()
                    payload["psd"] = psd_db

                # Filtro Inferencia AMC (Predicción de la IA en tiempo real)
                if self.config["run_amc_inference"] and model_dep.model is not None:
                    mod_class, confidence = model_dep.predict(ventana_iq)
                    payload["classification"] = {
                        "modulation": mod_class,
                        "probability": confidence,
                    }

                    # 🚀 ¡IMPRESIÓN EN CONSOLA! Monitoreo directo en tu terminal de Uvicorn
                    print(
                        f"[IA CONSOLA] Señal Generada: {mod_actual:7} | Predicción Red: {mod_class:7} (Confianza: {confidence*100:.2f}%)"
                    )

                # Transmitir por WebSockets a Postman / Frontend si hay clientes conectados
                if payload and stream_manager.active_connections:
                    asyncio.run(stream_manager.broadcast(payload))

                # Pausa de 1 segundo entre ráfagas para legibilidad de logs
                time.sleep(1.0)

            except Exception as e:
                logging.error(f"[SDR WORKER ERROR] Fallo en loop dinámico: {e}")
                time.sleep(1)

# Instanciación del Singleton único para todo el Backend
sdr_worker = SDRWorker()