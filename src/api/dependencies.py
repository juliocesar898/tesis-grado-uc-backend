import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import logging

from src.processing import spectrogram_stft, normalize_iq, sincronizar_cfo_rf

logger = logging.getLogger(__name__)


class ModelDependency:
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.classes: Optional[np.ndarray] = None

    def load(self, model_path: str, classes_path: str) -> None:
        """Carga en memoria los tensores y gráficos HDF5 de la red en etapa de arranque (lifespan)"""
        try:
            # Inicialización de ML model
            self.model = tf.keras.models.load_model(model_path)
            # Inicialización del diccionario NumPy con las clases
            try:
                self.classes = np.load(classes_path, allow_pickle=True)
            except Exception as e:
                logger.warning(
                    f"No se encontró mapping de clases válido en: {classes_path}. Usaremos genéricos."
                )
                self.classes = np.array([f"Mod_Class_{i}" for i in range(11)])

            logger.info(
                "Dependencias RFML (Modelo y Clases) cargadas exitosamente en memoria RAM."
            )
        except Exception as e:
            logger.error(f"Error cargando dependencias de inferencia H5: {e}")
            raise

    def predict(self, ventana_iq: np.ndarray):
        """
        Recibe una ráfaga IQ, genera los tensores 1D y 2D compatibles con 1024 muestras
        y ejecuta la inferencia en la red híbrida de 24 clases de manera unificada y sincronizada.
        """
        if self.model is None:
            raise ValueError("El modelo híbrido no ha sido cargado en memoria.")

        # 1. Sincronizar y Normalizar Unificadamente (Evita cálculos redundantes)
        if np.iscomplexobj(ventana_iq):
            # Aplicamos compensación de CFO en fase de ráfaga compleja original
            ventana_sincronizada = sincronizar_cfo_rf(ventana_iq)
            # Normalizamos y convertimos a tensor temporal de 2 canales usando funciones nativas
            iq_1d = normalize_iq(ventana_sincronizada, sincronizar=False)
        else:
            # En caso de que venga pre-empaquetado, lo tratamos como entrada final
            iq_1d = ventana_iq

        # Escalabilidad de ventana a 1024 muestras en eje temporal 1D
        window_size = 1024
        if len(iq_1d) > window_size:
            iq_1d = iq_1d[:window_size]
        elif len(iq_1d) < window_size:
            iq_1d = np.pad(iq_1d, ((0, window_size - len(iq_1d)), (0, 0)))

        # 2. Generar espectrograma 2D mediante STFT -> (32, 65, 1)
        # Nota: Al pasarle 'iq_1d' ya sincronizado, le indicamos descativar la redundancia (sincronizar=False)
        stft_2d = spectrogram_stft(iq_1d, sincronizar=False)

        # 3. Formar lotes binarios -> (1, 1024, 2) y (1, 32, 65, 1)
        batch_1d = np.expand_dims(iq_1d, axis=0)
        batch_2d = np.expand_dims(stft_2d, axis=0)

        # 4. Inferencia en el backend TensorFlow
        res = self.model.predict([batch_1d, batch_2d], verbose=0)
        probs = res[0][0] if isinstance(res, (list, tuple)) else res[0]

        # 5. Mapear salida a etiqueta
        idx_max = int(np.argmax(probs))
        confidence = float(probs[idx_max])

        if self.classes is not None and len(self.classes) > idx_max:
            predicted_class = str(self.classes[idx_max])
        else:
            predicted_class = f"Clase_{idx_max}"

        return predicted_class, confidence


# Singleton instanciado listo para inyectarse como global dependency usando Depends()
model_dep = ModelDependency()


def get_model() -> ModelDependency:
    return model_dep