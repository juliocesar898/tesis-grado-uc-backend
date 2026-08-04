import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import logging

from src.processing import spectrogram_stft, normalize_iq

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
        Recibe una ráfaga IQ (compleja o 128x2), genera los tensores 1D y 2D
        y ejecuta la inferencia en la red híbrida.
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido cargado en memoria.")

        # 1. Preparar entrada 1D -> (128, 2) SIN DISTORSIONAR LA AMPLITUD
        if np.iscomplexobj(ventana_iq):
            # Solo extraemos los componentes real e imaginario sin re-escalar
            iq_1d = np.column_stack((np.real(ventana_iq), np.imag(ventana_iq)))
        else:
            iq_1d = ventana_iq

        if len(iq_1d) > 128:
            iq_1d = iq_1d[:128]
        elif len(iq_1d) < 128:
            iq_1d = np.pad(iq_1d, ((0, 128 - len(iq_1d)), (0, 0)))

        # 2. Generar espectrograma 2D mediante STFT -> (32, 9, 1)
        stft_2d = spectrogram_stft(iq_1d)

        # 3. Formar lotes (batch size = 1) -> (1, 128, 2) y (1, 32, 9, 1)
        batch_1d = np.expand_dims(iq_1d, axis=0)
        batch_2d = np.expand_dims(stft_2d, axis=0)

        # 4. Inferencia en Keras
        res = self.model.predict([batch_1d, batch_2d], verbose=0)
        
        # Si Keras devuelve una tupla/lista de salidas, tomamos la primera (y única)
        if isinstance(res, (list, tuple)):
            probs = res[0][0]
        else:
            probs = res[0]

        # 5. Obtener clase con mayor probabilidad
        idx_max = int(np.argmax(probs))
        confidence = float(probs[idx_max])

        if hasattr(self, "classes") and self.classes is not None and len(self.classes) > idx_max:
            predicted_class = str(self.classes[idx_max])
        else:
            predicted_class = f"Clase_{idx_max}"

        return predicted_class, confidence


# Singleton instanciado listo para inyectarse como global dependencie usando Depends()
model_dep = ModelDependency()


def get_model() -> ModelDependency:
    return model_dep
