import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import logging

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
                logger.warning(f"No se encontró mapping de clases válido en: {classes_path}. Usaremos genéricos.")
                self.classes = np.array([f"Mod_Class_{i}" for i in range(11)])
                
            logger.info("Dependencias RFML (Modelo y Clases) cargadas exitosamente en memoria RAM.")
        except Exception as e:
            logger.error(f"Error cargando dependencias de inferencia H5: {e}")
            raise

    def predict(self, iq_batch: np.ndarray, stft_batch: np.ndarray) -> Tuple[str, float]:
        """Procesa paso feed-forward de forma Thread-Safe"""
        if self.model is None or self.classes is None:
            raise RuntimeError("API invocada sin cargar el backend clasificador.")
        
        preds = self.model.predict([iq_batch, stft_batch], verbose=0)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        modulation_class = str(self.classes[idx])
        
        return modulation_class, confidence

# Singleton instanciado listo para inyectarse como global dependencie usando Depends()
model_dep = ModelDependency()

def get_model() -> ModelDependency:
    return model_dep