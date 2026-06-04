from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.dependencies import model_dep
from src.api.routes import control, stream
import logging
from pathlib import Path

# Configuración base del sistema logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Definción de rutas con Path handling agnóstico a SO (Linux/Windows)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "clasificador_sdr.h5"
CLASSES_PATH = BASE_DIR / "models" / "clases_modulacion.npy"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Controlador de ciclo de vida unificado (FastAPI > 0.93+)
    Todo lo alojado aquí se evalúa pre-apertura de puertos TPC impidiendo
    retrasos de warmup si el hardware entra de golpe
    """
    logger.info(">>> Iniciando DeepSignal RFML Server...")

    # 1. Cargar tensores matemáticos. (Permite fallos silenciosos para pruebas iniciales)
    try:
        model_dep.load(str(MODEL_PATH), str(CLASSES_PATH))
    except Exception as e:
        logger.warning(
            f"Iniciando API degradada o pre-entrenamiento. Modelo .h5 missing: {e}"
        )

    yield
    # 2. Rutina de cierre preventivo
    logger.info("<<< Apagando Motor SDR. Cerrando procesos...")


# Instancia Root Asíncrona
app = FastAPI(
    title="DeepSignal IA & DSP",
    description="Backend AMC asíncrono para Inferencia de Modulaciones SDR",
    version="1.0.0",
    lifespan=lifespan,
)

# Integración del Namespace API Routing
app.include_router(control.router, prefix="/api/v1/control")
app.include_router(stream.router, prefix="/api/v1/stream")
