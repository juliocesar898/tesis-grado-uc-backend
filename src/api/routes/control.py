from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from src.capture import sdr_worker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Control SDR"])

class StreamConfig(BaseModel):
    mode: Optional[str] = Field(default="simulation", description="'simulation' o 'pluto'")
    pluto_freq_hz: Optional[int] = Field(default=2480000000, description="Frecuencia central en Hz (ej. 2480000000)")
    pluto_sample_rate: Optional[int] = Field(default=2000000, description="Tasa de muestreo en SPS (ej. 2000000)")
    pluto_gain_db: Optional[float] = Field(default=30.0, description="Ganancia Rx en dB")
    process_constellation: bool = Field(default=True)
    process_psd: bool = Field(default=True)
    run_amc_inference: bool = Field(default=True)
    # Atributos dinámicos de simulación y laboratorio docente
    min_snr_db: float = Field(default=6.0, description="Nivel mínimo de SNR en dB")
    max_snr_db: Optional[float] = Field(default=None, description="Nivel máximo de SNR en dB")
    allowed_classes: Optional[List[str]] = Field(default=None, description="Lista de modulaciones permitidas en el test")
    class_hold_repeats: int = Field(default=4, description="Número de ráfagas continuas de la misma modulación")

class ScanRequest(BaseModel):
    active: bool = Field(...)
    config: StreamConfig = StreamConfig()

@router.get("/status")
async def get_status():
    return {
        "status": "success",
        "is_running": sdr_worker.is_running,
        "config": sdr_worker.config
    }

@router.post("/scan")
async def control_scan(req: ScanRequest):
    logger.info(f"Petición de estado recibida: active={req.active}")
    
    # Inyectar dinámicamente la configuración en el Worker de simulación / hardware
    cfg_data = req.config.model_dump() if hasattr(req.config, "model_dump") else req.config.dict()
    sdr_worker.actualizar_configuracion(cfg_data)
    
    if req.active:
        sdr_worker.iniciar()
        status_msg = f"SDR Stream En Línea (Modo: {sdr_worker.config.get('mode')})"
    else:
        sdr_worker.detener()
        status_msg = "SDR Stream Detenido"
        
    return {
        "status": "success", 
        "message": status_msg, 
        "applied_config": sdr_worker.config
    }