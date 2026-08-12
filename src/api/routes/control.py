from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from src.capture import sdr_worker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Control SDR"])

class StreamConfig(BaseModel):
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

@router.post("/scan")
async def control_scan(req: ScanRequest):
    logger.info(f"Petición de estado recibida: active={req.active}")
    
    # Inyectar dinámicamente la configuración en el Worker de simulación
    sdr_worker.actualizar_configuracion(req.config.dict())
    
    if req.active:
        sdr_worker.iniciar()
        status_msg = "SDR Stream En Línea"
    else:
        sdr_worker.detener()
        status_msg = "SDR Stream Detenido"
        
    return {
        "status": "success", 
        "message": status_msg, 
        "applied_config": sdr_worker.config
    }