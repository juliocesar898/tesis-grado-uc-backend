from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import logging
from src.capture import sdr_worker  # Importamos el worker global

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Control SDR"])

class StreamConfig(BaseModel):
    process_constellation: bool = Field(default=True)
    process_psd: bool = Field(default=True)
    run_amc_inference: bool = Field(default=True)

class ScanRequest(BaseModel):
    active: bool = Field(...)
    config: StreamConfig = StreamConfig()

@router.post("/scan")
async def control_scan(req: ScanRequest):
    logger.info(f"Petición de estado recibida: active={req.active}")
    
    # Aplicar dinámicamente los filtros/configuraciones de características elegidas
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