from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Control SDR"])

class StreamConfig(BaseModel):
    process_constellation: bool = Field(default=True, description="Habilita vectorización cartesiana")
    process_psd: bool = Field(default=True, description="Calcula Fast Fourier Transform de la Señal")
    run_amc_inference: bool = Field(default=True, description="Feed-forward Tensor a red neuronal")

class ScanRequest(BaseModel):
    active: bool = Field(..., description="True inicia captura asíncrona, False cancela Threads")
    config: StreamConfig = StreamConfig()

@router.post("/scan")
async def control_scan(req: ScanRequest):
    """
    Endpoint de transición de Estado.
    Despierta tu transceptor SDR y delega la ejecución al loop de background sin estrangular Fast API.
    """
    logger.info(f"Petición escaneo estado='{req.active}', Config='{req.config.dict()}'")
    
    # Próximamente (Fase 3): Threading SDR Trigger Start/Stop y set variables bandera 
    
    status_msg = "SDR Stream En Linea" if req.active else "SDR Stream Detenido"
    return {"status": "success", "message": status_msg, "applied_config": req.config.dict()}