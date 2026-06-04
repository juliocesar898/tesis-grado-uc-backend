from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.api.websocket_manager import stream_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Stream SDR"])


@router.websocket("/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Endpoint de ingesta WSS. Al conectarse, su ID queda suscrito a
    la piscina de consumidores.
    """
    await stream_manager.connect(websocket)
    try:
        while True:
            # Sockets en escucha pasiva con ping
            # El broadcast() de websocket_manager insertará data desde hilo secundario capturador.
            data = await websocket.receive_text()
            logger.debug(f"Ping recibido del Web Client: {data}")

    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
