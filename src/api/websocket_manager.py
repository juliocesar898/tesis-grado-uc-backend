from fastapi import WebSocket
import typing
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Administra múltiples conexiones asíncronas para el Streaming DSP en SD (Broadcasting)"""

    def __init__(self):
        self.active_connections: typing.List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Nuevo cliente suscrito al Stream SDR. Totales: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(
            f"Cliente WS desconectado. Restantes: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict):
        """Envía el payload DSP + RFML asincrónicamente a todos los frontends conectados"""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Falla interrumpiendo un socket inactivo: {e}")
                self.disconnect(connection)


stream_manager = ConnectionManager()
