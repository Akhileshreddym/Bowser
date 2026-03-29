"""
Roomba Hardware Bridge — Sends drive commands to ESP32 over WebSocket.

Commands:
    0 = Forward
    1 = Backward
    2 = Left
    3 = Right
    4 = Stop
"""
import logging
import asyncio
import websockets

logger = logging.getLogger(__name__)

# ESP32 WebSocket endpoint
ESP32_WS_URL = "ws://172.20.10.2/control"

# Command byte mapping
CMD_FORWARD  = 0
CMD_BACKWARD = 1
CMD_LEFT     = 2
CMD_RIGHT    = 3
CMD_STOP     = 4

_ws_connection = None
_ws_lock = asyncio.Lock()


async def _get_connection():
    """Get or create a persistent WebSocket connection to the ESP32."""
    global _ws_connection
    try:
        if _ws_connection is None or _ws_connection.closed:
            _ws_connection = await asyncio.wait_for(
                websockets.connect(
                    ESP32_WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    max_queue=1
                ),
                timeout=2.0
            )
            logger.info(f"Connected to ESP32 at {ESP32_WS_URL}")
        return _ws_connection
    except Exception as e:
        _ws_connection = None
        logger.debug(f"ESP32 not reachable — simulation mode ({e})")
        return None


async def send_command(cmd: int):
    """Send a command byte to the ESP32.
    
    Args:
        cmd: One of 0,1,2,3,4 (forward/backward/left/right/stop).
    """
    async with _ws_lock:
        try:
            ws = await _get_connection()
            if ws:
                await ws.send(bytes([cmd]))
                logger.info(f"ESP32 ← CMD {cmd}")
        except Exception as e:
            global _ws_connection
            _ws_connection = None
            logger.debug(f"ESP32 send failed: {e}")


async def send_forward():
    await send_command(CMD_FORWARD)

async def send_backward():
    await send_command(CMD_BACKWARD)

async def send_left():
    await send_command(CMD_LEFT)

async def send_right():
    await send_command(CMD_RIGHT)

async def send_stop():
    """Emergency stop — always works even if connection drops."""
    await send_command(CMD_STOP)


def drive_string_to_command(drive_str: str) -> int:
    """Convert a 'drive velocity,turn' string into an ESP32 byte command.
    
    Args:
        drive_str: String like 'drive 100,-50' from the pacer agent.
    
    Returns:
        Command int: 0,1,2,3, or 4.
    """
    cleaned = drive_str.replace("drive ", "").replace("drive", "").strip()
    parts = cleaned.split(",")
    
    velocity = 0
    turn = 0
    if len(parts) == 2:
        try:
            velocity = int(parts[0].strip())
            turn = int(parts[1].strip())
        except ValueError:
            return CMD_STOP
    
    # Stop takes priority
    if velocity == 0 and turn == 0:
        return CMD_STOP
    
    # Turn takes priority over forward/backward if significant
    if abs(turn) > abs(velocity):
        if turn > 0:
            return CMD_LEFT
        else:
            return CMD_RIGHT
    
    # Forward/backward
    if velocity > 0:
        return CMD_FORWARD
    elif velocity < 0:
        return CMD_BACKWARD
    
    return CMD_STOP


def configure_esp32(ws_url: str):
    """Reconfigure the ESP32 WebSocket URL at runtime."""
    global ESP32_WS_URL, _ws_connection
    ESP32_WS_URL = ws_url
    _ws_connection = None  # Force reconnect
    logger.info(f"ESP32 target set to {ESP32_WS_URL}")
