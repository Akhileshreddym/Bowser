/*
 * Roomba ESP32 Bridge Firmware
 * ─────────────────────────────
 * Connects to WiFi, listens for WebSocket commands on /control,
 * and drives the Roomba via serial.
 *
 * Commands (single byte over WebSocket):
 *   '0' = Forward
 *   '1' = Backward
 *   '2' = Left
 *   '3' = Right
 *   '4' = Stop
 *
 * Wiring:
 *   ESP32 TX2 (GPIO17) → Roomba RX
 *   ESP32 RX2 (GPIO16) → Roomba TX
 *   ESP32 GND           → Roomba GND
 */

#include <WiFi.h>
#include <WebSocketsServer.h>

// ─── CONFIG ───
const char* WIFI_SSID     = "USF-eduroam-guest-Wifi";
const char* WIFI_PASSWORD = "";

WebSocketsServer webSocket = WebSocketsServer(80);
HardwareSerial& roombaSerial = Serial2;

unsigned long lastCommandTime = 0;
const unsigned long SAFETY_TIMEOUT_MS = 3000;  // Auto-stop after 3s of silence

// ─── ROOMBA COMMANDS ───
void roombaForward()  { roombaSerial.write(0); Serial.println("[ROOMBA] Forward");  }
void roombaBackward() { roombaSerial.write(1); Serial.println("[ROOMBA] Backward"); }
void roombaLeft()     { roombaSerial.write(2); Serial.println("[ROOMBA] Left");     }
void roombaRight()    { roombaSerial.write(3); Serial.println("[ROOMBA] Right");    }
void roombaStop()     { roombaSerial.write(4); Serial.println("[ROOMBA] Stop");     }

// ─── WEBSOCKET EVENT HANDLER ───
void onWebSocketEvent(uint8_t clientNum, WStype_t type, uint8_t * payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.printf("[WS] Client %u connected\n", clientNum);
            break;

        case WStype_DISCONNECTED:
            Serial.printf("[WS] Client %u disconnected\n", clientNum);
            roombaStop();  // Safety: stop on disconnect
            break;

        case WStype_TEXT:
            if (length > 0) {
                char cmd = (char)payload[0];
                lastCommandTime = millis();

                switch (cmd) {
                    case '0': roombaForward();  break;
                    case '1': roombaBackward(); break;
                    case '2': roombaLeft();     break;
                    case '3': roombaRight();    break;
                    case '4': roombaStop();     break;
                    default:
                        Serial.printf("[WS] Unknown command: %c\n", cmd);
                        break;
                }

                // Echo back confirmation
                webSocket.sendTXT(clientNum, "OK");
            }
            break;

        default:
            break;
    }
}

// ─── SETUP & LOOP ───
void setup() {
    Serial.begin(115200);
    roombaSerial.begin(115200, SERIAL_8N1, 16, 17);  // RX=16, TX=17

    Serial.println("\n[ESP32] Starting Roomba Bridge...");

    // Connect WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[WIFI] Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\n[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

    // Start WebSocket server
    webSocket.begin();
    webSocket.onEvent(onWebSocketEvent);
    Serial.println("[WS] WebSocket server started on /control");
}

void loop() {
    webSocket.loop();

    // Safety: stop if no command for 3 seconds
    if (lastCommandTime > 0 && (millis() - lastCommandTime) > SAFETY_TIMEOUT_MS) {
        roombaStop();
        lastCommandTime = 0;
        Serial.println("[SAFETY] Timeout: Roomba stopped");
    }
}
