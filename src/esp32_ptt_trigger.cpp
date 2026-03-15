// ESP32-CAM UDP push-to-talk trigger sketch.
//
// This file is intentionally separate from src/main.cpp so you can keep the
// existing camera test sketch untouched.
//
// Build this sketch by defining ENABLE_PTT_TRIGGER_SKETCH in build flags.

#ifdef ENABLE_PTT_TRIGGER_SKETCH

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// Provided by credentials.ini via build_flags in platformio.ini.
#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif

// Update this to your laptop's local IPv4 address.
#ifndef LAPTOP_IP
#define LAPTOP_IP "192.168.1.100"
#endif

constexpr uint16_t UDP_PORT = 5005;
constexpr int BUTTON_PIN = 13;
constexpr unsigned long DEBOUNCE_MS = 35;

constexpr char PTT_START_MSG[] = "PTT_START";
constexpr char PTT_STOP_MSG[] = "PTT_STOP";

WiFiUDP udp;
IPAddress laptopIp;

bool stableButtonState = HIGH;
bool lastRawButtonState = HIGH;
unsigned long lastEdgeMillis = 0;

void connectWiFi()
{
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(300);
        Serial.print('.');
    }
    Serial.println();
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
}

void sendUdpMessage(const char *message)
{
    udp.beginPacket(laptopIp, UDP_PORT);
    udp.print(message);
    udp.endPacket();
    Serial.print("Sent UDP: ");
    Serial.println(message);
}

void handleButtonStateChange(bool newStableState)
{
    // INPUT_PULLUP: LOW = pressed, HIGH = released.
    if (newStableState == LOW)
    {
        sendUdpMessage(PTT_START_MSG);
    }
    else
    {
        sendUdpMessage(PTT_STOP_MSG);
    }
}

void setup()
{
    Serial.begin(115200);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    connectWiFi();

    if (!laptopIp.fromString(LAPTOP_IP))
    {
        Serial.println("Invalid LAPTOP_IP. Check macro value.");
    }

    udp.begin(UDP_PORT);

    stableButtonState = digitalRead(BUTTON_PIN);
    lastRawButtonState = stableButtonState;
    lastEdgeMillis = millis();

    Serial.println("PTT trigger ready on GPIO 13.");
}

void loop()
{
    const bool rawState = digitalRead(BUTTON_PIN);

    if (rawState != lastRawButtonState)
    {
        lastRawButtonState = rawState;
        lastEdgeMillis = millis();
    }

    // Only commit state changes once input has remained stable long enough.
    if ((millis() - lastEdgeMillis) >= DEBOUNCE_MS && rawState != stableButtonState)
    {
        stableButtonState = rawState;
        handleButtonStateChange(stableButtonState);
    }

    delay(2);
}

#endif // ENABLE_PTT_TRIGGER_SKETCH
