/**
 * SilentPilot EMG Streamer Firmware
 * 
 * Target: ESP32 (tested on ESP32-WROOM-32)
 * 
 * Reads 4 analog channels at 250 Hz using ADC and streams binary
 * packets over USB Serial at 115200 baud.
 * 
 * Packet format (19 bytes):
 *   Header:    0xAA 0x55 (2 bytes)
 *   Version:   0x01 (1 byte)
 *   Seq:       uint16 big-endian (2 bytes)
 *   Timestamp: uint32 big-endian, microseconds (4 bytes)
 *   Ch0-Ch3:   uint16 big-endian each (8 bytes)
 *   CRC16:     uint16 big-endian (2 bytes) -- currently 0x0000
 * 
 * ADC Pins (ESP32):
 *   CH0 = GPIO 32 (ADC1_CH4)
 *   CH1 = GPIO 33 (ADC1_CH5)
 *   CH2 = GPIO 34 (ADC1_CH6)
 *   CH3 = GPIO 35 (ADC1_CH7)
 * 
 * For MyoWare 2.0 sensors:
 *   Connect the ENV (envelope) output to each ADC pin.
 *   GND and 3.3V to the sensor power pins.
 * 
 * For raw Ag/AgCl electrodes:
 *   Use an instrumentation amplifier (AD8232 or INA128) between
 *   the electrodes and the ADC pins.
 * 
 * Build:
 *   - Board: "ESP32 Dev Module" in Arduino IDE
 *   - Upload Speed: 921600
 *   - CPU Frequency: 240MHz
 */

// --- Configuration ---
#define SAMPLE_RATE    250    // Hz
#define NUM_CHANNELS   4
#define SERIAL_BAUD    115200
#define LED_PIN        2       // Built-in LED for heartbeat

// ADC pins (ESP32 ADC1 channels -- safe to use with WiFi)
const int ADC_PINS[NUM_CHANNELS] = {32, 33, 34, 35};

// Packet header
const uint8_t HEADER[2] = {0xAA, 0x55};
const uint8_t VERSION = 0x01;

// --- State ---
uint16_t seq = 0;
unsigned long lastSampleMicros = 0;
const unsigned long sampleIntervalMicros = 1000000UL / SAMPLE_RATE;  // 4000 us for 250 Hz

// Heartbeat
unsigned long lastBlinkMillis = 0;
bool ledState = false;

// --- Packet buffer ---
// 2 (header) + 1 (version) + 2 (seq) + 4 (timestamp) + 8 (4 channels * 2) + 2 (CRC) = 19 bytes
uint8_t packet[19];

void setup() {
  Serial.begin(SERIAL_BAUD);
  
  // Configure ADC
  analogReadResolution(12);  // 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db);  // Full range: 0-3.3V
  
  // Configure pins
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(ADC_PINS[i], INPUT);
  }
  
  pinMode(LED_PIN, OUTPUT);
  
  // Wait for serial
  delay(100);
  
  lastSampleMicros = micros();
}

void loop() {
  unsigned long now = micros();
  
  // Sample at fixed rate
  if (now - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros += sampleIntervalMicros;
    
    // Read all channels
    uint16_t channels[NUM_CHANNELS];
    for (int i = 0; i < NUM_CHANNELS; i++) {
      channels[i] = analogRead(ADC_PINS[i]);
    }
    
    // Build packet
    uint32_t timestamp = (uint32_t)(now & 0xFFFFFFFF);
    buildPacket(channels, timestamp);
    
    // Send packet
    Serial.write(packet, sizeof(packet));
    
    seq++;
  }
  
  // Heartbeat LED (blink every 500ms to confirm operation)
  if (millis() - lastBlinkMillis >= 500) {
    lastBlinkMillis = millis();
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
}

void buildPacket(uint16_t* channels, uint32_t timestamp) {
  int idx = 0;
  
  // Header
  packet[idx++] = HEADER[0];
  packet[idx++] = HEADER[1];
  
  // Version
  packet[idx++] = VERSION;
  
  // Sequence (big-endian)
  packet[idx++] = (seq >> 8) & 0xFF;
  packet[idx++] = seq & 0xFF;
  
  // Timestamp microseconds (big-endian)
  packet[idx++] = (timestamp >> 24) & 0xFF;
  packet[idx++] = (timestamp >> 16) & 0xFF;
  packet[idx++] = (timestamp >> 8) & 0xFF;
  packet[idx++] = timestamp & 0xFF;
  
  // Channels (big-endian)
  for (int i = 0; i < NUM_CHANNELS; i++) {
    packet[idx++] = (channels[i] >> 8) & 0xFF;
    packet[idx++] = channels[i] & 0xFF;
  }
  
  // CRC16 (placeholder -- set to 0 for hackathon speed)
  packet[idx++] = 0x00;
  packet[idx++] = 0x00;
}
