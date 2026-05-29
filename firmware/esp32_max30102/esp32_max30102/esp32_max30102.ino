#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h" // SparkFun library for MAX30102

// --- GLOBALS ---
#define BUZZER_PIN 12 
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET    -1 

// --- DUAL I2C PINS ---
#define I2C_SDA_SENSOR 32 
#define I2C_SCL_SENSOR 33 

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String lcdBuffer = ""; 
bool buzzerTriggered = false; 

// --- NON-BLOCKING BUZZER VARIABLES ---
int beepCount = 0;
unsigned long buzzerLastToggle = 0;

// Initialize Sensor & Second I2C Bus
MAX30105 particleSensor;
TwoWire I2C_SENSOR = TwoWire(1); 

void setup() {
  Serial.begin(115200);
  delay(1000); 

  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize ESP32 Hardware I2C 1 (OLED: GPIO 21 = SDA, GPIO 22 = SCL)
  Wire.begin(21, 22);
  Wire.setClock(100000); 

  // Initialize ESP32 Hardware I2C 2 (SENSOR: GPIO 32 = SDA, GPIO 33 = SCL)
  I2C_SENSOR.begin(I2C_SDA_SENSOR, I2C_SCL_SENSOR);
  I2C_SENSOR.setClock(400000); 

  // --- 1. OLED Boot Sequence ---
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 OLED allocation failed"));
    for(;;); 
  }
  
  display.clearDisplay();
  display.setTextSize(1);     
  display.setTextColor(SSD1306_WHITE); 
  
  display.setCursor(0, 0);
  display.print("Booting SAS...");
  display.setCursor(0, 16);
  display.print("Connecting...");
  display.display(); 

  // --- 2. Initialize Sensor at Fast I2C Speed on SECOND BUS ---
  if (!particleSensor.begin(I2C_SENSOR, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 failed to initialize. Check wiring.");
    while (1);
  }

  // --- 3. Configuration matching your 400Hz requirement ---
  byte ledBrightness = 60;   
  byte sampleAverage = 1;    
  byte ledMode = 2;          
  int sampleRate = 400;      
  int pulseWidth = 411;      
  int adcRange = 4096;       

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  Serial.println("Sensor initialized. Streaming to PC...");
}

// --- NON-BLOCKING BUZZER HANDLER ---
void handleBuzzer() {
  if (beepCount > 0) {
    unsigned long currentMillis = millis();
    // 150ms interval (100ms beep + 50ms silence)
    if (currentMillis - buzzerLastToggle >= 150) { 
      buzzerLastToggle = currentMillis;
      tone(BUZZER_PIN, 1000, 100); // Fire 1000Hz tone for exactly 100ms
      beepCount--; // Countdown the beeps
    }
  }
}

// Helper to trigger the beeps safely
void triggerTripleBeep() {
  beepCount = 3; 
  buzzerLastToggle = millis() - 150; // Force it to fire immediately
}

void loop() {
  // --- STAGE 0: HANDLE BACKGROUND BEEPING ---
  handleBuzzer();

  // --- STAGE 1: STREAM RAW DATA (400Hz) ---
  particleSensor.check(); 

  while (particleSensor.available()) {
    Serial.print("Red:");
    Serial.print(particleSensor.getFIFORed());
    Serial.print(",IR:");
    Serial.println(particleSensor.getFIFOIR());
    
    particleSensor.nextSample();
  }

  // --- STAGE 2: NON-BLOCKING SERIAL LISTENER (OLED UPDATES) ---
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\r') continue; 
    
    if (c == '\n') {
      int firstComma = lcdBuffer.indexOf(',');
      int secondComma = lcdBuffer.indexOf(',', firstComma + 1);

      if (firstComma > 0 && secondComma > 0) {
        String hr = lcdBuffer.substring(0, firstComma);
        String spo2 = lcdBuffer.substring(firstComma + 1, secondComma);
        String statusChar = lcdBuffer.substring(secondComma + 1);

        display.clearDisplay();
        display.setTextSize(1);
        display.setCursor(0, 0);

        if (statusChar == "E") {
          // NO SIGNAL STATE
          display.print("HR = ???  SpO2 = ???");
          display.setCursor(0, 16);
          display.print("Stat: NO SIGNAL");
          
          if (!buzzerTriggered) {
            triggerTripleBeep(); // CHANGED: Fire triple beep on first error
            buzzerTriggered = true;
          }
        } 
        else {
          // NORMAL or APNEA STATES
          buzzerTriggered = false; // Reset buzzer flag for next error
          
          display.print("HR = " + hr + " SpO2 = " + spo2 + "%");
          display.setCursor(0, 16);
          
          if (statusChar == "N") {
            display.setTextColor(SSD1306_WHITE, SSD1306_BLACK); 
            display.print("Stat: NORMAL");
          } else if (statusChar == "A") {
            display.setTextColor(SSD1306_BLACK, SSD1306_WHITE); 
            display.print(" APNEA DETECTED! ");
            display.setTextColor(SSD1306_WHITE, SSD1306_BLACK);
            
            triggerTripleBeep(); // CHANGED: Fire triple beep every time Apnea is detected
          } else {
            display.print("Stat: WAIT...");
          }
        }
        display.display();
      }
      lcdBuffer = ""; 
    } else {
      lcdBuffer += c; 
    }
  }
}