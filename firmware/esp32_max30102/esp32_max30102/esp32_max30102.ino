#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h" // SparkFun library for MAX30102

// --- GLOBALS ---
#define BUZZER_PIN 32
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET    -1 

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String lcdBuffer = ""; 
bool buzzerTriggered = false; 

// Initialize Sensor
MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  delay(1000); 

  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize ESP32 Hardware I2C (GPIO 21 = SDA, GPIO 22 = SCL)
  Wire.begin(21, 22);

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

  // --- 2. Initialize Sensor at Fast I2C Speed ---
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
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

void loop() {
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
            tone(BUZZER_PIN, 1000, 200); 
            buzzerTriggered = true;
          }
        } 
        else {
          // NORMAL or APNEA STATES
          buzzerTriggered = false; // Reset buzzer for next error
          
          display.print("HR = " + hr + " SpO2 = " + spo2 + "%");
          display.setCursor(0, 16);
          
          if (statusChar == "N") {
            display.setTextColor(SSD1306_WHITE, SSD1306_BLACK); 
            display.print("Stat: NORMAL");
          } else if (statusChar == "A") {
            display.setTextColor(SSD1306_BLACK, SSD1306_WHITE); 
            display.print(" APNEA DETECTED! ");
            display.setTextColor(SSD1306_WHITE, SSD1306_BLACK);
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