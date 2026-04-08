// arduino_autonomic_analog.ino
const int SENSOR_PIN = A0; 
const int BAUD_RATE = 9600;

void setup() {
  Serial.begin(BAUD_RATE);
}

void loop() {
  // Read analog autonomic/neural data (0-1023)
  int rawVal = analogRead(SENSOR_PIN);
  
  // Stream raw value to Python
  Serial.println(rawVal);
  
  // ~20Hz sampling rate
  delay(50); 
}
