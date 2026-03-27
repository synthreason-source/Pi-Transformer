// --------------------------------------------------------------------------
// Autonomic Signal Streamer for V18-RP
// Attach electrodes to Analog Pin 0 (A0) and GND
// --------------------------------------------------------------------------

const int electrodePin = A0;
float smoothedValue = 0.0;
const float alpha = 0.15; // EMA smoothing factor (lower = smoother but slower)

void setup() {
  // 115200 baud ensures the serial buffer doesn't bottleneck the sampling rate
  Serial.begin(115200);
  
  // Initial prime of the smoother
  smoothedValue = analogRead(electrodePin);
}

void loop() {
  int rawValue = analogRead(electrodePin);
  
  // Apply Exponential Moving Average (EMA)
  smoothedValue = (alpha * rawValue) + ((1.0 - alpha) * smoothedValue);
  
  // Output the smoothed float over Serial
  Serial.println(smoothedValue);
  
  // Delay 10ms for a ~100Hz sampling rate
  delay(10);
}
