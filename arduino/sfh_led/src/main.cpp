#include <Arduino.h>

void setup() {
  Serial.begin(9600);
  pinMode(11, OUTPUT);
  digitalWrite(11, LOW);
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg == "SFH") {
      digitalWrite(11, HIGH);
      delay(3000);
      digitalWrite(11, LOW);
    }
  }
}
