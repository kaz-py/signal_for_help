// Signal for Help — receptor serial
// Escucha "SFH" por serial y enciende el LED integrado 3 segundos.

void setup() {
  Serial.begin(9600);
  pinMode(11, OUTPUT);
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
