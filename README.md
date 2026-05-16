# Detector de Gesto "Signal for Help" en C++ con OpenCV


---

## ¿Qué es el gesto "Signal for Help"?

El *Signal for Help* (Señal de Ayuda) es un gesto de mano diseñado por la
*Canadian Women's Foundation* para que una persona en peligro pueda pedir ayuda
sin hablar, solo con una cámara visible. Es especialmente útil en videollamadas.

El gesto tiene **tres pasos en secuencia**:

```
Paso 1           Paso 2            Paso 3
=========        ===========       ============
  |||||            |||||             ______
  |||||            |||||            |      |   ← dedos cerrados
  |||||            |  |||           | pulg |     sobre el pulgar
  -----            | P|||           |______|
  PALMA            | UL|||
  abierta          | G| ||
                   |_A_|||
                   pulgar
                   adentro
```

1. **Palma abierta** — muestra la palma hacia la cámara con los 5 dedos extendidos.
2. **Pulgar adentro** — dobla el pulgar hacia la palma (sin cerrar los demás dedos).
3. **Dedos cerrados** — cierra los 4 dedos sobre el pulgar (como un puño con el pulgar dentro).

El programa detecta este gesto **en orden** y solo emite la alerta cuando se completa la secuencia.

---

## Conceptos previos que necesitas saber

### 1. ¿Qué es OpenCV?

OpenCV (*Open Computer Vision Library*) es una biblioteca de código abierto para
procesar imágenes y video en tiempo real. En este proyecto la usamos para:

- Capturar video de la webcam (`cv::VideoCapture`)
- Convertir imágenes entre espacios de color (`cv::cvtColor`)
- Dibujar figuras, texto y esqueletos sobre el video (`cv::circle`, `cv::line`, `cv::putText`)
- Cargar y ejecutar modelos de redes neuronales (`cv::dnn::Net`)

### 2. ¿Qué es un modelo de Deep Learning (red neuronal profunda)?

Un modelo de deep learning es un programa matemático que **aprendió** a resolver
una tarea a partir de miles de ejemplos. Aquí usamos el modelo **MediaPipe Hand Pose**
de Google/OpenCV, que fue entrenado con millones de imágenes de manos y aprendió a
predecir la posición exacta de las articulaciones.

No tenemos que entrenar nada nosotros — solo cargamos el modelo ya entrenado
(el archivo `.onnx`) y le damos imágenes de entrada.

### 3. ¿Qué es ONNX?

ONNX (*Open Neural Network Exchange*) es un formato de archivo estándar para guardar
modelos de redes neuronales. Es como un "PDF para modelos de IA": cualquier software
que soporte ONNX puede leer el modelo, sin importar con qué herramienta fue creado.

OpenCV puede leer modelos ONNX directamente con `cv::dnn::readNet("archivo.onnx")`.

### 4. ¿Qué son los 21 puntos clave (landmarks) de MediaPipe?

El modelo devuelve las coordenadas (x, y) de **21 puntos** de la mano:

```
            8   12  16  20       ← puntas de dedos (TIPS)
            |    |   |   |
            7   11  15  19       ← articulaciones DIP
            |    |   |   |
            6   10  14  18       ← articulaciones PIP
            |    |   |   |
        4   5    9  13  17       ← articulaciones MCP (nudillos)
       /|   |    |   |   |
      3 |   |    |   |   |
      |  \__|    |   |   |
      2    pulgar base
      1
      0   ← MUÑECA (WRIST)
```

Cada número es un índice. Por ejemplo:
- `kp[0]` = muñeca
- `kp[4]` = punta del pulgar
- `kp[8]` = punta del índice
- `kp[12]` = punta del medio

### 5. ¿Qué es el espacio de color YCrCb?

Los colores de piel se detectan mucho mejor en el espacio **YCrCb** que en RGB:

- **Y** = luminancia (brillo). La piel puede ser clara u oscura → no nos importa.
- **Cr** = componente rojo-crominancia. Está en un rango estable para la piel humana.
- **Cb** = componente azul-crominancia. También estable para la piel humana.

Ventaja: la piel sigue apareciendo en el mismo rango de Cr/Cb aunque la iluminación
cambie, porque Y absorbe los cambios de brillo.

---

## Arquitectura del sistema

El programa funciona en 5 etapas encadenadas por cada fotograma de la cámara:

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Webcam    │ →  │  1. SkinDetector │ →  │ 2. HandPose      │
│  640×480    │    │  YCrCb + morfol. │    │  Estimator ONNX  │
└─────────────┘    └──────────────────┘    └──────────────────┘
                          ↓ ROIs                   ↓ 21 puntos
                   ┌──────────────────┐    ┌──────────────────┐
                   │  5. AlertSystem  │ ←  │ 4. GestureDetect │ ← 3. landmarks
                   │  Overlay + notif │    │  Máquina estados │
                   └──────────────────┘    └──────────────────┘
```

### Módulo 1: `SkinDetector` — Detección de piel

**Archivo:** [include/skin_detector.hpp](include/skin_detector.hpp) · [src/skin_detector.cpp](src/skin_detector.cpp)

**¿Qué hace?**
Analiza cada fotograma y encuentra regiones de color "piel". Esto actúa como
un primer filtro muy rápido: si no hay piel, no hay mano.

**¿Cómo funciona paso a paso?**

```
Imagen BGR →  convertir a YCrCb  →  inRange(Cr: 128-185, Cb: 70-138)
                                           ↓
                              máscara binaria (blanco = piel)
                                           ↓
               Operaciones morfológicas (quitar ruido y cerrar huecos)
                                           ↓
               findContours → filtrar por área y forma → ROIs candidatas
```

**¿Por qué morfología?**
La máscara de piel tiene "huecos" y "ruido". Las operaciones morfológicas los corrigen:
- `MORPH_OPEN` = erosión + dilatación → elimina píxeles aislados de ruido.
- `MORPH_CLOSE` = dilatación + erosión → rellena huecos pequeños dentro de la piel.

**¿Por qué filtramos por solidity (solidez)?**
La *solidez* = área del contorno / área del casco convexo. Un puño tiene ~0.85.
Una mano abierta tiene ~0.60. Un objeto alargado (brazo solo) tiene ~0.35.
Filtrando entre 0.30–1.0 rechazamos objetos muy raros.

---

### Módulo 2: `HandPoseEstimator` — Estimación de pose

**Archivo:** [include/hand_pose.hpp](include/hand_pose.hpp) · [src/hand_pose.cpp](src/hand_pose.cpp)

**¿Qué hace?**
Para cada región candidata del SkinDetector, recorta esa área de la imagen,
la pasa por el modelo de red neuronal y obtiene los 21 puntos de la mano.

**Modelo usado:**
`handpose_estimation_mediapipe_2023feb.onnx` — es el modelo MediaPipe Hand Landmark
exportado desde TensorFlow Lite y convertido a formato ONNX. Fue entrenado por Google.

**Preprocesado de la imagen (¡punto crítico!):**

```cpp
// IMPORTANTE: este modelo fue convertido de TFLite, que usa formato NHWC
// (N=lote, H=alto, W=ancho, C=canales). OpenCV por defecto crea blobs en
// formato NCHW — eso produce un error. Debemos crear el blob manualmente.

cv::resize(crop, resized, {224, 224});          // escalar a tamaño del modelo
cv::cvtColor(resized, resized, COLOR_BGR2RGB);  // BGR → RGB
resized.convertTo(imgF, CV_32F, 1.0/255.0);    // normalizar a [0, 1]
int sz[] = {1, 224, 224, 3};                   // forma NHWC
cv::Mat blob(4, sz, CV_32F, imgF.data);        // blob 4D sin copiar datos
net_.setInput(blob);
```

**Salidas del modelo:**
| Índice | Nombre | Forma | Significado |
|--------|--------|-------|-------------|
| 0 | `Identity`   | (1, 63) | 21 puntos × (x, y, z) en píxeles del modelo 224×224 |
| 1 | `Identity_1` | (1, 1)  | Score de presencia de mano [0–1] |
| 2 | `Identity_2` | (1, 1)  | Lateralidad (0=izquierda, 1=derecha) |
| 3 | `Identity_3` | (1, 63) | Puntos en coordenadas 3D mundiales (no usados) |

**Score de confianza:**
Si `Identity_1 < 0.35`, descartamos la detección. Esto previene que objetos que
parecen piel (pero no son manos) activen el sistema.

**Mapeo de coordenadas:**
El modelo trabaja en espacio 224×224. Luego re-escalamos al fotograma original:
```cpp
frame_x = roi.x + landmark_x * (roi.width  / 224.0f);
frame_y = roi.y + landmark_y * (roi.height / 224.0f);
```

---

### Módulo 3: `GestureDetector` — Detección del gesto (máquina de estados)

**Archivo:** [include/gesture_detector.hpp](include/gesture_detector.hpp) · [src/gesture_detector.cpp](src/gesture_detector.cpp)

**¿Qué es una máquina de estados?**
Es un modelo de software que tiene un **estado actual** y **transiciones** entre estados
según condiciones. Solo puede estar en un estado a la vez.

```
                  palma abierta
                  por ≥ 0.8 seg
  ┌──────┐  ─────────────────────────→  ┌────────────┐
  │ IDLE │                              │ OPEN_HAND  │
  └──────┘                              └────────────┘
     ↑                                       │
     │ sin mano                              │ pulgar
     │ por > 1.5s                            │ adentro
     │                                       ↓
     │                              ┌──────────────────┐
     │                              │  THUMB_TUCKED    │
     │                              └──────────────────┘
     │                                       │
     │                                       │ dedos cerrados
     │                                       │ por ≥ 0.5 seg
     │                                       ↓
     └──────────────────────────── ┌──────────────────────┐
                 mano abierta      │  SIGNAL_COMPLETE     │ ← ¡ALERTA!
                 de nuevo          └──────────────────────┘
```

**¿Cómo se detecta si un dedo está extendido?**

Se usa distancia geométrica. La escala de referencia es:
```
palmLen = distancia(MUÑECA, NUDILLO_MEDIO)
```

Un dedo está **extendido** si su punta está más lejos de la muñeca que su nudillo:
```
dist(TIP, MUÑECA) > dist(MCP, MUÑECA) × 1.15
```

Esto funciona para cualquier orientación de la mano (no asume que apunta arriba).

Un dedo está **doblado** si su punta ha vuelto al nivel del nudillo:
```
dist(TIP, MUÑECA) < dist(MCP, MUÑECA) × 1.25
```

**¿Por qué la zona entre 1.15 y 1.25?**
Es una "zona neutral" donde el dedo está en posición intermedia.
Solo contamos como extendido (≥1.15) o doblado (<1.25), evitando estados ambiguos.

**¿Cómo se detecta si el pulgar está adentro?**

Calculamos el **centro de la palma** (promedio de los 4 nudillos):
```cpp
palmCenter = (kp[INDEX_MCP] + kp[MIDDLE_MCP] + kp[RING_MCP] + kp[PINKY_MCP]) / 4
```

El pulgar está **adentro** si su punta está cerca del centro:
```
dist(THUMB_TIP, palmCenter) < palmLen × 0.85
```

Cuando el pulgar está extendido, esa distancia es ≈ 1.0–1.5× palmLen.
Cuando está doblado sobre la palma, baja a ≈ 0.2–0.6× palmLen.

**Tolerancia a frames erróneos (`MAX_FAIL_FRAMES = 8`):**
Si en un frame el dedo parece doblado (por ruido del modelo), no reseteamos inmediatamente.
Esperamos hasta 8 frames consecutivos fallidos (~0.25 segundos a 30fps) antes de retroceder.

---

### Módulo 4: `AlertSystem` — Sistema de alerta

**Archivo:** [include/alert_system.hpp](include/alert_system.hpp) · [src/alert_system.cpp](src/alert_system.cpp)

Cuando se detecta el gesto completo:
1. **Overlay visual** — capa roja semi-transparente + texto grande en el video.
2. **Borde parpadeante** — borde rojo que pulsa 3 veces por segundo.
3. **Sonido de terminal** — `\a` (beep del sistema).
4. **Notificación del sistema** — `notify-send` (Linux) muestra una notificación en el escritorio.
5. **Log en consola** — imprime la hora exacta de detección.

La alerta permanece visible durante **8 segundos** y se apaga sola.

---

## Cómo compilar y ejecutar

### Requisitos

```bash
# Verificar que OpenCV esté instalado
pkg-config --modversion opencv4
# Debe mostrar: 4.x.x

# Verificar CMake
cmake --version
# Debe mostrar: 3.16 o mayor
```

### Compilación

```bash
# 1. Descargar el modelo de red neuronal (~4 MB)
bash download_models.sh

# 2. Crear directorio de compilación
mkdir -p build && cd build

# 3. Configurar con CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. Compilar (usa todos los núcleos del procesador)
cmake --build . -j$(nproc)

# 5. Ejecutar
./signal_for_help
```

### Opciones de ejecución

```bash
./signal_for_help                          # cámara 0, modelo por defecto
./signal_for_help ruta/al/modelo.onnx     # modelo personalizado
./signal_for_help ruta/al/modelo.onnx 1   # cámara 1
```

---

## Controles durante la ejecución

| Tecla | Acción |
|-------|--------|
| `Q` o `Esc` | Salir del programa |
| `R`         | Reiniciar la máquina de estados (volver a IDLE) |
| `G`         | Activar/desactivar panel debug (muestra estado de cada dedo) |
| `D`         | Activar/desactivar máscara de piel (diagnóstico) |
| `V`         | Modo verbose: imprime valores geométricos en consola |

---

## Cómo hacer el gesto correctamente

1. **Colócate a 40–80 cm de la cámara** con buena iluminación.
2. **Levanta la mano** con la palma mirando a la cámara. Mantén los dedos bien abiertos.
3. Espera a que la barra verde del HUD llegue al 100% (≈ 0.8 segundos).
4. **Dobla el pulgar** hacia la palma (los demás dedos siguen abiertos).
5. Espera a que la barra azul llegue al 100% (≈ 0.5 segundos).
6. **Cierra los dedos** sobre el pulgar, formando un puño con el pulgar dentro.
7. ¡La alerta roja aparece en pantalla!

**Panel de debug (tecla G):**
Activa el panel para ver en tiempo real qué detecta el sistema:
- Verde = dedo extendido, Azul = dedo doblado
- Los tres círculos inferiores muestran el progreso de los 3 pasos

---

## Estructura de archivos

```
signal_for_help/
│
├── CMakeLists.txt          ← Configuración de compilación
├── download_models.sh      ← Script para descargar el modelo ONNX
│
├── models/
│   └── handpose_estimation_mediapipe_2023feb.onnx   ← Modelo de red neuronal
│
├── include/                ← Cabeceras (.hpp) — definen las interfaces
│   ├── skin_detector.hpp
│   ├── hand_pose.hpp
│   ├── gesture_detector.hpp
│   └── alert_system.hpp
│
└── src/                    ← Implementaciones (.cpp)
    ├── main.cpp            ← Bucle principal, cámara, interfaz gráfica
    ├── skin_detector.cpp   ← Segmentación de piel (YCrCb)
    ├── hand_pose.cpp       ← Inferencia ONNX, 21 puntos
    ├── gesture_detector.cpp← Máquina de estados del gesto
    └── alert_system.cpp    ← Overlay visual y notificaciones
```

---

## Preguntas frecuentes

**¿Por qué necesitamos dos etapas (piel + red neuronal)?**
La detección de piel es muy rápida pero poco precisa (puede confundir un brazo con una mano).
La red neuronal es precisa pero lenta si la aplicamos a toda la imagen.
Combinándolas: la piel nos da candidatos rápidos, la red confirma que es una mano real.

**¿Por qué el sistema no reacciona a objetos del fondo?**
- El `SkinDetector` solo acepta regiones de color piel → rechaza objetos no-piel.
- El `HandPoseEstimator` da un score de confianza < 0.35 para objetos que no son manos → rechazados.
- La `GestureDetector` requiere la secuencia exacta IDLE→OPEN→THUMB→CLOSED → imposible con un objeto estático.

**¿Por qué el modelo necesita entrada NHWC y no NCHW?**
El modelo fue creado con TensorFlow/TFLite que usa formato NHWC (alto×ancho×canales).
OpenCV usa NCHW (canales×alto×ancho) por defecto. Si usamos `blobFromImage`, el orden es
incorrecto y el modelo falla. Por eso creamos el blob manualmente con la forma correcta.

**¿Puedo usar otra cámara o video pre-grabado?**
```bash
./signal_for_help modelos/handpose.onnx 1     # cámara USB en índice 1
```
Para video pregrabado, modifica `main.cpp` línea `cv::VideoCapture cap(camIndex, ...)` por:
```cpp
cv::VideoCapture cap("video.mp4");
```

**¿Cómo ajusto la sensibilidad?**
En `src/gesture_detector.cpp` puedes cambiar:
- `1.15f` (threshold fingerExtended) → más alto = más difícil extender
- `1.25f` (threshold fingerCurled) → más bajo = más difícil cerrar
- `0.85f` (threshold thumbTucked) → más bajo = requiere pulgar más adentro
- `0.8f` (openHoldSec) → segundos que debe mantenerse la palma abierta

---

## Referencias

- [OpenCV Hand Keypoint Detection](https://learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/)
- [Modelo MediaPipe en HuggingFace](https://huggingface.co/opencv/handpose_estimation_mediapipe)
- [OpenCV Model Zoo — Hand Pose](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)
- [Signal for Help — Canadian Women's Foundation](https://canadianwomen.org/signal-for-help/)
- [MediaPipe Hand Landmarks](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
