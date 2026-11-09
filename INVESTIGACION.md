# Investigación: ajuste automático de exposición en tiempo real

**Fecha:** 21 sep 2026
**Autor:** Gustavo Estrada
**Proyecto:** BodySense V1 — cámaras estéreo USB + OpenCV en Windows

## 1. Pregunta

¿Podemos hacer un algoritmo que ajuste Exposure / Brightness / Contrast solo, según el entorno donde esté la cámara?

Contexto: hoy los valores se fijan a mano en `test_cam_controls.py` (Exposure -4, Brightness 0, Contrast 50) y cada sala tiene luz distinta. La meta es que el sistema se adapte solo y mantenga imagen estable para MediaPipe y la triangulación.

## 2. Cómo funciona el "auto" que ya trae el driver

Antes de inventar algo, lo que ya existe:

- Las webcams USB (UVC) traen auto-exposición en firmware: mide el brillo y mueve tiempo de exposición y ganancia.
- En OpenCV se controla con `CAP_PROP_AUTO_EXPOSURE`. En DirectShow el valor 1.0 suele significar control manual y otros valores (p. ej. 0.25 según reportes de la comunidad) dejan el auto del driver. Es inconsistente entre drivers: hay que verificar con `cap.get` qué acepta cada cámara (issue opencv/opencv#9738, foros de ELP/ArduCam).
- Problema para nosotros: el auto del driver optimiza que la imagen "se vea bonita", no que MediaPipe detecte bien ni que las dos cámaras queden parejas. Además cada cámara hace su propio auto por separado y el estéreo sufre si quedan disparejas.

Conclusión: el auto del driver sirve como punto de partida, pero para BodySense conviene control manual con algoritmo propio.

## 3. Algoritmos existentes

### 3.1 Brillo medio + control proporcional (el clásico industrial)

Es lo que hacen Basler (Exposure Auto / Gain Auto, GenICam) y Teledyne: se define un **brillo objetivo** y se ajusta el tiempo de exposición dentro de límites hasta alcanzarlo; si la exposición llega al tope, se mueve la ganancia.

- Cómo: `error = objetivo - brillo_medido`, `exposicion += k * error`, con límites inferior/superior y amortiguamiento (damping) para no oscilar.
- Pros: simple, rápido, funciona en tiempo real, fácil de implementar con `cap.set`.
- Contras: el brillo medio se deja engañar (ventana brillante atrás, ropa blanca/negra). No distingue qué parte de la imagen importa.

### 3.2 Histograma óptimo (Schulz et al.)

En vez del promedio, analiza el histograma de brillo y busca distribuir los tonos sin quemar blancos ni tapar negros ("Using brightness histogram to perform optimum auto exposure").

- Pros: más robusto que el promedio ante fondos claros/oscuros.
- Contras: más cómputo, hay que definir qué histograma es "óptimo" para nuestra escena (persona + tablero).

### 3.3 ROI ponderado (región de interés)

Basler lo llama Auto Function ROI: el brillo objetivo se mide solo en una zona (p. ej. donde está la persona), no en todo el cuadro. La variante de Schulz para conductor mide solo la región de la cara.

- Pros: para nosotros es ideal, la persona es lo que importa, no la ventana de atrás.
- Contras: hay que saber dónde está la persona (podemos reusar la caja de MediaPipe del cuadro anterior).

### 3.4 Perfil exposición-primero, ganancia-después

Regla estándar en visión industrial: primero se sube exposición (imagen limpia) y solo cuando hay riesgo de motion blur se sube ganancia (que mete ruido). Basler lo formaliza como Auto Function Profile.

- Pros: evita el error clásico de subir ganancia y llenar todo de ruido, lo que mata la detección de esquinas del tablero y los landmarks.
- Contras: con persona en movimiento, exposición larga = blur. Hay que poner techo a la exposición.

### 3.5 Enfoques avanzados (referencia, no para V1)

- **HDR / multi-exposición:** combina varias exposiciones. Bueno pero lento y las webcams baratas no lo dan.
- **Control activo para odometría visual** (Zhang et al., ICRA 2017): ajusta exposición para maximizar información de gradientes, no brillo. Interesante a futuro porque nuestro tablero vive de gradientes.
- **Aprendizaje por refuerzo** (artículo 2024, arXiv 2404.01636): política que aprende a mover exposición/ganancia/apertura. Potente pero necesita entrenamiento y es overkill hoy.

## 4. Programas que ya lo implementan (para copiar ideas)

| Programa | Qué hace | Qué copiar |
|---|---|---|
| Basler pylon + GenICam | Exposure Auto / Gain Auto con objetivo, límites, ROI y damping | El diseño completo: objetivo + límites + ROI + perfil |
| FLIR / Teledyne SpinView | Auto exposure con propiedades configurables | Misma arquitectura, otra marca |
| Intel RealSense | Auto-exposición por defecto con ROI | ROI sobre la persona |
| v4l2 (Linux) | Controles `exposure_auto`, `exposure_absolute` por ioctl | Lógica de control manual equivalente |
| OBS / apps de cámara | Auto con compensación de contraluz | Backlight compensation como idea |
| OpenMV | Control de exposición/ganancia por script para contraste | Inspiración para modo "calibración" vs "ejercicio" |
| Drivers ROS (pylon-ros, usb_cam) | Exponen el auto como parámetros | Cómo parametrizarlo limpio |

## 5. Contras generales (honestos)

1. **Cada driver miente distinto:** en DSHOW/MSMF los rangos y el flag de auto cambian por cámara. Hay que calibrar el software contra cada modelo.
2. **Oscilación:** sin damping ni banda muerta, la imagen "respira" (claro-oscuro-claro). Para ejercicio eso cansa y rompe el tracking.
3. **Dos cámaras, un problema doble:** si cada una ajusta por su cuenta quedan disparejas y el estéreo se degrada. Hay que acoplarlas (misma banda objetivo, ajustes simétricos).
4. **Conflicto calibración vs ejercicio:** el tablero quiere una exposición y la persona otra. Probablemente se necesiten dos perfiles.
5. **Latencia de `cap.set`:** algunos drivers tardan varios cuadros en aplicar; el control debe ser lento (2-4 Hz), no cada cuadro.

## 6. Recomendación para BodySense

Empezar por **3.1 + 3.3 + 3.4** combinados, que es exactamente el diseño industrial probado:

```
cada 0.5 s:
  brillo = media de gris en ROI (caja de MediaPipe, o centro si no hay persona)
  si brillo < 110: subir exposure un paso (max limite, max cambio/seg)
  si exposure al tope y sigue oscuro: subir brightness
  si brillo > 130: bajar en orden inverso
  aplicar a ambas camaras con la misma banda objetivo
```

Parámetros iniciales: banda 110-130, paso exposure 1.0 (escala log de DSHOW), techo exposure para no meter blur con persona en movimiento.

## 7. Experimentos para validar

1. Escena con luz cambiante (prender/apagar luz): fijo vs auto driver vs algoritmo. Métrica: desviación del brillo + % cuadros con detección MediaPipe en ambas cámaras.
2. Impacto en calibración: RMS estéreo con cada modo.
3. Estéreo parejo: diferencia de brillo cam0 vs cam1 en el tiempo.

## 8. Fuentes

- Basler docs: Exposure Auto, Gain Auto, Auto Function Profile, Auto Function ROI.
- Teledyne: Using Auto Exposure (application note).
- Schulz et al.: Using brightness histogram to perform optimum auto exposure.
- Zhang et al.: Active Exposure Control for Robust Visual Odometry in HDR Environments (ICRA 2017).
- Control de exposición por aprendizaje reforzado (arXiv 2404.01636, 2024).
- OpenCV: `CAP_PROP_AUTO_EXPOSURE`, issue opencv/opencv#9738; foros ELP / ArduCam / Pi Camera sobre exposición manual.

---

# Investigación: cámara Android + cámara externa USB en Python/OpenCV

**Fecha:** 28 sep 2026
**Autor:** Gustavo Estrada
**Proyecto:** BodySense V1 — sistema estéreo con cámaras mixtas (Android + USB)

## 1. Pregunta

¿Podemos usar la cámara de un celular Android como una de las cámaras del sistema estéreo, junto con una cámara externa USB, sin problemas de permisos ni sincronización?

Contexto: el plan original era usar 2 webcams USB. Hubo un cambio: ahora se quiere usar 1 cámara Android (como webcam) + 1 cámara externa USB. La investigación cubre cómo hacerlo en Python/OpenCV, qué problemas pueden surgir y cómo resolverlos.

## 2. Opciones para usar cámara Android como webcam

### 2.1 DroidCam (recomendado)

- **Qué es:** app que convierte el teléfono en webcam para PC
- **Conexión:** WiFi o USB
- **Gratis:** sí, en definición estándar (480p/720p)
- **Pago:** DroidCamX para 1080p y sin marca de agua
- **Cómo funciona:** instala un driver de cámara virtual en Windows. OpenCV lo ve como una webcam normal.
- **Ventajas:** funciona con cualquier programa que use cámaras, fácil de configurar
- **Desventajas:** la versión gratis tiene resolución limitada; WiFi añade latencia

### 2.2 Iriun Webcam

- **Qué es:** alternativa similar a DroidCam
- **Conexión:** WiFi o USB
- **Gratis:** sí, con resolución limitada
- **Ventajas:** buena calidad de imagen, funciona bien con OpenCV
- **Desventajas:** similar a DroidCam en limitaciones

### 2.3 Otras opciones (no recomendadas para este proyecto)

- **IP Webcam:** requiere stream RTSP, más complejo de integrar con OpenCV
- **DroidCam OBS Camera:** solo para OBS, no como webcam del sistema
- **scrcpy:** espejo de pantalla, no es webcam

### Recomendación

**DroidCam por USB** es la mejor opción para BodySense:
- Menor latencia que WiFi
- OpenCV lo detecta como cámara normal
- No requiere configuración de red
- La versión gratis es suficiente para 720p

## 3. Cómo funciona en OpenCV

### 3.1 Enumeración de cámaras

En Windows, OpenCV usa DirectShow (DSHOW) o Media Foundation (MSMF) para acceder a cámaras. Los índices (0, 1, 2...) pueden cambiar entre reinicios.

```python
import cv2

# Listar cámaras disponibles
def listar_camaras(max_camaras=10):
    camaras = []
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                nombre = cap.getBackendName()
                camaras.append((i, nombre))
            cap.release()
    return camaras

print(listar_camaras())
```

### 3.2 Abrir cámaras por índice (inestable)

```python
# Esto PUEDE FALLAR si Windows reordena las cámaras
cap_usb = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # cámara USB
cap_android = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # DroidCam
```

### 3.3 Abrir cámaras por nombre (recomendado)

```python
# Más estable: buscar por nombre
def abrir_por_nombre(nombre_buscar, max_camaras=10):
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # En Windows, el nombre no se obtiene directamente
            # pero se puede verificar con resolución o propiedades
            ret, frame = cap.read()
            if ret:
                # Verificar si es la cámara correcta por resolución
                h, w = frame.shape[:2]
                if nombre_buscar == "USB" and w >= 1280:
                    return cap
                elif nombre_buscar == "Android" and w < 1280:
                    return cap
        cap.release()
    return None
```

### 3.4 Configurar formato de video

```python
# Configurar ambas cámaras al mismo formato
FORMATO = cv2.CAP_PROP_FOURCC
codec = cv2.VideoWriter_fourcc(*'MJPG')

for cap in [cap_usb, cap_android]:
    cap.set(FORMATO, codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
```

## 4. Problemas de permisos en Windows

### 4.1 Permisos de privacidad de Windows

Windows 10/11 tiene configuraciones de privacidad que pueden bloquear el acceso a cámaras:

**Ruta:** Configuración → Privacidad y seguridad → Cámara

**Solución:** Asegurarse de que:
1. "Acceso a la cámara" esté activado
2. "Permitir que las apps accedan a la cámara" esté activado
3. Python esté en la lista de apps permitidas

### 4.2 Drivers de DroidCam/Iriun

DroidCam e Iriun instalan drivers de cámara virtual. Estos drivers:
- Aparecen como "DroidCam Source" o "Iriun Webcam" en el administrador de dispositivos
- Pueden requerir permisos de administrador para instalarse
- Pueden entrar en conflicto con otros drivers de cámara virtual

**Solución:** 
- Instalar como administrador
- Verificar en "Administrador de dispositivos" → "Dispositivos de imagen" que aparezcan correctamente
- Si hay conflicto, desinstalar otros drivers de cámara virtual (OBS Virtual Camera, ManyCam, etc.)

### 4.3 Permisos de Python

En Windows, Python no necesita permisos especiales para acceder a cámaras, pero:
- Si se ejecuta desde un entorno virtual, hereda los permisos del usuario
- Si se ejecuta como servicio, puede necesitar configuración adicional

## 5. Problemas potenciales con dos cámaras (Android + USB)

### 5.1 Sincronización (el mayor problema)

**Problema:** Las cámaras no están sincronizadas por hardware. La cámara Android tiene latencia de red/procesamiento, la USB no.

**Impacto en BodySense:** La triangulación estéreo requiere que ambas imágenes sean del mismo momento. Si una cámara va 200ms detrás de la otra, la profundidad calculada será incorrecta.

**Soluciones:**
1. **Sincronización por software con `grab()` + `retrieve()`:**
   ```python
   # Grabar primero, después decodificar
   cap_usb.grab()
   cap_android.grab()
   ret1, frame_usb = cap_usb.retrieve()
   ret2, frame_android = cap_android.retrieve()
   ```
   Esto reduce la diferencia de tiempo pero no la elimina.

2. **Usar DroidCam por USB (no WiFi):** reduce la latencia de ~200ms a ~50ms

3. **Aceptar la asincronía:** si la persona no se mueve rápido, la diferencia es despreciable

### 5.2 Índices de cámara inestables

**Problema:** Windows puede reordenar las cámaras entre reinicios. La cámara 0 de hoy puede ser la 1 de mañana.

**Solución:** 
- Enumerar cámaras al inicio del programa
- Identificar cada cámara por resolución o propiedades únicas
- No hardcodear índices

### 5.3 Latencia de red (WiFi)

**Problema:** DroidCam/Iriun por WiFi tiene 100-300ms de latencia.

**Impacto:** La imagen de la cámara Android va "atrasada" respecto a la USB.

**Solución:** 
- **Usar conexión USB** (recomendado): latencia ~50ms
- Si no es posible, usar WiFi 5GHz (no 2.4GHz)
- Reducir resolución para menor latencia

### 5.4 Formato de video diferente

**Problema:** La cámara Android puede dar formato YUV o MJPG, la USB puede dar MJPG o H264. Si los formatos son diferentes, el procesamiento puede fallar o ser inconsistente.

**Solución:**
- Forzar ambas cámaras al mismo formato (MJPG recomendado)
- Verificar con `cap.get(cv2.CAP_PROP_FOURCC)` qué formato usa cada una

### 5.5 Ancho de banda USB

**Problema:** Dos cámaras USB pueden saturar el bus USB, especialmente si son USB 2.0.

**Impacto:** Frames perdidos, desconexiones, imagen corrupta.

**Solución:**
- Conectar las cámaras a puertos USB diferentes (idealmente USB 3.0)
- Reducir resolución o FPS si hay problemas
- Usar un hub USB con alimentación propia

### 5.6 Resolución diferente

**Problema:** La cámara Android puede ser 720p, la USB 1080p. Para estéreo, ambas deben tener la misma resolución.

**Solución:**
- Configurar ambas a la misma resolución (720p recomendado para equilibrio calidad/rendimiento)
- Si no se puede, redimensionar una de las imágenes

## 6. Código de ejemplo completo

```python
import cv2
import numpy as np

class SistemaCamarasMixto:
    def __init__(self):
        self.cap_usb = None
        self.cap_android = None
    
    def iniciar(self):
        # Buscar cámaras por resolución
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    # Cámara USB suele ser 1080p, Android 720p
                    if w >= 1280 and self.cap_usb is None:
                        self.cap_usb = cap
                        print(f"Cámara USB encontrada en índice {i}")
                    elif w < 1280 and self.cap_android is None:
                        self.cap_android = cap
                        print(f"Cámara Android encontrada en índice {i}")
                    else:
                        cap.release()
                else:
                    cap.release()
            
            if self.cap_usb and self.cap_android:
                break
        
        if not self.cap_usb or not self.cap_android:
            raise RuntimeError("No se encontraron ambas cámaras")
        
        # Configurar formato
        self._configurar_formato(self.cap_usb)
        self._configurar_formato(self.cap_android)
    
    def _configurar_formato(self, cap):
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, codec)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    def leer_frames(self):
        # Sincronización por software
        self.cap_usb.grab()
        self.cap_android.grab()
        
        ret1, frame_usb = self.cap_usb.retrieve()
        ret2, frame_android = self.cap_android.retrieve()
        
        if not ret1 or not ret2:
            return None, None
        
        return frame_usb, frame_android
    
    def liberar(self):
        if self.cap_usb:
            self.cap_usb.release()
        if self.cap_android:
            self.cap_android.release()

# Uso
sistema = SistemaCamarasMixto()
sistema.iniciar()

while True:
    frame_usb, frame_android = sistema.leer_frames()
    if frame_usb is None or frame_android is None:
        break
    
    # Procesar frames...
    cv2.imshow("USB", frame_usb)
    cv2.imshow("Android", frame_android)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sistema.liberar()
cv2.destroyAllWindows()
```

## 7. Recomendaciones finales para BodySense

1. **Usar DroidCam por USB** (no WiFi) para la cámara Android
2. **Configurar ambas cámaras a 720p MJPG 30fps** para consistencia
3. **Enumerar cámaras al inicio**, no hardcodear índices
4. **Usar `grab()` + `retrieve()`** para sincronización por software
5. **Verificar permisos de Windows** antes de ejecutar
6. **Probar con `diagnostico_cam.py`** para verificar que ambas cámaras funcionan antes de integrar
7. **Considerar la latencia**: si la persona se mueve rápido, la asincronía puede afectar la triangulación. En ese caso, usar dos cámaras USB es mejor.

## 8. Fuentes

- DroidCam: https://www.dev47apps.com/
- Iriun Webcam: https://iriun.com/
- OpenCV VideoCapture docs: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- OpenCV issues sobre múltiples cámaras: https://github.com/opencv/opencv/issues?q=is%3Aissue+multiple+cameras

---

# Investigación: ajuste de tiempo de exposición en cámara monocromática (B&N) USB

**Fecha:** 13 oct 2026
**Autor:** Gustavo Estrada
**Proyecto:** BodySense V1 — control fino de shutter en cámara monocromática USB

## 1. Pregunta

¿Por qué no se puede modificar el tiempo de exposición de la cámara monocromática (blanco y negro) USB desde Python/OpenCV, y qué estrategias existen para tomar control manual del shutter (DirectShow, SDK del fabricante, libuvc, ffmpeg, registro de Windows, hardware)?

Contexto: hoy se intentó con `cap.set(cv2.CAP_PROP_EXPOSURE)` y `CAP_PROP_AUTO_EXPOSURE` bajo DSHOW/MSMF, sin éxito: el valor no se aplica o el driver lo ignora. La imagen queda inestable para MediaPipe y para el estéreo. Necesitamos shutter manual para estabilidad.

## 2. Lo que se sabe hoy

- Las webcams UVC exponen controles estándar (exposure, auto-exposure, gain) vía `VideoProcAmp`/`CameraControl` (DirectShow) o Media Foundation.
- OpenCV pasa esos controles por `cv::VideoCapture::set`, pero muchos drivers de cámaras monocromáticas / industriales baratas solo aceptan auto-exposición o ignoran el set.
- En nuestro caso: cámara B&N USB. Ningún método probado en Python/OpenCV cambia el shutter. Posibles causas a confirmar:
  1. El control se llama diferente en el driver (nombre/propiedad fuera del enum de OpenCV).
  2. La cámara es UVC-compliant solo parcialmente: expone auto-exposure pero no exposure_absolute.
  3. El driver requiere comunicación directa (ioctl/Vendor requests) fuera de OpenCV.
  4. Hay un obturador físico (rotación mecánica) que OpenCV no puede tocar.
  5. El backend DSHOW/MSMF está aplicando el valor a otra cámara (índice equivocado).

## 3. Estrategias a investigar

### 3.1 DirectShow crudo (fuera de OpenCV)
Usar `pygrabber` / `comtypes` / `win32com` para acceder a `IAMCameraControl` (exposure, iris, focus) e `IAMVideoProcAmp` (brightness, contrast, hue, saturation, sharpness, gamma, white balance, gain, backlight compensation, powerline_frequency) y llamar `SetProperty` directamente.
- Pros: sin dependencia del firmware; control completo de propiedades.
- Contras: más código COM; propiedades disponibles dependen del driver.

### 3.2 SDK del fabricante
Si la cámara es Basler (ace), Allied Vision (Vimba), FLIR (Spinnaker), ELP/ArduCam, etc., usar su SDK: pylon/Vimba/Spinnaker o librería del fabricante ofrece control absoluto del shutter, ganancia, línea de trigger.
- Pros: control real, documentado, estable.
- Contras: depende de que tengamos el modelo exacto; SDKs distintos por marca.

### 3.3 libuvc / pyusb
Hablar directamente con el descriptor UVC por USB (Vendor requests) para leer/escribir `Exposure Time Absolute` (control `CT_EXPOSURE_TIME_ABSOLUTE_CONTROL`) y `Auto Exposure Mode`.
- Pros: bypass completo del driver de Windows.
- Contras: en Windows necesita driver WinUSB/libusb y permisos; más bajo nivel.

### 3.4 ffmpeg / PyAV
Usar ffmpeg con `-exposure` / opciones de cámara o `pyav` para abrir el dispositivo y aplicar opciones antes de decodificar.
- Pros: a veces ffmpeg soporta más controles que OpenCV.
- Contras: configuración frágil; no resuelve si el driver no expone el control.

### 3.5 Configuración del driver / registro de Windows
Algunos drivers de cámara guardan el shutter en el registro o en un panel de control (Administrador de dispositivos → propiedades del dispositivo, app del fabricante). Ajustar ahí y que OpenCV solo lea frames.
- Pros: simple si el driver lo permite.
- Contras: no es programático; no sirve para auto-adaptativo en tiempo real.

### 3.6 Obturador físico / hardware
Verificar si la cámara tiene diafragma/obturador mecánico que se mueve por botón o rueda. Si es así, OpenCV nunca lo va a controlar.
- Pros: cero software.
- Contras: no automatizable.

### 3.7 Identificación previa (crítica)
Antes de cualquier estrategia, saber exactamente qué cámara es:
- `cap.getBackendName()`, `diagnostico_cam.py` (brillo, resolución, backend), Administrador de dispositivos (nombre, ID de hardware, driver), ID USB (VID/PID) en `usbview`/Administrador de dispositivos.

## 4. Plan de experimentos

0. **Primer paso recomendado:** identificación exacta (marca/modelo/driver/VID/PID) + backend con `diagnostico_cam.py`. Hasta saber eso no se elige entre SDK/libuvc/ffmpeg/registro.
1. En `diagnostico_cam.py`: imprimir backend, nombre del dispositivo, rango aceptado por `CAP_PROP_EXPOSURE` (lee antes/después de `set`) y VID/PID si es posible.
2. Script de prueba DirectShow crudo: enumerar propiedades `IAMCameraControl`/`IAMVideoProcAmp` disponibles para esa cámara y guardarlas.
3. Si DirectShow crudo no expone exposure: probar SDK del fabricante (si se identifica marca) o libuvc.
4. Comparar resultado vs cámara USB normal (color) para confirmar que el problema es específico de la monocromática.
5. Decidir estrategia final y si la cámara B&N se mantiene en BodySense o se reemplaza.

## 5. Pendientes / tareas

- [ ] Capturar marca, modelo, VID/PID y driver de la cámara B&N.
- [ ] Actualizar `diagnostico_cam.py` para imprimir backend e info del dispositivo.
- [ ] Crear `test_mono_exposure.py` con estrategias DirectShow crudo.
- [ ] Evaluar SDK/libuvc si toca.
- [ ] Documentar hallazgos y decisión final aquí.

## 6. Fuentes

- Microsoft Docs: `IAMCameraControl`, `IAMVideoProcAmp`, DirectShow Device Control.
- OpenCV `VideoCapture` properties y backends (`CAP_DSHOW`, `CAP_MSMF`).
- libuvc: UVC control requests (`CT_EXPOSURE_TIME_ABSOLUTE_CONTROL`, `CT_AUTO_EXPOSURE_MODE_CONTROL`).
- Documentación de SDKs: Basler pylon, Allied Vision Vimba, FLIR Spinnaker, ELP/ArduCam.

## 7. Hallazgos de la prueba (13 oct 2026)

**Cámara identificada:** externa monocromática `USB Camera`, VID `0C45` PID `6366` (chip SONiX), DSHOW index 0.  
Verificado por marca: frame con canales B/G/R iguales (`between_std=0.00`) en DSHOW y ventana visual B&N.

**Resultados del sondeo (`diagnostico_cam.py`, `cmp_mono.py`, `test_mono_exposure.py`, `ae_matrix.py`, `scan_brightness.py`, `iam_exposure.py`):**
1. **MSMF (idx0):** `cap.set(CAP_PROP_*)` devuelve `ok=True` pero ningún valor cambia (Exposure/Brightness/Gain todos ignorados). Conclusión: MSMF es inútil para controlar esta cámara.
2. **DSHOW (idx0):** `CAP_PROP_EXPOSURE` sí cambia el `readback`, pero **no altera el brillo de la imagen** (media ~48-51 para -3..-9). `CAP_PROP_AUTO_EXPOSURE` se queda en `-1.0` y no responde. El driver no aplica realmente el shutter.
3. **DirectShow crudo:** el filtro `USB Camera` **no soporta `IAMCameraControl`** (`QueryInterface` devuelve "Interfaz no compatible"). Por tanto el driver **no expone el control de exposición a nivel UVC/DirectShow**; el `CAP_PROP_EXPOSURE` de OpenCV es un valor "fantasma".
4. **CAP_PROP_BRIGHTNESS (DSHOW):** sí funciona de forma lineal y real (media 33.7 a -64 hasta 118.8 a +64). `CONTRAST` y `GAIN` también se aplican.

**Conclusión:** para esta cámara (0C45:6366) **no hay forma de modificar el tiempo de exposición desde Python**, ni con OpenCV ni con DirectShow crudo. El único control real por software es Brightness (y Contrast/Gain), que es ganancia digital, no shutter.

**Opciones reales:**
- **A (software):** usar `Brightness` como pseudo-exposición en el algoritmo propio de BodySense (banda objetivo, amortiguamiento, límites) y documentar que es ganancia digital.
- **B (hardware):** ajustar con iluminación/shutter físico o cambiar a una cámara UVC-compliant con SDK real (MindVision, ELP/ArduCam global shutter monocromático) que soporte `Exposure Time Absolute`.
- **C (bajo nivel):** intentar `libuvc`/pyusbVendor requests (requiere WinUSB/libusb y reinstalar driver) con baja probabilidad de éxito en este chip (0C45) y riesgo de romper otras apps.

**Decisión sugerida para V1:** opción A para no bloquear el proyecto; dejar la opción B como mejora si la estabilidad lo exige.

**Implementado (13 oct 2026):** `bodySense/control_brillo.py` — clase `ControladorBrillo` con algoritmo objetivo/limites/zona muerta/amortiguacion sobre `CAP_PROP_BRIGHTNESS`, lista para integrar a `capturar.py`/`main.py`.
