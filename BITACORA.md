# Bitácora BodySense — V1

**Fecha:** 21 sep 2026
**Autor:** Gustavo Estrada
**Rama:** `fix/charuco-camaras-imports`

Empiezo esta bitácora para llevar un registro ordenado del proyecto y que cualquier agente (o yo en el futuro) entienda el estado actual sin tener que leer todo el código.

---

## 1. Qué es BodySense V1

Un sistema estéreo con 2 webcams para ver el esqueleto en 3D y apoyar el ejercicio físico. La idea: la persona hace sentadillas y el sistema indica si la técnica es correcta.

Flujo completo (todo se corre desde **un solo programa**):

```
python inicio.py
```

Te abre el menú **BodySense V1** (ventana clara con 4 tarjetas en cuadrícula 2x2):

| Tarjeta | Qué hace |
|---|---|
| 1 Tablero | Eliges ajedrez o charuco, ajustas medidas, botón Medir con cámara |
| 2 Capturar | Tomas 30 pares de fotos del tablero (botón Capturar), Probar y Borrar |
| 3 Calibrar | Botón Generar, te dice FOV / baseline y si está creíble |
| 4 Sesión | Pones tu usuario, palomeas si eres entrenador, botones Grabar referencia / Practicar / Solo ver |

Después cada script se abre en su ventanita OpenCV. El menú espera a que termine y se actualiza solo.

---

## 2. Cómo funciona por dentro (sin tanto rollo)

- **Calibración:** fotos del tablero con las 2 cámaras → `generar_calibracion.py` (le pasas `charuco` si es ese) → guarda `calibracion/stereo.npz` (matrices, distorsión, R, T). Si sale muy mal, **la rechaza** y te sale pop-up rojo diciéndote QUÉ hiciste mal (pocas inclinaciones, tablero muy chico, cámaras chuecas, etc). No guarda nada feo.
- **Captura:** `capturar.py [ajedrez|charuco]`, cámaras **0 y 1** (ojo: el 2 no existe en DSHOW, y abrir dos veces la misma cámara duplica la imagen y falsea todo).
- **Tiempo real:** `main.py --usuario Juan [--referencia | --sin-grabar]`. Cuenta 3-2-1 sin grabar, luego **ESPACIO empieza/pausa** (lo registrado solo entra mientras se graba), ESC sale. Triangula con MediaPipe de ambas cámaras y dibuja el avatar en OpenGL.
- **Anti-chicle:** si te sales del cuadro o haces algo raro, el avatar ya no se estira horrible. Se congela la última pose buena en gris + aviso `VUELVE AL CUADRO`. Candados: visibilidad > 0.5, mínimo 8 puntos, volumen 0.5-5 m, huesos que no se alarguen 3x de golpe.
- **Base de datos:** `registro.py` → SQLite `bodysense.db` + CSVs por sesión. Tablas `usuarios, sesiones, metricas`. Métricas invariantes a estatura (esa es la clave): inclinación tronco, desnivel hombros, torsión torso, adelanto cabeza. El entrenador graba referencia (`es_referencia=1`) y el alumno se compara contra eso. `python registro.py --check` corre el demo con cuerpos sintéticos.

---

## 3. Decisiones que tomamos (para no revertirlas sin pensar)

1. **Charuco vs ajedrez:** ajedrez exige ver el tablero COMPLETO en ambas cámaras, en estéreo se pierden muchos pares. Charuco (defaults `5x7, 30/22mm, DICT_5X5_100`) tolera oclusión. Pero ojo: pedir mínimo 10 esquinas comunes por par, con 4 la calibración sale inestable.
2. **opencv-contrib-python**, no `opencv-python` solo. Charuco vive en contrib.
3. **Un solo entry-point** (`inicio.py`), no 7 scripts sueltos. Los scripts viejos quedan como módulos con `main()` para debug, pero el usuario solo ve el menú.
4. **Rechazo duro de calibración** en vez de guardar con warning. En software de ejercicio no se tolera distorsión fea.
5. **Referencia = rango (mediana + p10/p90)**, no foto congelada. Así lo parado se vuelve outlier y no contamina.

---

## 4. Hallazgos de esta sesión (21 sep 2026)

- El `stereo.npz` viejo estaba sospechoso: `fx cam0 ~1622` vs `cam1 ~1004` (misma cámara deberían parecerse) y rotación ~26° (mucho para estéreo paralelo). Por eso el avatar a veces se veía raro.
- `main.py` y `capturar.py` abrían la **misma cámara 1 dos veces** (cambio sin commitear). Eso no es estéreo, es la misma imagen duplicada → profundidad falsa. Se corrigió a 0 y 1.
- Imports rotos: `from Herramientas` y `from Graficos` no existían; lo real es `utils.Herramientas` y `visualizacion` (stubs). Ya quedó.
- `inicio.py` lanzaba `menu.py` y `probar_render.py` que no existen → botones muertos. Se quitaron del paso 4, ahora es Grabar/Practicar/Solo ver.
- `inicio.py` buscaba scripts en `utils/` (`H.BASE`) → nunca los encontraba. Ahora usa `buscar_archivo_desesperadamente` y acepta args (`capturar.py [tipo]`, `generar... [charuco]`).
- Las `capturas/` actuales son 30 pares de **ajedrez**, no hay ni una charuco → la comparación A/B ajedrez vs charuco queda pendiente (imprimir charuco y capturar mismos poses).
- Menú rediseñado: fondo marfil con tarjetas blancas, título serif **BodySense V1**, iconos grandes dibujados, cuadrícula 2x2 separada del título, tarjeta bajo el cursor con elevación, sombra y borde vino.
- V1 de sesiones probado sin cámaras: referencia 30° vs alumno 45° da dif +15° correcta. `registro --check` pasa.

---

## 5. Pendientes

- [ ] Imprimir charuco real con los valores por defecto y capturar 20-30 pares → comparar RMS ajedrez vs charuco.

- [ ] Imprimir charuco real con los defaults y capturar 20-30 pares → correr comparador RMS ajedrez vs charuco.
- [ ] Probar el menú nuevo con cámaras de verdad (Probar + Capturar + Generar).
- [ ] Grabar 1 referencia de sentadilla con entrenador y 1 práctica de alumno, ver el diff en consola.
- [ ] Contador de reps automático (por ángulo de rodilla) — en V1 es manual con ESPACIO, no olvidar.
- [ ] Commitear la rama `fix/charuco-camaras-imports` (quedó sin commit a propósito hoy).

---

*Nota para agentes futuros: si rompes algo, corre `python -m py_compile` en `main.py, inicio.py, capturar.py, registro.py` y `python registro.py --check` antes de decir que está listo. Y no abras dos veces la misma cámara, en serio.*

---

## 6. Línea abierta: auto-exposición en tiempo real (21 sep 2026)

**Pregunta:** ¿podemos hacer un algoritmo que ajuste Exposure/Brightness/Contrast solo, según el entorno?

**Contexto del hallazgo:** la integrada HP daba negro total (brillo 0.0) y resultó ser el shutter físico. Al abrirlo, `diagnostico_cam.py` reportó brillo ~120 en DSHOW. Eso mostró que el brillo medio del cuadro es una buena señal para decidir valores.

**Idea V1 del algoritmo:**
1. Medir brillo medio del cuadro en gris cada ~0.5 s (no cada cuadro, para no temblar).
2. Objetivo: banda 110-130. Si está fuera, mover Exposure un paso (0.5-1.0) hacia el objetivo; si Exposure toca su límite, mover Brightness.
3. Límites duros por parámetro (los rangos de `test_cam_controls.py`) y cambio máximo por segundo, para no oscilar con luces bruscas.
4. Por cámara independiente (cam0 y cam1 viven en luz distinta).

**Experimentos pendientes:**
- [ ] Fijo vs auto del driver vs algoritmo propio, misma escena con luz cambiante.
- [ ] Métrica: estabilidad del brillo + RMS de calibración con cada modo.
- [ ] Definir si vive en `test_cam_controls.py` (herramienta) o se integra a `capturar.py`/`main.py`.

---

## 7. Línea abierta: cámara Android + cámara USB (28 sep 2026)

**Cambio de planes:** el plan original era 2 webcams USB. Ahora se quiere usar **1 cámara Android (como webcam) + 1 cámara externa USB**.

**Investigación:** esta semana haré la investigación sobre:
- Cómo usar la cámara de un celular Android como webcam en Python/OpenCV
- Comandos para acceder a ambas cámaras sin problemas de permisos
- Problemas que podrían surgir al tener dos cámaras de tipos diferentes
- Soluciones y recomendaciones

**Estado:** investigación en curso. Ver `INVESTIGACION.md` para el documento completo.

---

## 8. Línea abierta: tiempo de exposición en cámara monocromática B&N USB (13 oct 2026)

**Hallazgo:** la cámara blanco y negro (monocromática) conectada por USB **no acepta el ajuste de `Exposure` desde Python/OpenCV** por ninguno de los métodos probados (`cap.set(CAP_PROP_EXPOSURE)`, `CAP_PROP_AUTO_EXPOSURE`, backend DSHOW/MSMF, etc.). El tiempo de obturación parece quedar bloqueado en lo que decida el driver/firmware. El auto del driver deja la imagen inestable para MediaPipe y el estéreo.

**Meta:** investigar estrategias para controlar manualmente el tiempo de exposición de esa cámara y documentarlas para decidir si se integra a BodySense (captura / main / diagnostico).

**Orden sugerido (primer paso recomendado):** 1) identificación exacta de marca/modelo/driver/VID/PID + backend con `diagnostico_cam.py`; 2) DirectShow crudo (`IAMCameraControl`/`IAMVideoProcAmp`) para confirmar si el control existe; 3) solo si falla, SDK del fabricante o libuvc; 4) último recurso: registro/obturador físico.

**Plan de investigación (esta semana):**
- [ ] Identificar marca/modelo y chip/driver exactos de la cámara B&N (Administrador de dispositivos, `diagnostico_cam.py`, `cap.getBackendName()`).
- [ ] Probar control vía DirectShow crudo (`IAMCameraControl` / `IAMVideoProcAmp`) fuera de OpenCV.
- [ ] Probar SDK del fabricante (pylon, ELP/ArduCam, etc.) y libuvc/pyusb.
- [ ] Probar alternativas: ffmpeg/pyav, configuración del driver en el registro de Windows, ajuste en hardware (shutter físico).
- [ ] Documentar qué estrategias funcionan en `INVESTIGACION.md` (tercer bloque).
- [ ] Decidir si esta cámara se usa como una de las dos o solo para pruebas, según control de exposición disponible.

**Estado:** cerrada (13 oct 2026). Ver `INVESTIGACION.md` (tercer bloque, §7 Hallazgos).

**Resumen del hallazgo:**
- Cámara mono identificada: `USB Camera`, VID `0C45` PID `6366` (DSHOW idx0).
- MSMF (idx0) no controla nada (Exposure/Brightness ignorados).
- DSHOW `CAP_PROP_EXPOSURE` cambia el readback pero **no altera la imagen**; además el filtro no soporta `IAMCameraControl` (DirectShow crudo) → no es control real de shutter.
- **Solo `CAP_PROP_BRIGHTNESS`** (y Contrast/Gain) funcionan de verdad en DSHOW.
- Conclusión: **con esta cámara no se puede modificar el tiempo de exposición desde Python**. Usar Brightness como pseudo-exposición (opción A) o cambiar de cámara (opción B).

**Archivos de prueba generados:** `diagnostico_cam.py`, `preview_camaras.py`, `cmp_mono.py`, `test_mono_exposure.py`, `ae_matrix.py`, `scan_brightness.py`, `iam_exposure.py` (en `bodySense/`).

**Implementado (opción A):** `bodySense/control_brillo.py` (`ControladorBrillo`) listo para integrar.
