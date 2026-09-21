"""Menu de BodySense: el flujo de calibracion en cuatro pasos.

Muestra en que punto estas leyendo el disco (cuantos pares hay, si existe la
calibracion y si es creible), deja editar la config del tablero, y lanza cada
script cuando toca.

Se hizo con pygame y no con tkinter por dos razones: pygame ya es dependencia
del proyecto (main.py lo usa) y dibuja sus propios pixeles, asi que se ve
igual en macOS y en Windows. El tkinter del macOS de este proyecto es Tcl/Tk
8.5.9, cuyo tema aqua es de 2009.

    python3 inicio.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pygame

import utils.Herramientas as H

# ==========================================
# APARIENCIA - estilo menu 2x2 con tarjetas grandes
# ==========================================
ANCHO = 1000
ALTO = 720
MARGEN = 24
GAP = 16
CARD_W = (ANCHO - MARGEN * 2 - GAP) // 2
CARD_H = 250
Y_TITULO = 100         # grilla mas abajo, separada del titulo

# Estilo claro como la foto de referencia: fondo marfil, tarjetas blancas,
# titulo serif vino, iconos grandes dibujados.
FONDO = (244, 241, 235)
TARJETA = (255, 255, 255)
BORDE = (225, 218, 208)
TINTA = (62, 32, 32)
TINTA_SUAVE = (130, 110, 110)
TINTA_TENUE = (170, 155, 150)
ACENTO = (122, 30, 35)
ACENTO_CLARO = (150, 45, 50)
VERDE = (63, 160, 120)
AMBAR = (200, 140, 60)
ROJO = (190, 70, 60)

# Titulo serif como la foto, cuerpo sans limpio.
FAM_TITULO = "georgia,timesnewroman,times,serif"
FAM = "segoeui,helveticaneue,helvetica,arial,dejavusans"
FAM_MONO = "menlo,consolas,dejavusansmono,couriernew,monospace"

BASE_PROYECTO = os.path.dirname(H.BASE)
ARCHIVO_AVATAR = os.path.join(BASE_PROYECTO, "avatar_config.json")
ARCHIVO_STEREO = os.path.join(H.CARPETA_CALIBRACION, "stereo.npz")

# Minimo de pares que generar_calibracion.py exige para siquiera intentarlo.
MIN_PARES = 10


def leer_rechazo():
    """Texto del ultimo rechazo de calibracion, o None si no hay."""
    ruta = os.path.join(H.CARPETA_CALIBRACION, "ultimo_rechazo.txt")
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def limpiar_rechazo():
    try:
        os.remove(os.path.join(H.CARPETA_CALIBRACION, "ultimo_rechazo.txt"))
    except OSError:
        pass


# ==========================================
# ESTADO LEIDO DEL DISCO
# ==========================================
def leer_avatar():
    """Nombre y color del avatar, o los defaults si no hay archivo."""
    try:
        with open(ARCHIVO_AVATAR, encoding="utf-8") as f:
            d = json.load(f)
        color = d.get("color_piel", "#3498db").lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        return str(d.get("nombre", "Jugador 1")), rgb
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "Jugador 1", (52, 152, 219)


def leer_calibracion():
    """Resumen del stereo.npz, o None si no hay.

    El ancho de imagen no se guarda en el .npz, asi que se estima como cx*2:
    el punto principal cae cerca del centro del sensor. Solo se usa para el
    campo de vision que se muestra, no para calcular nada.
    """
    if not os.path.exists(ARCHIVO_STEREO):
        return None
    try:
        d = np.load(ARCHIVO_STEREO, allow_pickle=True)
        fovs, anchos = [], []
        for clave in ("mtx1", "mtx2"):
            m = d[clave]
            ancho = 2 * float(m[0, 2])
            anchos.append(ancho)
            fovs.append(H.fov_grados(float(m[0, 0]), ancho))
        bajo, alto = H.RANGO_FOV_CREIBLE
        return {
            "fecha": time.localtime(os.path.getmtime(ARCHIVO_STEREO)),
            "fovs": fovs,
            "anchos": anchos,
            "baseline": float(np.linalg.norm(d["T"])),
            "creible": all(bajo < f < alto for f in fovs),
        }
    except Exception as e:                      # npz truncado, clave faltante
        return {"error": str(e)}


def probar_camaras():
    """Abre las dos camaras, lee un cuadro y reporta. Tarda ~1 s.

    ponytail: bloquea la ventana ese segundo. Hacerlo en un hilo pediria
    sincronizar el acceso a OpenCV por un mensaje de una linea.
    """
    import cv2

    partes = []
    for i in (0, 1):
        cap = cv2.VideoCapture(i, H.BACKEND_CAMARA)
        ok, frame = (False, None)
        if cap.isOpened():
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            partes.append(f"cam{i}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            partes.append(f"cam{i}: NO responde")
    return "   ".join(partes)


# ==========================================
# UI DE MODO INMEDIATO
# ==========================================
class UI:
    """Cada cuadro se redibuja entero y los widgets reportan su click.

    Sin estado de widgets: no hay que registrar botones ni mantener una lista
    paralela que se desincronice de lo que se ve.
    """

    def __init__(self, pantalla):
        self.p = pantalla
        self.click = None
        self.mouse = (0, 0)
        self.f_titulo = pygame.font.SysFont(FAM_TITULO, 34, bold=True)
        self.f_sub = pygame.font.SysFont(FAM, 14)
        self.f_paso = pygame.font.SysFont(FAM, 17, bold=True)
        self.f = pygame.font.SysFont(FAM, 13)
        self.f_bold = pygame.font.SysFont(FAM, 13, bold=True)
        self.f_chico = pygame.font.SysFont(FAM, 11)
        self.mono = pygame.font.SysFont(FAM_MONO, 13)
        self.mono_chico = pygame.font.SysFont(FAM_MONO, 11)

    # --- primitivas ---
    def texto(self, s, x, y, color=TINTA, fuente=None, derecha=False):
        img = (fuente or self.f).render(str(s), True, color)
        r = img.get_rect()
        setattr(r, "topright" if derecha else "topleft", (x, y))
        self.p.blit(img, r)
        return r

    def tarjeta(self, x, y, w, h):
        # Sombra suave + tarjeta blanca como la foto.
        # Si el cursor esta sobre la card, sombra mas grande = efecto elevado.
        hover = pygame.Rect(x, y, w, h).collidepoint(self.mouse)
        if hover:
            pygame.draw.rect(self.p, (215, 205, 190), (x + 5, y + 9, w, h), border_radius=12)
            pygame.draw.rect(self.p, (222, 212, 198), (x + 2, y + 6, w, h), border_radius=12)
        else:
            pygame.draw.rect(self.p, (230, 222, 210), (x + 3, y + 4, w, h), border_radius=12)
        pygame.draw.rect(self.p, TARJETA, (x, y, w, h), border_radius=12)
        pygame.draw.rect(self.p, ACENTO if hover else BORDE, (x, y, w, h),
                         2 if hover else 1, border_radius=12)

    def icono_grande(self, tipo, cx, cy):
        """Icono grande dibujado (no depende de emojis de la fuente)."""
        vino, gris = ACENTO, (200, 185, 175)
        if tipo == "tablero":
            s, ox, oy = 13, cx - 19, cy - 19
            for r in range(3):
                for c in range(3):
                    col = vino if (r + c) % 2 == 0 else gris
                    pygame.draw.rect(self.p, col, (ox + c * s, oy + r * s, s, s), border_radius=2)
        elif tipo == "captura":
            pygame.draw.rect(self.p, vino, (cx - 26, cy - 14, 52, 32), border_radius=8)
            pygame.draw.circle(self.p, (255, 255, 255), (cx, cy + 2), 11)
            pygame.draw.circle(self.p, vino, (cx, cy + 2), 6)
            pygame.draw.rect(self.p, vino, (cx - 10, cy - 22, 20, 8), border_radius=3)
        elif tipo == "calib":
            pygame.draw.circle(self.p, vino, (cx, cy), 16, 4)
            pygame.draw.circle(self.p, vino, (cx, cy), 5)
            for a in (0, 60, 120, 180, 240, 300):
                import math
                x1 = cx + 16 * math.cos(math.radians(a))
                y1 = cy + 16 * math.sin(math.radians(a))
                x2 = cx + 23 * math.cos(math.radians(a))
                y2 = cy + 23 * math.sin(math.radians(a))
                pygame.draw.line(self.p, vino, (x1, y1), (x2, y2), 4)
        elif tipo == "sesion":
            pygame.draw.circle(self.p, vino, (cx, cy - 10), 10)
            pygame.draw.rect(self.p, vino, (cx - 13, cy + 4, 26, 20), border_radius=8)

    def marca(self, cx, cy, estado):
        """Circulo de estado: 'ok', 'medio', 'falta' o 'error'."""
        color = {"ok": VERDE, "medio": AMBAR, "falta": TINTA_TENUE,
                 "error": ROJO}[estado]
        pygame.draw.circle(self.p, color, (cx, cy), 11, 0 if estado != "falta" else 2)
        if estado == "ok":                       # palomita a dos lineas
            pygame.draw.lines(self.p, TARJETA, False,
                              [(cx - 5, cy), (cx - 1, cy + 4), (cx + 5, cy - 4)], 2)
        elif estado == "medio":
            pygame.draw.circle(self.p, TARJETA, (cx, cy), 4)
        elif estado == "error":
            pygame.draw.line(self.p, TARJETA, (cx - 4, cy - 4), (cx + 4, cy + 4), 2)
            pygame.draw.line(self.p, TARJETA, (cx + 4, cy - 4), (cx - 4, cy + 4), 2)

    def boton(self, rect, etiqueta, activo=True, tono=None):
        """Dibuja el boton y devuelve True si se le hizo click este cuadro."""
        r = pygame.Rect(rect)
        encima = r.collidepoint(self.mouse) and activo
        base = tono or (235, 228, 218)

        if not activo:
            relleno, letra = (240, 236, 230), TINTA_TENUE
        elif tono is ACENTO:
            relleno = ACENTO_CLARO if encima else ACENTO
            letra = (255, 255, 255)
        elif tono is ROJO:
            relleno = (250, 225, 220) if encima else (248, 235, 232)
            letra = ROJO
        else:
            relleno = (250, 246, 240) if encima else base
            letra = TINTA

        pygame.draw.rect(self.p, relleno, r, border_radius=8)
        pygame.draw.rect(self.p, ACENTO if tono is ACENTO else BORDE, r, 1, border_radius=8)

        img = self.f_bold.render(etiqueta, True, letra)
        self.p.blit(img, img.get_rect(center=r.center))
        return bool(activo and self.click and r.collidepoint(self.click))

    def stepper(self, x, y, valor, paso, bajo, alto, texto, ancho_val=62):
        """[-] valor [+]. Devuelve el valor nuevo, recortado al rango."""
        nuevo = valor
        if self.boton((x, y, 26, 24), "−", valor - paso >= bajo):
            nuevo = valor - paso
        img = self.mono.render(texto, True, TINTA)
        self.p.blit(img, img.get_rect(center=(x + 26 + ancho_val // 2, y + 12)))
        if self.boton((x + 26 + ancho_val, y, 26, 24), "+", valor + paso <= alto):
            nuevo = valor + paso
        return min(alto, max(bajo, nuevo)), x + 26 + ancho_val + 26

    def barra(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.p, (235, 228, 218), (x, y, w, h), border_radius=h // 2)
        if frac > 0:
            pygame.draw.rect(self.p, color, (x, y, max(h, int(w * min(1.0, frac))), h),
                             border_radius=h // 2)


# ==========================================
# APLICACION
# ==========================================
class Menu:
    def __init__(self):
        # La config va primero: alto_necesario() la necesita para saber si el
        # paso 1 arranca plegado.
        self.cfg, avisos = H.cargar_config()
        self.mensaje = "AVISO: " + avisos[0] if avisos else ""

        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("BodySense V1")
        self.ui = UI(self.pantalla)
        self.reloj = pygame.time.Clock()

        self.proc = None          # subprocess en curso, o None
        self.proc_nombre = ""
        self.ocupado = False
        self.confirmar_borrado = False
        self.usuario = ""
        self.es_entrenador = False
        self.editando_usuario = False
        self.rechazo = None
        self.refrescar()

    # --- estado ---
    def alto_necesario(self):
        return ALTO

    def refrescar(self):
        self.pares = len(H.pares_de_capturas())
        self.calib = leer_calibracion()
        self.avatar_nombre, self.avatar_color = leer_avatar()
        if not self.usuario:
            self.usuario = self.avatar_nombre

    def guardar(self, clave, valor):
        self.cfg[clave] = valor
        self.cfg, avisos = H.guardar_config(self.cfg)
        self.mensaje = ("AVISO: " + avisos[0]) if avisos else "Config guardada."

    # --- procesos ---
    def actualizar_proceso(self):
        """Una vez por cuadro. Al terminar el script, relee el disco.

        Se hace aqui y no en una propiedad porque los cuatro pasos preguntan
        si hay algo corriendo, y una propiedad con efectos secundarios se
        dispararia en el primero que pregunte.
        """
        if self.proc is not None and self.proc.poll() is not None:
            codigo = self.proc.returncode
            nombre = self.proc_nombre
            self.proc = None
            self.refrescar()
            self.mensaje = (f"{nombre} termino."
                            if codigo == 0 else
                            f"{nombre} salio con codigo {codigo}.")
            if nombre.startswith("generar") and codigo != 0:
                txt = leer_rechazo()
                if txt:
                    self.rechazo = txt
        self.ocupado = self.proc is not None

    def lanzar(self, script, args=None):
        """Corre un script en su propio proceso.

        Con Popen el menu sigue respondiendo, y mientras haya uno corriendo se
        apagan los botones: dos procesos peleandose por las camaras es el
        error que mas facil se comete con un menu de botones.
        """
        ruta = H.buscar_archivo_desesperadamente(script)
        if not ruta:
            self.mensaje = f"No se encontro {script}."
            return
        cmd = [sys.executable, ruta] + list(args or [])
        self.proc = subprocess.Popen(cmd,
                                     cwd=os.path.dirname(ruta))
        self.proc_nombre = script
        self.mensaje = f"Corriendo {script}... revisa su ventana."

    def borrar_capturas(self):
        import shutil

        n = self.pares
        try:
            shutil.rmtree(H.CARPETA_CAPTURAS)
            self.mensaje = f"Borrados {n} pares. La calibracion vieja sigue ahi."
        except OSError as e:
            self.mensaje = f"No se pudo borrar: {e}"
        self.refrescar()

    # --- tarjetas grandes estilo menu ---
    def card_head(self, x, y, w, icono, num, titulo, estado):
        u = self.ui
        u.marca(x + 26, y + 26, estado)
        u.texto(num, x + 52, y + 15, TINTA_TENUE, u.f_paso)
        u.texto(titulo, x + 76, y + 15, TINTA, u.f_paso)
        u.icono_grande(icono, x + w - 44, y + 34)

    def paso_tablero(self, x, y, w, h):
        u = self.ui
        cols, filas = self.cfg["checkerboard"]
        tipo = self.cfg.get("tablero_tipo", "ajedrez")
        es_aj = tipo == "ajedrez"
        if tipo == "charuco":
            cc = self.cfg.get("charuco_cuadros", [5, 7])
            resumen = f"charuco {cc[0]}x{cc[1]}"
        else:
            resumen = f"ajedrez {cols}x{filas}"

        u.tarjeta(x, y, w, h)
        self.card_head(x, y, w, "tablero", "1", "Tablero  " + resumen, "ok")
        u.texto("Tipo de tablero", x + 24, y + 52, TINTA_SUAVE, u.f_chico)
        if u.boton((x + 24, y + 72, 200, 40), "Ajedrez", not self.ocupado,
                   tono=None if es_aj else None):
            if not es_aj:
                self.guardar("tablero_tipo", "ajedrez")
        if u.boton((x + 232, y + 72, 200, 40), "Charuco", not self.ocupado):
            if es_aj:
                self.guardar("tablero_tipo", "charuco")
        u.texto("Esquinas", x + 24, y + 128, TINTA_SUAVE, u.f)
        nc, nx = u.stepper(x + 110, y + 123, cols, 1, 3, 20, str(cols), 34)
        nf, _ = u.stepper(nx + 20, y + 123, filas, 1, 3, 20, str(filas), 34)
        if (nc, nf) != (cols, filas):
            self.guardar("checkerboard", [nc, nf])
        u.texto("Cuadro", x + 24, y + 156, TINTA_SUAVE, u.f)
        mm = self.cfg["tamano_cuadro_mm"]
        nmm, _ = u.stepper(x + 110, y + 151, mm, 1.0, 1.0, 500.0, f"{mm:g} mm")
        if nmm != mm:
            self.guardar("tamano_cuadro_mm", nmm)
        if u.boton((x + 24, y + h - 48, w - 48, 36), "Medir con camara", not self.ocupado, ACENTO):
            self.lanzar("medir_charuco.py" if tipo == "charuco" else "medir_tablero.py")

    def paso_capturas(self, x, y, w, h):
        u = self.ui
        meta = self.cfg["total_fotos"]
        u.tarjeta(x, y, w, h)
        self.card_head(x, y, w, "captura", "2", "Capturar",
                       "ok" if self.pares >= MIN_PARES else "medio" if self.pares else "falta")
        u.texto(f"{self.pares} de {meta} pares", x + 24, y + 52, TINTA, u.mono)
        u.barra(x + 24, y + 74, w - 48, 10, self.pares / max(1, meta),
                VERDE if self.pares >= MIN_PARES else AMBAR)
        u.texto("Meta", x + 24, y + 96, TINTA_SUAVE, u.f)
        nm, _ = u.stepper(x + 80, y + 91, meta, 5, 10, 200, str(meta), 40)
        if nm != meta:
            self.guardar("total_fotos", nm)
        libre = not self.ocupado
        if u.boton((x + 24, y + 130, w - 48, 40), "Capturar fotos", libre, ACENTO):
            self.lanzar("capturar.py", [self.cfg.get("tablero_tipo", "ajedrez")])
        if u.boton((x + 24, y + 176, (w - 56) // 2, 32), "Probar", libre):
            self.mensaje = probar_camaras()
        if u.boton((x + 32 + (w - 56) // 2, y + 176, (w - 56) // 2, 32), "Borrar",
                   libre and self.pares > 0, ROJO):
            self.confirmar_borrado = True

    def paso_calibracion(self, x, y, w, h):
        u = self.ui
        c = self.calib
        estado = ("falta" if c is None else "error" if "error" in c
                  else "ok" if c["creible"] else "medio")
        u.tarjeta(x, y, w, h)
        self.card_head(x, y, w, "calib", "3", "Calibrar", estado)
        if c is None:
            u.texto("Sin stereo.npz. Captura 10+ pares primero.",
                    x + 24, y + 56, TINTA_SUAVE, u.f)
        elif "error" in c:
            u.texto("stereo.npz ilegible.", x + 24, y + 56, ROJO, u.f)
            u.texto(c["error"][:52], x + 24, y + 78, TINTA_TENUE, u.f_chico)
        else:
            u.texto(time.strftime("%d %b %H:%M", c["fecha"]), x + 24, y + 52, TINTA, u.mono)
            u.texto(f"FOV {c['fovs'][0]:.0f} / {c['fovs'][1]:.0f}  base {c['baseline']:.0f}mm",
                    x + 24, y + 76, TINTA_SUAVE, u.f_chico)
            u.texto("Creible." if c["creible"] else "IMPLAUSIBLE: recaptura.",
                    x + 24, y + 98, VERDE if c["creible"] else AMBAR, u.f_chico)
        listo = self.pares >= MIN_PARES and not self.ocupado
        etiqueta = "Regenerar calibracion" if c else "Generar calibracion"
        if u.boton((x + 24, y + h - 48, w - 48, 36), etiqueta, listo, ACENTO):
            tipo = self.cfg.get("tablero_tipo", "ajedrez")
            self.lanzar("generar_calibracion.py", ["charuco"] if tipo == "charuco" else [])

    def paso_ejecutar(self, x, y, w, h):
        u = self.ui
        u.tarjeta(x, y, w, h)
        hay = self.calib is not None and "error" not in self.calib
        self.card_head(x, y, w, "sesion", "4", "Sesion",
                       "ok" if hay else "falta")
        u.texto("Usuario", x + 24, y + 54, TINTA_SUAVE, u.f)
        caja = pygame.Rect(x + 100, y + 50, 180, 28)
        pygame.draw.rect(self.pantalla, (36, 48, 62) if self.editando_usuario else (18, 22, 28),
                         caja, border_radius=6)
        pygame.draw.rect(self.pantalla, ACENTO if self.editando_usuario else BORDE,
                         caja, 1, border_radius=6)
        u.texto(self.usuario or "nombre...", x + 108, y + 56, TINTA, u.mono)
        if self.ui.click and caja.collidepoint(self.ui.click):
            self.editando_usuario = True
        etiqueta_check = "[x] Entrenador" if self.es_entrenador else "[ ] Entrenador"
        if u.boton((x + 290, y + 50, 150, 28), etiqueta_check, not self.ocupado):
            self.es_entrenador = not self.es_entrenador
            self.editando_usuario = False
        u.texto("ESPACIO graba/pausa, ESC sale.", x + 24, y + 88, TINTA_TENUE, u.f_chico)
        libre = not self.ocupado
        nombre = (self.usuario or "Jugador 1").strip() or "Jugador 1"
        if u.boton((x + 24, y + 112, w - 48, 40), "Grabar referencia", hay and libre, ACENTO):
            self.editando_usuario = False
            self.lanzar("main.py", ["--usuario", nombre, "--referencia"])
        if u.boton((x + 24, y + 158, (w - 56) // 2, 36), "Practicar", hay and libre):
            self.editando_usuario = False
            self.lanzar("main.py", ["--usuario", nombre])
        if u.boton((x + 32 + (w - 56) // 2, y + 158, (w - 56) // 2, 36), "Solo ver", hay and libre):
            self.editando_usuario = False
            self.lanzar("main.py", ["--usuario", nombre, "--sin-grabar"])

    def dialogo_borrado(self):
        u = self.ui
        velo = pygame.Surface(self.pantalla.get_size(), pygame.SRCALPHA)
        velo.fill((10, 13, 17, 205))
        self.pantalla.blit(velo, (0, 0))

        x, y, w, h = 110, 250, 500, 170
        u.tarjeta(x, y, w, h)
        u.texto(f"¿Borrar {self.pares} pares de fotos?", x + 26, y + 24, TINTA,
                u.f_paso)
        u.texto("No se puede deshacer. Tendrias que volver a capturar desde cero.",
                x + 26, y + 56, TINTA_SUAVE, u.f)
        u.texto("La calibracion ya generada no se toca.",
                x + 26, y + 76, TINTA_TENUE, u.f_chico)
        if u.boton((x + 26, y + 112, 150, 34), "Si, borrar", True, ROJO):
            self.confirmar_borrado = False
            self.borrar_capturas()
        if u.boton((x + 196, y + 112, 130, 34), "Cancelar"):
            self.confirmar_borrado = False

    def dialogo_rechazo(self):
        u = self.ui
        velo = pygame.Surface(self.pantalla.get_size(), pygame.SRCALPHA)
        velo.fill((10, 13, 17, 205))
        self.pantalla.blit(velo, (0, 0))

        x, y, w, h = 90, 120, 820, 460
        pygame.draw.rect(self.pantalla, (255, 255, 255), (x, y, w, h), border_radius=12)
        pygame.draw.rect(self.pantalla, ROJO, (x, y, w, h), 2, border_radius=12)
        u.texto("Calibracion rechazada: esta muy mal", x + 26, y + 18, ROJO, u.f_paso)
        u.texto("Asi se distorsionaria el avatar. No se guardo nada.",
                x + 26, y + 44, TINTA_SUAVE, u.f_chico)
        lineas = (self.rechazo or "").splitlines()[:9]
        yy = y + 72
        for ln in lineas:
            u.texto(ln[:100], x + 26, yy, TINTA, u.f_chico)
            yy += 20
        u.texto("Tip rapido: 15+ pares, tablero al 30-60%, inclinalo en todas direcciones,",
                x + 26, y + h - 72, TINTA_SUAVE, u.f_chico)
        u.texto("camaras paralelas y misma resolucion.",
                x + 26, y + h - 54, TINTA_SUAVE, u.f_chico)
        if u.boton((x + w - 176, y + h - 44, 150, 32), "Entendido", True, ACENTO):
            self.rechazo = None
            limpiar_rechazo()
            self.refrescar()

    def dibujar(self):
        u = self.ui
        self.pantalla.fill(FONDO)
        u.texto("BodySense V1", 24, 12, TINTA, u.f_titulo)
        u.texto("Toca una tarjeta: 1 tablero, 2 fotos, 3 calibrar, 4 sesion",
                26, 50, TINTA_SUAVE, u.f_sub)
        if self.ocupado:
            u.texto(f"corriendo {self.proc_nombre}", ANCHO - 24, 30, AMBAR,
                    u.f_chico, True)

        x0, y0 = MARGEN, Y_TITULO
        x1 = x0 + CARD_W + GAP
        y1 = y0 + CARD_H + GAP
        # Efecto elevado: la card bajo el cursor sube 5 px.
        import pygame as _pg
        m = self.ui.mouse
        dy0 = -5 if _pg.Rect(x0, y0, CARD_W, CARD_H).collidepoint(m) else 0
        dy1 = -5 if _pg.Rect(x1, y0, CARD_W, CARD_H).collidepoint(m) else 0
        dy2 = -5 if _pg.Rect(x0, y1, CARD_W, CARD_H).collidepoint(m) else 0
        dy3 = -5 if _pg.Rect(x1, y1, CARD_W, CARD_H).collidepoint(m) else 0
        self.paso_tablero(x0, y0 + dy0, CARD_W, CARD_H)
        self.paso_capturas(x1, y0 + dy1, CARD_W, CARD_H)
        self.paso_calibracion(x0, y1 + dy2, CARD_W, CARD_H)
        self.paso_ejecutar(x1, y1 + dy3, CARD_W, CARD_H)

        if self.mensaje:
            color = ROJO if self.mensaje.startswith(("AVISO", "No se pudo")) else TINTA_SUAVE
            u.texto(self.mensaje[:120], 24, ALTO - 28, color, u.f_chico)

    def correr(self):
        andando = True
        while andando:
            self.ui.click = None
            self.ui.mouse = pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    andando = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    if self.rechazo:
                        self.rechazo = None
                        limpiar_rechazo()
                        self.refrescar()
                    elif self.editando_usuario:
                        self.editando_usuario = False
                    elif self.confirmar_borrado:
                        self.confirmar_borrado = False
                    else:
                        andando = False
                elif ev.type == pygame.KEYDOWN and self.editando_usuario:
                    if ev.key == pygame.K_RETURN or ev.key == pygame.K_KP_ENTER:
                        self.editando_usuario = False
                    elif ev.key == pygame.K_BACKSPACE:
                        self.usuario = self.usuario[:-1]
                    elif ev.unicode and len(self.usuario) < 24:
                        self.usuario += ev.unicode
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.ui.click = ev.pos

            self.actualizar_proceso()

            if self.rechazo:
                click = self.ui.click
                self.ui.click = None
                self.dibujar()
                self.ui.click = click
                self.dialogo_rechazo()
            elif self.confirmar_borrado:
                # El dialogo se traga los clicks: el fondo se redibuja sin
                # ellos para que no se pueda apretar "Capturar" por detras.
                click = self.ui.click
                self.ui.click = None
                self.dibujar()
                self.ui.click = click
                self.dialogo_borrado()
            else:
                self.dibujar()

            pygame.display.flip()
            self.reloj.tick(30)
        pygame.quit()


if __name__ == "__main__":
    Menu().correr()
