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

import Herramientas as H

# ==========================================
# APARIENCIA
# ==========================================
ANCHO, ALTO = 720, 700

FONDO = (18, 22, 28)
TARJETA = (27, 34, 43)
BORDE = (42, 52, 63)
TINTA = (230, 235, 240)
TINTA_SUAVE = (143, 160, 176)
TINTA_TENUE = (100, 115, 130)
ACENTO = (76, 155, 232)
VERDE = (63, 191, 143)
AMBAR = (224, 166, 75)
ROJO = (224, 101, 91)

# Familias por plataforma; SysFont acepta la lista y toma la primera que exista.
FAM = "segoeui,helveticaneue,helvetica,arial,dejavusans"
FAM_MONO = "menlo,consolas,dejavusansmono,couriernew,monospace"

ARCHIVO_AVATAR = os.path.join(H.BASE, "avatar_config.json")
ARCHIVO_STEREO = os.path.join(H.CARPETA_CALIBRACION, "stereo.npz")

# Minimo de pares que generar_calibracion.py exige para siquiera intentarlo.
MIN_PARES = 10


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
        self.f_titulo = pygame.font.SysFont(FAM, 24, bold=True)
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
        pygame.draw.rect(self.p, TARJETA, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.p, BORDE, (x, y, w, h), 1, border_radius=10)

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
        base = tono or BORDE

        if not activo:
            relleno, letra = (34, 41, 50), TINTA_TENUE
        elif tono is ACENTO:
            relleno = (96, 172, 245) if encima else ACENTO
            letra = (12, 20, 30)
        elif tono is ROJO:
            # Fondo oscuro con letra y borde rojos: relleno rojo mas letra roja
            # deja el boton en blanco, que es como quedo la primera version.
            relleno, letra = ((66, 38, 38) if encima else (48, 31, 33)), ROJO
        else:
            relleno = tuple(min(255, c + 18) for c in base) if encima else base
            letra = TINTA

        pygame.draw.rect(self.p, relleno, r, border_radius=7)
        if tono is ROJO and activo:
            pygame.draw.rect(self.p, ROJO, r, 1, border_radius=7)

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
        pygame.draw.rect(self.p, (44, 53, 64), (x, y, w, h), border_radius=h // 2)
        if frac > 0:
            pygame.draw.rect(self.p, color, (x, y, max(h, int(w * min(1.0, frac))), h),
                             border_radius=h // 2)


# ==========================================
# APLICACION
# ==========================================
class Menu:
    def __init__(self):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("BodySense")
        self.ui = UI(self.pantalla)
        self.reloj = pygame.time.Clock()

        self.cfg, avisos = H.cargar_config()
        self.mensaje = "AVISO: " + avisos[0] if avisos else ""
        self.proc = None          # subprocess en curso, o None
        self.proc_nombre = ""
        self.ocupado = False
        self.confirmar_borrado = False
        self.refrescar()

    # --- estado ---
    def refrescar(self):
        self.pares = len(H.pares_de_capturas())
        self.calib = leer_calibracion()
        self.avatar_nombre, self.avatar_color = leer_avatar()

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
            self.proc = None
            self.refrescar()
            self.mensaje = (f"{self.proc_nombre} termino."
                            if codigo == 0 else
                            f"{self.proc_nombre} salio con codigo {codigo}.")
        self.ocupado = self.proc is not None

    def lanzar(self, script):
        """Corre un script en su propio proceso.

        Con Popen el menu sigue respondiendo, y mientras haya uno corriendo se
        apagan los botones: dos procesos peleandose por las camaras es el
        error que mas facil se comete con un menu de botones.
        """
        self.proc = subprocess.Popen([sys.executable, os.path.join(H.BASE, script)],
                                     cwd=H.BASE)
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

    # --- dibujo ---
    def paso_tablero(self, y):
        u, w, h = self.ui, ANCHO - 48, 122
        u.tarjeta(24, y, w, h)
        u.marca(50, y + 26, "ok")
        u.texto("1", 68, y + 15, TINTA_TENUE, u.f_paso)
        u.texto("Medir el tablero", 88, y + 15, TINTA, u.f_paso)
        u.texto("El tamano del cuadro fija la escala metrica de todo el sistema.",
                88, y + 38, TINTA_SUAVE, u.f_chico)

        cols, filas = self.cfg["checkerboard"]
        u.texto("Esquinas internas", 88, y + 66, TINTA_SUAVE, u.f)
        nc, x = u.stepper(210, y + 61, cols, 1, 3, 20, str(cols), 34)
        u.texto("×", x + 6, y + 64, TINTA_TENUE, u.f)
        nf, _ = u.stepper(x + 20, y + 61, filas, 1, 3, 20, str(filas), 34)
        if (nc, nf) != (cols, filas):
            self.guardar("checkerboard", [nc, nf])

        u.texto("Cuadro", 88, y + 94, TINTA_SUAVE, u.f)
        mm = self.cfg["tamano_cuadro_mm"]
        nmm, _ = u.stepper(210, y + 89, mm, 1.0, 1.0, 500.0, f"{mm:g} mm")
        if nmm != mm:
            self.guardar("tamano_cuadro_mm", nmm)

        if u.boton((w - 132, y + 70, 156, 34), "Medir con camara", not self.ocupado):
            self.lanzar("medir_tablero.py")
        return y + h + 12

    def paso_capturas(self, y):
        u, w, h = self.ui, ANCHO - 48, 148
        meta = self.cfg["total_fotos"]
        u.tarjeta(24, y, w, h)
        u.marca(50, y + 26, "ok" if self.pares >= MIN_PARES
                else "medio" if self.pares else "falta")
        u.texto("2", 68, y + 15, TINTA_TENUE, u.f_paso)
        u.texto("Capturar fotos del tablero", 88, y + 15, TINTA, u.f_paso)

        u.texto(f"{self.pares} de {meta} pares", 88, y + 40, TINTA, u.mono)
        u.barra(88, y + 60, 300, 8, self.pares / max(1, meta),
                VERDE if self.pares >= MIN_PARES else AMBAR)
        if self.pares < MIN_PARES:
            u.texto(f"generar_calibracion.py pide {MIN_PARES} minimo",
                    398, y + 57, TINTA_TENUE, u.f_chico)

        u.texto("Meta", 88, y + 82, TINTA_SUAVE, u.f)
        nm, _ = u.stepper(150, y + 77, meta, 5, 10, 200, str(meta), 40)
        if nm != meta:
            self.guardar("total_fotos", nm)

        u.texto("Tamano minimo", 260, y + 82, TINTA_SUAVE, u.f)
        mi = self.cfg["min_ancho_tablero"]
        nmi, _ = u.stepper(370, y + 77, round(mi, 2), 0.01, 0.02, 0.90,
                           f"{mi * 100:.0f}%", 46)
        if abs(nmi - mi) > 1e-9:
            self.guardar("min_ancho_tablero", round(nmi, 2))

        libre = not self.ocupado
        if u.boton((88, y + 112, 120, 26), "Capturar", libre):
            self.lanzar("capturar.py")
        if u.boton((216, y + 112, 130, 26), "Probar camaras", libre):
            self.mensaje = probar_camaras()
        if u.boton((w - 84, y + 112, 108, 26), "Borrar fotos",
                   libre and self.pares > 0, ROJO):
            self.confirmar_borrado = True
        return y + h + 12

    def paso_calibracion(self, y):
        u, w, h = self.ui, ANCHO - 48, 122
        c = self.calib
        estado = ("falta" if c is None else "error" if "error" in c
                  else "ok" if c["creible"] else "medio")
        u.tarjeta(24, y, w, h)
        u.marca(50, y + 26, estado)
        u.texto("3", 68, y + 15, TINTA_TENUE, u.f_paso)
        u.texto("Generar la calibracion", 88, y + 15, TINTA, u.f_paso)

        if c is None:
            u.texto("No hay stereo.npz todavia.", 88, y + 40, TINTA_SUAVE, u.f)
            u.texto("main.py no puede correr sin el.", 88, y + 60, TINTA_TENUE, u.f_chico)
        elif "error" in c:
            u.texto("stereo.npz existe pero no se puede leer.", 88, y + 40, ROJO, u.f)
            u.texto(c["error"][:74], 88, y + 60, TINTA_TENUE, u.f_chico)
        else:
            u.texto(time.strftime("%d %b %Y  %H:%M", c["fecha"]), 88, y + 40,
                    TINTA, u.mono)
            u.texto(f"FOV {c['fovs'][0]:.0f}° / {c['fovs'][1]:.0f}°"
                    f"    baseline {c['baseline']:.0f} mm"
                    f"    calibrada a {c['anchos'][0]:.0f} y {c['anchos'][1]:.0f} px",
                    88, y + 62, TINTA_SUAVE, u.mono_chico)
            if c["creible"]:
                u.texto("Campo de vision creible.", 88, y + 82, VERDE, u.f_chico)
            else:
                # Dos lineas: en una sola el texto se metia debajo del boton.
                u.texto("CAMPO DE VISION IMPLAUSIBLE - quedo mal condicionada.",
                        88, y + 80, AMBAR, u.f_chico)
                u.texto("Recaptura con el tablero mas grande y mas inclinado.",
                        88, y + 96, TINTA_SUAVE, u.f_chico)

        listo = self.pares >= MIN_PARES and not self.ocupado
        etiqueta = "Regenerar" if c else "Generar"
        if u.boton((w - 108, y + 58, 132, 34), etiqueta, listo):
            self.lanzar("generar_calibracion.py")
        if self.pares < MIN_PARES:
            u.texto("faltan fotos", w - 42, y + 94, TINTA_TENUE, u.f_chico, True)
        return y + h + 12

    def paso_ejecutar(self, y):
        u, w, h = self.ui, ANCHO - 48, 122
        u.tarjeta(24, y, w, h)
        hay = self.calib is not None and "error" not in self.calib
        u.marca(50, y + 26, "ok" if hay else "falta")
        u.texto("4", 68, y + 15, TINTA_TENUE, u.f_paso)
        u.texto("Ver el esqueleto 3D", 88, y + 15, TINTA, u.f_paso)

        u.texto("Avatar", 88, y + 42, TINTA_SUAVE, u.f)
        pygame.draw.rect(self.pantalla, self.avatar_color, (140, y + 42, 16, 16),
                         border_radius=4)
        u.texto(self.avatar_nombre, 164, y + 42, TINTA, u.f)
        u.texto("Flechas giran la vista, Z y X acercan, ESC sale.",
                88, y + 66, TINTA_TENUE, u.f_chico)

        libre = not self.ocupado
        if u.boton((88, y + 88, 148, 26), "Configurar avatar", libre):
            self.lanzar("menu.py")
        if u.boton((248, y + 88, 150, 26), "Probar sin camaras", libre):
            self.lanzar("probar_render.py")
        if u.boton((w - 84, y + 82, 108, 36), "EJECUTAR", hay and libre, ACENTO):
            self.lanzar("main.py")
        return y + h

    def dialogo_borrado(self):
        u = self.ui
        velo = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
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

    def dibujar(self):
        u = self.ui
        self.pantalla.fill(FONDO)
        u.texto("BodySense", 24, 22, TINTA, u.f_titulo)
        u.texto("Flujo de calibracion", 24, 52, TINTA_SUAVE, u.f)
        cols, filas = self.cfg["checkerboard"]
        u.texto(f"{cols}×{filas} esquinas · {self.cfg['tamano_cuadro_mm']:g} mm",
                ANCHO - 24, 26, TINTA_TENUE, u.mono_chico, True)
        if self.ocupado:
            u.texto(f"corriendo {self.proc_nombre}", ANCHO - 24, 46, AMBAR,
                    u.f_chico, True)

        y = 84
        y = self.paso_tablero(y)
        y = self.paso_capturas(y)
        y = self.paso_calibracion(y)
        y = self.paso_ejecutar(y)

        if self.mensaje:
            color = ROJO if self.mensaje.startswith(("AVISO", "No se pudo")) else TINTA_SUAVE
            u.texto(self.mensaje[:96], 24, y + 16, color, u.f_chico)

    def correr(self):
        andando = True
        while andando:
            self.ui.click = None
            self.ui.mouse = pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    andando = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    if self.confirmar_borrado:
                        self.confirmar_borrado = False
                    else:
                        andando = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.ui.click = ev.pos

            self.actualizar_proceso()

            if self.confirmar_borrado:
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
