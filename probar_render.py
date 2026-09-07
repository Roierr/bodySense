"""Prueba el render 3D sin cámaras, con un esqueleto sintético.

Sirve para separar dos fallas que se ven igual (ventana negra):
  - las cámaras no detectan a la persona
  - el código de dibujado no funciona

Si aquí aparece un muñeco girando, el render y Graficos.py estan bien y el
problema esta en la deteccion. Usa la misma configuracion de OpenGL, la misma
escala y las mismas funciones de dibujo que main.py.

    python3 probar_render.py       (ESC o cerrar para salir)
"""
import json
import math

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from Herramientas import buscar_archivo_desesperadamente
from Graficos import dibujar_hueso, dibujar_joint, dibujar_cabeza

# Mismas constantes que main.py, para que la prueba sea representativa
ESCALA_GIGANTE = 1.5
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

# Esqueleto sintetico en milimetros, con los indices de MediaPipe.
# Convencion igual a la que sale de la triangulacion: Y crece hacia ABAJO
# (como en una imagen) y Z es la distancia a las camaras. Persona de pie de
# ~1.6 m, a 2 m de distancia.
CUERPO_MM = {
     0: [   0, -700, 2000],   # nariz
    11: [ 200, -500, 2000],   # hombro izq
    12: [-200, -500, 2000],   # hombro der
    13: [ 330, -250, 2000],   # codo izq
    14: [-330, -250, 2000],   # codo der
    15: [ 380,    0, 2000],   # muñeca izq
    16: [-380,    0, 2000],   # muñeca der
    23: [ 120,    0, 2000],   # cadera izq
    24: [-120,    0, 2000],   # cadera der
    25: [ 130,  450, 2000],   # rodilla izq
    26: [-130,  450, 2000],   # rodilla der
    27: [ 140,  900, 2000],   # tobillo izq
    28: [-140,  900, 2000],   # tobillo der
}


def color_del_avatar():
    """Lee el color de avatar_config.json igual que main.py, con el mismo default."""
    cfg = buscar_archivo_desesperadamente("avatar_config.json")
    if cfg:
        try:
            with open(cfg) as f:
                h = json.load(f)["color_piel"].lstrip("#")
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except (OSError, ValueError, KeyError):
            pass
    return (0.2, 0.6, 1.0)


def main():
    color = color_del_avatar()

    # Mismo pipeline de escalado y negacion de Y que main.py
    pts = {i: [p[0] * ESCALA_GIGANTE, -p[1] * ESCALA_GIGANTE, p[2] * ESCALA_GIGANTE]
           for i, p in CUERPO_MM.items()}
    # Centro del cuerpo, para girarlo sobre si mismo y no en una orbita
    cx = sum(p[0] for p in pts.values()) / len(pts)
    cy = sum(p[1] for p in pts.values()) / len(pts)
    cz = sum(p[2] for p in pts.values()) / len(pts)

    pygame.init()
    pygame.display.set_mode((1000, 800), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Prueba de render - sin camaras")
    gluPerspective(45, (1000 / 800), 0.1, 50000.0)
    glTranslatef(0.0, -150.0, -900.0)
    glRotatef(180, 0, 1, 0)
    glDisable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    reloj = pygame.time.Clock()
    angulo = 0.0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Gira el cuerpo sobre su propio eje vertical: si se ve la profundidad
        # cambiar al girar, la reconstruccion 3D se esta dibujando de verdad.
        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glRotatef(angulo, 0, 1, 0)
        glTranslatef(-cx, -cy, -cz)

        for a, b in CONEXIONES:
            dibujar_hueso(pts[a], pts[b], color)
            dibujar_joint(pts[a], color)
            dibujar_joint(pts[b], color)
        dibujar_cabeza(pts[0], color)

        glPopMatrix()

        pygame.display.flip()
        angulo = (angulo + 0.7) % 360
        reloj.tick(60)


def demo():
    """Chequeo sin ventana: el esqueleto sintetico es consistente y dibujable."""
    assert all(i in CUERPO_MM for par in CONEXIONES for i in par), \
        "hay conexiones que apuntan a puntos que no existen"
    assert 0 in CUERPO_MM, "falta la nariz (indice 0) para dibujar la cabeza"
    # Tras negar Y, la nariz debe quedar ARRIBA de los tobillos en el eje de OpenGL
    nariz_y = -CUERPO_MM[0][1]
    tobillo_y = -CUERPO_MM[27][1]
    assert nariz_y > tobillo_y, f"el cuerpo sale de cabeza: nariz={nariz_y} tobillo={tobillo_y}"
    # Altura plausible de una persona: entre 1.2 y 2.2 m
    altura = abs(CUERPO_MM[0][1] - CUERPO_MM[27][1])
    assert 1200 <= altura <= 2200, f"altura implausible: {altura} mm"
    print(f"OK: {len(CUERPO_MM)} puntos, {len(CONEXIONES)} huesos, altura {altura} mm")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        demo()
    else:
        main()
