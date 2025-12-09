from OpenGL.GL import *
from OpenGL.GLU import *
import math


def dibujar_hueso(p1, p2, color):
    x1, y1, z1 = p1; x2, y2, z2 = p2
    dx, dy, dz = x2-x1, y2-y1, z2-z1
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 0.01: return

    glPushMatrix()
    glTranslatef(x1, y1, z1)
    ax = 57.29578 * math.atan2(dy, dx)
    if dx < 0: ax += 180
    phi = math.degrees(math.acos(dz/dist)) if dist > 0 else 0
    theta = math.degrees(math.atan2(dy, dx))
    glRotatef(theta, 0, 0, 1); glRotatef(phi, 0, 1, 0)
    
    glColor3f(*color)
    quad = gluNewQuadric()
    GROSOR = 25 
    gluCylinder(quad, GROSOR, GROSOR, dist, 8, 1) 
    glPopMatrix()

def dibujar_joint(pos, color):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, 40, 10, 10) 
    glPopMatrix()

def dibujar_cabeza(pos, color):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, 180, 16, 16) 
    glPopMatrix()