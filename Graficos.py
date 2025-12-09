from OpenGL.GL import * #Gl es la biblioteca principal de OpenGL que contiene funciones para renderizar gráficos 2D y 3D
from OpenGL.GLU import * #Glu es una biblioteca auxiliar para OpenGL lo q hace es facilitar ciertas tareas como dibujar formas complejas
import math #Math lo usamos para operaciones matemáticas como cálculos trigonométricos y raíces cuadradas


# Dibujar un hueso entre dos puntos 3D como un cilindro para representar el esqueleto
def dibujar_hueso(p1, p2, color):
    x1, y1, z1 = p1; x2, y2, z2 = p2
    dx, dy, dz = x2-x1, y2-y1, z2-z1
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 0.01: return

    # Dibujar cilindro entre p1 y p2

    glPushMatrix()
    glTranslatef(x1, y1, z1)
    # Calcular ángulos de rotación
    ax = 57.29578 * math.atan2(dy, dx)
    if dx < 0: ax += 180
    # Rotar cilindro para alinearlo entre los dos puntos
    phi = math.degrees(math.acos(dz/dist)) if dist > 0 else 0
    theta = math.degrees(math.atan2(dy, dx))
    
    glRotatef(theta, 0, 0, 1); glRotatef(phi, 0, 1, 0)
    
    # Dibujar el cilindro
    glColor3f(*color)
    quad = gluNewQuadric()
    GROSOR = 25 # Grosor del hueso
    gluCylinder(quad, GROSOR, GROSOR, dist, 8, 1)
    # Tapar los extremos del cilindro 
    glPopMatrix()

# Dibujar una esfera para representar una articulación del esqueleto en 3D
def dibujar_joint(pos, color):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, 40, 10, 10) 
    glPopMatrix()

# Dibujar la cabeza como una esfera en 3D
def dibujar_cabeza(pos, color):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, 180, 16, 16) 
    glPopMatrix()