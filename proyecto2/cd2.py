
from __future__ import annotations
import os, sys, ssl, math, time, urllib.request

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import keyboard as key 

# ── Descarga automática del modelo MediaPipe ──────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo MediaPipe...")
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Descarga completada.")
    except Exception as e:
        print(f"Error descarga: {e}")

# ── MediaPipe Hand Landmarker ─────────────────────────────────────────────
_mp_base    = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
_mp_opts    = vision.HandLandmarkerOptions(base_options=_mp_base, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(_mp_opts)

WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE  = "Ciudad 3D Interactiva"

moverx, movery, moverz = 0.0, 1.0, 20.0
dx, dz = 0.0, -1.0
girar = 0.0

# ── Variables de animación global ─────────────────────────────────────────
 
ang_perro3,  dir_perro3  = 0.0, 12.0  
ang_peli,    dir_peli    = 0.0, 10.0  
ang_pollo,   dir_pollo   = 0.0, 10.0  
ang_pez                  = 0.0        
mov_abeja,   dir_abeja   = 0.0, 0.05  
d_barco,     dir_barco   = 0.0, 0.3   
ang_coco1,   dir_coco1   = 0.0,-10.0  
ang_coco2,   dir_coco2   = 0.0,-10.0  
angulo_dia               = 0.0        
r_cielo, g_cielo, b_cielo = 0.4, 0.75, 1.0
ola_tam,     dir_ola     = 0.0, 0.05  

# -------------------------------------------------------------------------
# GESTOS DE MANO 
# -------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),         # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),    # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17)                                 # Base de la palma
]

def contar_dedos_arriba(lm):
    dedos = 0
    if lm[8].y < lm[6].y: dedos += 1   # Índice
    if lm[12].y < lm[10].y: dedos += 1 # Medio
    if lm[16].y < lm[14].y: dedos += 1 # Anular
    if lm[20].y < lm[18].y: dedos += 1 # Meñique
    return dedos

def es_paz(lm):
    # Índice y medio arriba, anular y meñique abajo (Amor y Paz)
    return lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y

def es_palma_abierta(lm):
    # Todos los dedos extendidos
    return contar_dedos_arriba(lm) == 4

def es_puno(lm):
    # Ningún dedo extendido
    return contar_dedos_arriba(lm) == 0

# -------------------------------------------------------------------------
# CONFIGURACIÓN OPENGL
# -------------------------------------------------------------------------
def init_glfw():
    if not glfw.init(): raise Exception("No se pudo inicializar GLFW") 
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None) 
    if not window: 
        glfw.terminate(); raise Exception("No se pudo crear la ventana") 
    glfw.make_context_current(window) 
    glfw.swap_interval(1)
    return window 

def setup_opengl(): 
    glClearColor(0.4, 0.75, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS) 
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
    glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST) 

def setup_lights(): 
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL) 
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) 
    glLightfv(GL_LIGHT0, GL_POSITION, (10, 20, 10, 0)) 
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1)) 
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.6, 0.6, 0.65, 1)) 

def normal_tri(v1, v2, v3): 
    ux, uy, uz = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
    vx, vy, vz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
    nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx 
    l = math.sqrt(nx*nx + ny*ny + nz*nz) 
    if l < 1e-8: return (0,1,0) 
    return (nx/l, ny/l, nz/l) 

def draw_rectangulo(lado1, lado2, color):  
    glBegin(GL_QUADS); glColor3f(*color); glNormal3f(0, 1, 0)  
    glVertex3f(-lado1, 0, lado2); glVertex3f(lado1, 0, lado2) 
    glVertex3f(lado1, 0, -lado2); glVertex3f(-lado1, 0, -lado2) 
    glEnd() 

def draw_cube(alto, ancho, largo, color1, color2, color3): 
    ancho, largo = ancho/2, largo/2 
    glBegin(GL_QUADS) 
    glColor3f(*color1); glNormal3f(0,0,1); glVertex3f(-ancho, 0, largo); glVertex3f(ancho, 0, largo); glVertex3f(ancho, alto, largo); glVertex3f(-ancho, alto, largo) 
    glNormal3f(0,0,-1); glVertex3f(-ancho, 0, -largo); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, alto, -largo); glVertex3f(-ancho, alto, -largo) 
    glNormal3f(-1,0,0); glVertex3f(-ancho, 0, -largo); glVertex3f(-ancho, 0, largo); glVertex3f(-ancho, alto, largo); glVertex3f(-ancho, alto, -largo) 
    glNormal3f(1,0,0); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, 0, largo); glVertex3f(ancho, alto, largo); glVertex3f(ancho, alto, -largo) 
    glColor3f(*color2); glNormal3f(0,1,0); glVertex3f(-ancho, alto, -largo); glVertex3f(ancho, alto, -largo); glVertex3f(ancho, alto, largo); glVertex3f(-ancho, alto, largo) 
    glColor3f(*color3); glNormal3f(0,-1,0); glVertex3f(-ancho, 0, -largo); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, 0, largo); glVertex3f(-ancho, 0, largo) 
    glEnd() 

def draw_piramide(alto, ancho, largo, altura, color):
    glBegin(GL_TRIANGLES); glColor3f(*color) 
    ancho /= 2; largo /= 2; alto = alto + altura; cima = (0, alto, 0)
    caras = [((-ancho, altura, largo), (ancho, altura, largo)), ((ancho, altura, -largo), (-ancho, altura, -largo)), ((-ancho, altura, -largo), (-ancho, altura, largo)), ((ancho, altura, largo), (ancho, altura, -largo))]
    for v1, v2 in caras:
        glNormal3f(*normal_tri(v1, v2, cima)); glVertex3f(*v1); glVertex3f(*v2); glVertex3f(*cima) 
    glEnd() 

def draw_sphere(radius, color=(1, 1, 1)): 
    glColor3f(*color); quad = gluNewQuadric(); gluQuadricNormals(quad, GLU_SMOOTH); gluSphere(quad, radius, 16, 16); gluDeleteQuadric(quad) 

def draw_cilindro_tapado(ang_in, ang_fin, radius, largo, color): 
    glBegin(GL_TRIANGLE_STRIP) 
    for i in range(ang_in, ang_fin+1, 10): 
        x1, y1 = math.cos(math.radians(i)), math.sin(math.radians(i)) 
        glColor3f(*color); glNormal3f(x1, y1, 0) 
        glVertex3f(x1 * radius, y1 * radius, 0); glVertex3f(x1 * radius, y1 * radius, largo) 
    glEnd() 
    glBegin(GL_TRIANGLE_FAN); glColor3f(*color); glNormal3f(0,0,-1); glVertex3f(0,0,0)
    for i in range(ang_in, ang_fin+1, 10): glVertex3f(math.cos(math.radians(i))*radius, math.sin(math.radians(i))*radius, 0)
    glEnd()
    glBegin(GL_TRIANGLE_FAN); glColor3f(*color); glNormal3f(0,0,1); glVertex3f(0,0,largo)
    for i in range(ang_fin, ang_in-1, -10): glVertex3f(math.cos(math.radians(i))*radius, math.sin(math.radians(i))*radius, largo)
    glEnd()

def mover_en_circulo(draw_funcion, radio, ang_cambio, angulo_base, voltear, altura=0.0): 
    theta = ang_cambio + angulo_base
    x, z = radio * math.cos(theta), radio * math.sin(theta) 
    glPushMatrix(); glTranslatef(x, altura, z); glRotatef(math.degrees(-theta) + voltear, 0, 1, 0) 
    draw_funcion(); glPopMatrix() 

# -------------------------------------------------------------------------
# CATÁLOGO DE OBJETOS COMPLETOS
# -------------------------------------------------------------------------
def draw_flor(color_petalo, color_centro):
    glPushMatrix(); glTranslate(0, 0.2, 0)
    glPushMatrix(); glRotate(90, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.03, 0.2, (0.2, 0.8, 0.2)); glPopMatrix()
    draw_sphere(0.08, color_centro)
    for i in range(5):
        glPushMatrix(); glRotate(i * 72, 0, 1, 0); glTranslate(0.12, 0, 0); draw_sphere(0.08, color_petalo); glPopMatrix()
    glPopMatrix()

def draw_hongo(color_sombrero):
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.1, 0.3, (0.9, 0.9, 0.8)); glPopMatrix()
    glTranslate(0, 0.3, 0); draw_sphere(0.25, color_sombrero)
    glPopMatrix()

def draw_comida_mercado():
    glPushMatrix()
    glTranslate(-0.3, 0.9, 0); draw_sphere(0.1, (0.9, 0.1, 0.1)) 
    glTranslate(0.3, 0, 0.2); draw_sphere(0.1, (0.2, 0.8, 0.2))  
    glTranslate(0.2, 0, -0.2); draw_sphere(0.08, (0.9, 0.5, 0.1)) 
    glPopMatrix()

def draw_puesto_mercado(color_toldo):
    glPushMatrix()
    glTranslate(0, 0.8, 0); draw_cube(0.1, 2.0, 1.0, (0.5, 0.3, 0.1), (0.5, 0.3, 0.1), (0.5, 0.3, 0.1))
    for x in [-0.9, 0.9]:
        for z in [-0.4, 0.4]:
            glPushMatrix(); glTranslate(x, -0.4, z); draw_cube(0.8, 0.1, 0.1, (0.3, 0.2, 0.1), (0.3, 0.2, 0.1), (0.3, 0.2, 0.1)); glPopMatrix()
    for x in [-0.9, 0.9]:
        glPushMatrix(); glTranslate(x, 0.8, -0.4); draw_cube(1.5, 0.1, 0.1, (0.3, 0.2, 0.1), (0.3, 0.2, 0.1), (0.3, 0.2, 0.1)); glPopMatrix()
    glTranslate(0, 1.6, 0); glRotate(25, 1, 0, 0)
    for i in range(-5, 5):
        color = color_toldo if i % 2 == 0 else (0.9, 0.9, 0.9)
        glPushMatrix(); glTranslate(i * 0.22 + 0.11, 0, 0); draw_cube(0.05, 0.22, 1.6, color, color, color); glPopMatrix()
    glPopMatrix()
    draw_comida_mercado()

def draw_huerto():
    glPushMatrix()
    glTranslate(0, 0.02, 0); draw_rectangulo(2.0, 2.0, (0.4, 0.2, 0.1))
    for x in [-1.0, 0, 1.0]:
        for z in [-1.0, 0, 1.0]:
            glPushMatrix(); glTranslate(x, 0.2, z); draw_sphere(0.25, (0.9, 0.5, 0.1)) 
            glTranslate(0, 0.2, 0); draw_cilindro_tapado(0, 360, 0.05, 0.2, (0.2, 0.6, 0.2)); glPopMatrix()
    glPopMatrix()

def draw_sombrilla():
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.05, 2.0, (0.8, 0.8, 0.8)); glRotate(-270, 1, 0, 0)
    glTranslate(0, 2.0, 0)
    for i in range(8):
        color = (0.9, 0.2, 0.2) if i % 2 == 0 else (0.9, 0.9, 0.9)
        glPushMatrix(); glRotate(i * 45, 0, 1, 0); draw_piramide(0.5, 1.5, 1.5, 0, color); glPopMatrix()
    glPopMatrix()

def draw_toalla_playa():
    glPushMatrix(); glTranslate(0, 0.03, 0)
    for i in range(5):
        color = (0.9, 0.4, 0.6) if i % 2 == 0 else (0.9, 0.9, 0.9)
        glPushMatrix(); glTranslate(0, 0, i*0.4 - 0.8); draw_cube(0.02, 1.2, 0.4, color, color, color); glPopMatrix()
    glPopMatrix()

def draw_cerca():
    glPushMatrix()
    for i in range(-2, 3):
        glPushMatrix(); glTranslate(i*0.6, 0.3, 0); draw_cube(0.6, 0.15, 0.1, (0.5, 0.3, 0.1), (0.6, 0.4, 0.2), (0.4, 0.2, 0.1)); glPopMatrix()
    glTranslate(0, 0.4, 0); draw_cube(0.1, 3.0, 0.15, (0.5, 0.3, 0.1), (0.6, 0.4, 0.2), (0.4, 0.2, 0.1))
    glPopMatrix()

def draw_arbusto():
    glPushMatrix()
    glTranslate(0, 0.4, 0); draw_sphere(0.5, (0.2, 0.6, 0.2))
    glTranslate(0.2, -0.1, 0.2); draw_sphere(0.3, (0.15, 0.55, 0.15))
    glTranslate(-0.4, -0.1, -0.2); draw_sphere(0.35, (0.25, 0.65, 0.25))
    glPopMatrix()

def draw_casa_colorida(color_pared, color_techo):
    glPushMatrix()
    draw_cilindro_tapado(0, 360, 1.2, 1.2, color_pared)
    glTranslate(0, 1.2, 0); draw_piramide(1.5, 2.5, 2.5, 0, color_techo)
    glTranslate(0, -0.8, 1.2); glRotate(90, 1, 0, 0); draw_rectangulo(0.3, 0.5, (0.4, 0.2, 0.1))
    glPopMatrix()

def draw_casa_zen(color_pared, color_techo):
    glPushMatrix()
    draw_cube(1.2, 2.2, 2.0, color_pared, color_pared, color_pared) 
    glTranslate(0, 1.2, 0); draw_cube(0.2, 2.6, 2.4, color_techo, color_techo, color_techo)
    glTranslate(0, 0.2, 0); draw_piramide(1.0, 2.2, 2.0, 0, color_techo)
    glTranslate(0, -1.0, 1.01); glRotate(90, 1, 0, 0); draw_rectangulo(0.4, 0.5, (0.9, 0.9, 0.8))
    glTranslate(0, 0, -0.01); draw_rectangulo(0.45, 0.55, (0.3, 0.2, 0.1))
    glPopMatrix()

def draw_reloj_animado(t): 
   glColor3f(0.9, 0.9, 0.9); quad = gluNewQuadric(); gluDisk(quad, 0, 0.8, 32, 1) 
   glPushMatrix(); glRotatef(-t * 20, 0, 0, 1); glLineWidth(4); glBegin(GL_LINES); glColor3f(0, 0, 0); glVertex3f(0, 0, 0.01); glVertex3f(0, 0.4, 0.01); glEnd(); glPopMatrix() 
   glPushMatrix(); glRotatef(-t * 150, 0, 0, 1); glLineWidth(2); glBegin(GL_LINES); glColor3f(0.2, 0.2, 0.2); glVertex3f(0, 0, 0.02); glVertex3f(0, 0.6, 0.02); glEnd(); glPopMatrix() 

def draw_edificio_reloj(t): 
   draw_cube(8, 4, 4, (0.8, 0.7, 0.6),(0.8, 0.7, 0.6),(0.8, 0.7, 0.6)) 
   for fila in range(1, 4): 
       for col in [-1, 1]: 
           glPushMatrix(); glTranslatef(col * 0.8, fila * 1.8, 2.01); draw_cube(1, 0.8, 0.01, (0.2, 0.6, 0.9),(0.2, 0.6, 0.9),(0.2, 0.6, 0.9)); glPopMatrix() 
   glPushMatrix(); glTranslatef(0, 8, 0); draw_cube(2.5, 3, 3, (0.6, 0.2, 0.2),(0.6, 0.2, 0.2),(0.6, 0.2, 0.2)); glPopMatrix()
   glPushMatrix(); glTranslatef(0, 1.25, 2.01); draw_reloj_animado(t); glPopMatrix() 

def draw_tienda_campana(color_tela):
    glPushMatrix()
    glBegin(GL_TRIANGLES); glColor3f(*color_tela)
    glNormal3f(0, 0, 1); glVertex3f(-1.5, 0, 1.5); glVertex3f(1.5, 0, 1.5); glVertex3f(0, 2.0, 1.5)
    glNormal3f(0, 0, -1); glVertex3f(-1.5, 0, -1.5); glVertex3f(1.5, 0, -1.5); glVertex3f(0, 2.0, -1.5)
    glEnd()
    glBegin(GL_QUADS); glColor3f(color_tela[0]*0.9, color_tela[1]*0.9, color_tela[2]*0.9)
    glNormal3f(-1, 1, 0); glVertex3f(-1.5, 0, 1.5); glVertex3f(-1.5, 0, -1.5); glVertex3f(0, 2.0, -1.5); glVertex3f(0, 2.0, 1.5)
    glNormal3f(1, 1, 0); glVertex3f(1.5, 0, 1.5); glVertex3f(1.5, 0, -1.5); glVertex3f(0, 2.0, -1.5); glVertex3f(0, 2.0, 1.5)
    glEnd()
    glTranslate(0, 0, 1.51); draw_rectangulo(0.4, 0.01, (0.1, 0.1, 0.1))
    glPopMatrix()

def draw_tronco_asiento():
    glPushMatrix(); glRotate(90, 0, 1, 0); draw_cilindro_tapado(0, 360, 0.3, 1.5, (0.45, 0.28, 0.15)); glPopMatrix()

def draw_fuente_ac(t):
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 2.5, 0.3, (0.7, 0.7, 0.7))
    glTranslate(0, 0, 0.3); draw_cilindro_tapado(0, 360, 2.2, 0.1, (0.1, 0.7, 0.9))
    glTranslate(0, 0, 0.1); draw_cilindro_tapado(0, 360, 0.6, 1.0, (0.8, 0.8, 0.8))
    glTranslate(0, 0, 1.0); draw_cilindro_tapado(0, 360, 1.2, 0.2, (0.7, 0.7, 0.7))
    glTranslate(0, 0, 0.2); draw_cilindro_tapado(0, 360, 1.0, 0.1, (0.1, 0.7, 0.9))
    glPopMatrix()
    glPushMatrix(); glTranslate(0, 2.0, 0); h = abs(math.sin(t*8))*0.4; draw_sphere(0.3 + h, (0.8, 0.9, 1.0)); glPopMatrix()

def draw_faro():
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 1.0, 1.5, (0.9, 0.9, 0.9)) 
    glTranslate(0, 0, 1.5); draw_cilindro_tapado(0, 360, 0.9, 1.5, (0.8, 0.2, 0.2))
    glTranslate(0, 0, 1.5); draw_cilindro_tapado(0, 360, 0.8, 1.0, (0.9, 0.9, 0.9)) 
    glPopMatrix()
    glPushMatrix(); glTranslate(0, 4.2, 0); draw_sphere(0.5, (1.0, 1.0, 0.5)); draw_piramide(1.0, 1.5, 1.5, 0.3, (0.2, 0.2, 0.2)); glPopMatrix()

def draw_puente_ac():
    glPushMatrix()
    for i in range(-3, 4):
        glPushMatrix(); altura = 0.5 - abs(i)*0.15; glTranslate(i * 0.8, altura, 0)
        draw_cube(0.2, 0.8, 3.5, (0.6, 0.3, 0.1), (0.7, 0.4, 0.2), (0.5, 0.2, 0.1)) 
        glTranslate(0, 0.3, 1.6); draw_cube(0.4, 0.2, 0.2, (0.5, 0.2, 0.1), (0.5, 0.2, 0.1), (0.5, 0.2, 0.1))
        glTranslate(0, 0, -3.2); draw_cube(0.4, 0.2, 0.2, (0.5, 0.2, 0.1), (0.5, 0.2, 0.1), (0.5, 0.2, 0.1))
        glPopMatrix()
    glPopMatrix()

def draw_arbol_manzanas():
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.2, 1.5, (0.45, 0.28, 0.15)); glPopMatrix()
    glPushMatrix(); glTranslate(0, 1.4, 0); draw_sphere(0.8, (0.2, 0.8, 0.3)); glTranslate(0, 0.5, 0); draw_sphere(0.6, (0.3, 0.9, 0.3)); glPopMatrix()
    for fx, fy, fz in [(0.6, 1.4, 0.5), (-0.5, 1.6, 0.6), (0.2, 1.3, -0.7)]:
        glPushMatrix(); glTranslate(fx, fy, fz); draw_sphere(0.15, (0.9, 0.1, 0.1)); glPopMatrix()

def draw_palmera():
    glPushMatrix(); glRotate(260, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.15, 1.5, (0.6, 0.4, 0.2)); glPopMatrix()
    glPushMatrix(); glTranslate(0, 1.4, 0)
    for i in range(4): glRotate(90, 0, 1, 0); draw_piramide(0.1, 1.5, 0.5, 0, (0.4, 0.9, 0.2))
    glTranslate(0.2, -0.1, 0.2); draw_sphere(0.15, (0.8, 0.5, 0.2)); glTranslate(-0.4, 0, -0.2); draw_sphere(0.15, (0.8, 0.5, 0.2)); glPopMatrix()

def draw_monito_nieve():
    glPushMatrix()
    draw_sphere(0.6, (1.0, 1.0, 1.0)); glTranslate(0, 0.8, 0); draw_sphere(0.45, (1.0, 1.0, 1.0))
    glTranslate(0, 0.6, 0); draw_sphere(0.35, (1.0, 1.0, 1.0)) 
    glPushMatrix(); glTranslate(0, 0, 0.3); glRotate(90, 1, 0, 0); draw_piramide(0.3, 0.1, 0.1, 0, (1.0, 0.5, 0.0)); glPopMatrix()
    glPushMatrix(); glTranslate(-0.1, 0.1, 0.3); draw_sphere(0.05, (0, 0, 0)); glPopMatrix()
    glPushMatrix(); glTranslate(0.1, 0.1, 0.3); draw_sphere(0.05, (0, 0, 0)); glPopMatrix()
    glTranslate(0, 0.3, 0); draw_cube(0.4, 0.4, 0.4, (0.8, 0.1, 0.1), (0.8, 0.1, 0.1), (0.8, 0.1, 0.1)); glPopMatrix()

# --- ANIMALITOS  ---
def draw_perro_caminando(color, t): 
    tambaleo = math.cos(t * 8.0) * 15
    glPushMatrix(); glTranslate(0, 0.3, 0); glRotate(tambaleo, 0, 0, 1)
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.2, 0.5, color); glPopMatrix()
    glTranslate(0, 0.3, 0.3); draw_sphere(0.2, color)
    glTranslate(-0.15, 0.1, 0); draw_cube(0.3, 0.1, 0.1, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
    glTranslate(0.3, 0, 0); draw_cube(0.3, 0.1, 0.1, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1)); glPopMatrix()

def draw_gato_caminando(color, t):
    tambaleo = math.sin(t * 8.0) * 15
    glPushMatrix(); glTranslate(0, 0.3, 0); glRotate(tambaleo, 0, 0, 1)
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.2, 0.4, color); glPopMatrix()
    glTranslate(0, 0.4, 0.2); draw_sphere(0.2, color)
    glTranslate(-0.1, 0.15, 0); draw_piramide(0.15, 0.1, 0.1, 0, color)
    glTranslate(0.2, 0, 0); draw_piramide(0.15, 0.1, 0.1, 0, color); glPopMatrix()

def draw_conejo_caminando(color, t):
    tambaleo = math.cos(t * 8.0) * 15
    glPushMatrix(); glTranslate(0, 0.3, 0); glRotate(tambaleo, 0, 0, 1)
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.2, 0.4, color); glPopMatrix()
    glTranslate(0, 0.5, 0); draw_sphere(0.25, color)
    glTranslate(-0.1, 0.2, 0); draw_cube(0.4, 0.1, 0.1, color, color, color)
    glTranslate(0.2, 0, 0); draw_cube(0.4, 0.1, 0.1, color, color, color); glPopMatrix()

def draw_auto(color): 
    glPushMatrix(); glTranslate(0,0.001,0) 
    glPushMatrix(); glTranslate(0,0.4,0); draw_cube(0.3,1.2,0.6,color,color,color); glPopMatrix() 
    glPushMatrix(); glTranslate(0,0.7,0); draw_cube(0.2,0.8,0.4,(0.9,0.9,1.0),(0.9,0.9,1.0),(0.9,0.9,1.0)); glPopMatrix() 
    for x, z in [(0.3, -0.2), (0.3, 0.2), (-0.3, -0.2), (-0.3, 0.2)]: glPushMatrix(); glTranslate(x,0.2,z); draw_cilindro_tapado(0,360,0.2,0.1,(0.1,0.1,0.1)); glPopMatrix() 
    glPopMatrix() 

def draw_nube():
    glPushMatrix(); draw_sphere(1.0, (1,1,1)); glTranslate(1, 0, 0); draw_sphere(0.8, (1,1,1)); glTranslate(-2, 0, 0); draw_sphere(0.8, (1,1,1)); glPopMatrix()

def draw_globo(t, color, offset):
    glPushMatrix(); y = 4.0 + math.sin(t * 2.0 + offset) * 0.5; glTranslate(0, y, 0); draw_sphere(0.3, color)
    glDisable(GL_LIGHTING); glColor3f(1,1,1); glBegin(GL_LINES); glVertex3f(0, -0.3, 0); glVertex3f(0, -1.0, 0); glEnd(); glEnable(GL_LIGHTING)
    glPopMatrix()

def draw_regalo(t, offset):
    glPushMatrix(); y = 3.0 + math.sin(t * 2.0 + offset) * 0.5; glTranslate(0, y, 0); glRotate(t * 50, 0, 1, 0)
    draw_cube(0.4, 0.4, 0.4, (0.8, 0.8, 0.8), (0.8, 0.1, 0.1), (0.8, 0.8, 0.8)); glPopMatrix()

def draw_fogata(t):
    glPushMatrix(); glTranslate(0, 0.1, 0); glRotate(45, 0, 1, 0); draw_cilindro_tapado(0, 360, 0.1, 0.6, (0.4, 0.2, 0.1))
    glRotate(90, 0, 1, 0); draw_cilindro_tapado(0, 360, 0.1, 0.6, (0.4, 0.2, 0.1)); glTranslate(0, 0.2, 0)
    for i in range(4): glRotate(90*i + t*50, 0, 1, 0); draw_piramide(0.4, 0.2, 0.2, 0, (0.9, 0.5, 0.1))
    glPopMatrix()

# ═══════════════════════════════════════════════════════════════════════════
# NUEVOS OBJETOS ESTILO ANIMAL CROSSING
# ═══════════════════════════════════════════════════════════════════════════

def draw_buzon():
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.07, 1.0, (0.85, 0.85, 0.85)); glPopMatrix()
    glTranslate(0, 0.9, 0); draw_cube(0.5, 0.55, 0.45, (0.9, 0.15, 0.15), (0.9, 0.15, 0.15), (0.9, 0.15, 0.15))
    glTranslate(0, 0.5, 0); draw_piramide(0.15, 0.55, 0.45, 0, (0.75, 0.1, 0.1))
    glTranslate(0.3, -0.3, 0); draw_cube(0.3, 0.05, 0.2, (0.9, 0.8, 0.1), (0.9, 0.8, 0.1), (0.9, 0.8, 0.1))
    glPopMatrix()

def draw_banco_madera():
    glPushMatrix()
    color_m = (0.55, 0.33, 0.13)
    draw_cube(0.1, 1.4, 0.5, color_m, color_m, color_m)
    glTranslate(0, 0.35, -0.22); draw_cube(0.35, 1.4, 0.08, color_m, color_m, color_m)
    for px in [-0.6, 0.6]:
        for pz in [-0.18, 0.18]:
            glPushMatrix(); glTranslate(px, -0.3, pz); draw_cube(0.3, 0.08, 0.08, color_m, color_m, color_m); glPopMatrix()
    glPopMatrix()

def draw_maceta(color_flor):
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.22, 0.3, (0.7, 0.38, 0.22)); glPopMatrix()
    glTranslate(0, 0.3, 0); draw_flor(color_flor, (1.0, 0.9, 0.3))
    glPopMatrix()

def draw_tienda_nook():
    glPushMatrix()
    draw_cube(2.2, 3.5, 2.8, (0.6, 0.88, 0.72), (0.6, 0.88, 0.72), (0.55, 0.82, 0.65))
    glTranslate(0, 2.2, 0); draw_cube(0.3, 3.7, 2.9, (0.85, 0.5, 0.2), (0.85, 0.5, 0.2), (0.85, 0.5, 0.2))
    glTranslate(0, 0.3, 0); draw_piramide(1.0, 3.5, 2.8, 0, (0.8, 0.42, 0.15))
    glTranslate(0, -1.8, 1.42); draw_cube(0.3, 1.2, 0.05, (0.95, 0.88, 0.2), (0.95, 0.88, 0.2), (0.95, 0.88, 0.2))
    glTranslate(0, -0.8, 0.01); draw_cube(0.9, 0.8, 0.04, (0.5, 0.28, 0.1), (0.5, 0.28, 0.1), (0.5, 0.28, 0.1))
    glTranslate(-1.0, 0.8, 0); draw_cube(0.7, 0.7, 0.04, (0.75, 0.92, 1.0), (0.75, 0.92, 1.0), (0.75, 0.92, 1.0))
    glTranslate(2.0, 0.0, 0); draw_cube(0.7, 0.7, 0.04, (0.75, 0.92, 1.0), (0.75, 0.92, 1.0), (0.75, 0.92, 1.0))
    glPopMatrix()

def draw_museo():
    glPushMatrix()
    color_museo = (0.82, 0.78, 0.70)
    draw_cube(3.5, 5.0, 4.0, color_museo, (0.75, 0.71, 0.63), color_museo)
    for s in range(3):
        glPushMatrix(); glTranslate(0, s*0.2, 2.0 + s*0.3)
        draw_cube(0.2, 5.0 - s*0.3, 0.3, (0.88, 0.84, 0.76), (0.88, 0.84, 0.76), (0.88, 0.84, 0.76))
        glPopMatrix()
    for cx in [-1.8, -0.6, 0.6, 1.8]:
        glPushMatrix(); glTranslate(cx, 0, 2.1); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.2, 3.5, (0.88, 0.85, 0.78)); glPopMatrix()
    glTranslate(0, 3.5, 2.1); draw_piramide(0.8, 4.8, 0.5, 0, color_museo)
    glTranslate(0, 0.0, -2.1); draw_sphere(1.0, (0.5, 0.65, 0.8))
    glPopMatrix()

def draw_muelle(largo=6.0):
    glPushMatrix()
    color_m = (0.5, 0.3, 0.12)
    pasos = int(largo / 0.5)
    for i in range(pasos):
        glPushMatrix(); glTranslate(0, 0.05, i * 0.5)
        draw_cube(0.06, 1.4, 0.45, color_m, (0.55, 0.35, 0.15), color_m)
        glPopMatrix()
    for iz in [0.0, largo * 0.5, largo]:
        for ix in [-0.55, 0.55]:
            glPushMatrix(); glTranslate(ix, -0.6, iz)
            glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.08, 0.7, (0.4, 0.22, 0.08)); glPopMatrix()
    for ix in [-0.7, 0.7]:
        glPushMatrix(); glTranslate(ix, 0.4, largo * 0.25)
        draw_cube(0.15, 0.08, largo * 0.5, (0.55, 0.33, 0.13), (0.55, 0.33, 0.13), (0.55, 0.33, 0.13))
        glPopMatrix()
    glPopMatrix()

def draw_piedra_decorativa(radio=0.4):
    glPushMatrix(); glScale(1.0, 0.6, 0.85)
    draw_sphere(radio, (0.62, 0.62, 0.65))
    glPopMatrix()

def draw_mariposa(t, offset):
    aleteo = math.sin(t * 10 + offset) * 25
    glPushMatrix()
    glTranslate(0, 0.05 * math.sin(t * 3 + offset), 0)
    glRotate(aleteo, 0, 0, 1); draw_sphere(0.08, (0.9, 0.4, 0.9))
    glRotate(-2 * aleteo, 0, 0, 1); draw_sphere(0.08, (0.9, 0.4, 0.9))
    glPopMatrix()

def draw_bote_pesca():
    glPushMatrix()
    color_b = (0.55, 0.28, 0.1)
    draw_cube(0.45, 1.8, 0.9, color_b, (0.6, 0.32, 0.12), color_b)
    glTranslate(0, 0.45, 0); draw_cube(0.15, 1.6, 0.75, (0.88, 0.82, 0.72), (0.88, 0.82, 0.72), (0.88, 0.82, 0.72))
    glTranslate(0, 0.15, -0.2); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.04, 1.2, (0.45, 0.25, 0.08))
    glTranslate(0.2, 0, 0.8); draw_cube(0.3, 0.35, 0.02, (0.9, 0.2, 0.2), (0.9, 0.2, 0.2), (0.9, 0.2, 0.2))
    glPopMatrix()

def draw_campana_entrada():
    glPushMatrix()
    for px in [-0.8, 0.8]:
        glPushMatrix(); glTranslate(px, 0, 0); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.1, 2.5, (0.5, 0.3, 0.1)); glPopMatrix()
    glTranslate(0, 2.4, 0); draw_cube(0.15, 1.6, 0.15, (0.5, 0.3, 0.1), (0.5, 0.3, 0.1), (0.5, 0.3, 0.1))
    glTranslate(0, -0.4, 0); draw_sphere(0.3, (0.88, 0.75, 0.2))
    glTranslate(0, -0.25, 0); draw_sphere(0.07, (0.6, 0.5, 0.1))
    glPopMatrix()

def draw_arbol_cerezo():
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.18, 1.8, (0.45, 0.28, 0.18)); glRotate(-270, 1, 0, 0)
    glTranslate(0, 1.8, 0)
    draw_sphere(0.9, (0.98, 0.72, 0.78))
    glTranslate(0.6, 0.3, 0.3); draw_sphere(0.65, (0.95, 0.65, 0.72))
    glTranslate(-1.2, 0.1, -0.5); draw_sphere(0.7, (1.0, 0.78, 0.82))
    glPopMatrix()

def draw_letrero_pueblo():
    glPushMatrix()
    for px in [-0.7, 0.7]:
        glPushMatrix(); glTranslate(px, 0, 0); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.08, 1.6, (0.45, 0.28, 0.1)); glPopMatrix()
    glTranslate(0, 1.3, 0); draw_cube(0.45, 1.4, 0.1, (0.85, 0.62, 0.28), (0.9, 0.68, 0.32), (0.85, 0.62, 0.28))
    for sx in [-0.5, 0.5]:
        glPushMatrix(); glTranslate(sx, 0.1, 0.06); draw_sphere(0.12, (0.95, 0.3, 0.4)); glPopMatrix()
    glPopMatrix()

def draw_jardin_flores():
    colores = [(0.95, 0.2, 0.3), (0.95, 0.7, 0.1), (0.5, 0.2, 0.9), (0.2, 0.8, 0.4)]
    posiciones = [(-0.5,0,0), (0.5,0,0.3), (0,0,-0.5), (-0.3,0,0.6), (0.6,0,-0.3), (-0.6,0,0.5)]
    for i, (fx, fy, fz) in enumerate(posiciones):
        glPushMatrix(); glTranslate(fx, fy, fz)
        draw_flor(colores[i % len(colores)], (1.0, 0.9, 0.3))
        glPopMatrix()

def draw_pozo():
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.6, 0.5, (0.65, 0.65, 0.68))
    draw_cilindro_tapado(0, 360, 0.5, 0.5, (0.55, 0.55, 0.58)); glRotate(-270, 1, 0, 0)
    for px in [-0.55, 0.55]:
        glPushMatrix(); glTranslate(px, 0, 0); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.06, 1.4, (0.5, 0.3, 0.1)); glPopMatrix()
    glTranslate(0, 1.3, 0); draw_piramide(0.7, 1.4, 0.8, 0, (0.75, 0.42, 0.18))
    glPopMatrix()

def draw_hoguera_playa(t):
    glPushMatrix(); glTranslate(0, 0.05, 0)
    for i in range(6):
        ang = math.radians(i * 60)
        glPushMatrix(); glTranslate(math.cos(ang)*0.4, 0, math.sin(ang)*0.4)
        glScale(1, 0.5, 1); draw_sphere(0.13, (0.55, 0.55, 0.58)); glPopMatrix()
    glPushMatrix(); glRotate(30, 0, 1, 0); glRotate(270, 1, 0, 0)
    draw_cilindro_tapado(0, 360, 0.06, 0.5, (0.4, 0.22, 0.08)); glPopMatrix()
    glPushMatrix(); glRotate(-30, 0, 1, 0); glRotate(270, 1, 0, 0)
    draw_cilindro_tapado(0, 360, 0.06, 0.5, (0.4, 0.22, 0.08)); glPopMatrix()
    h = 0.3 + abs(math.sin(t * 7)) * 0.2
    glPushMatrix(); glTranslate(0, 0.3, 0); glScale(1, h*2, 1)
    draw_sphere(0.18, (1.0, 0.55, 0.1)); glPopMatrix()
    glPushMatrix(); glTranslate(0, 0.45, 0); glScale(0.6, h*1.5, 0.6)
    draw_sphere(0.18, (1.0, 0.85, 0.1)); glPopMatrix()
    glPopMatrix()

def draw_silla_playa():
    glPushMatrix()
    color_s = (0.2, 0.55, 0.85)
    glRotate(-15, 1, 0, 0); draw_cube(0.08, 0.9, 0.55, color_s, color_s, color_s)
    glTranslate(0, 0.08, -0.25); glRotate(-40, 1, 0, 0)
    draw_cube(0.08, 0.9, 0.5, color_s, color_s, color_s)
    glPopMatrix()

def draw_lampara_jardin():
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.12, 0.08, (0.4, 0.4, 0.45))
    draw_cilindro_tapado(0, 360, 0.05, 1.4, (0.5, 0.5, 0.55)); glRotate(-270, 1, 0, 0)
    glTranslate(0, 1.35, 0); draw_sphere(0.2, (0.42, 0.42, 0.48))
    draw_sphere(0.12, (1.0, 0.95, 0.7))
    glPopMatrix()

# ═══════════════════════════════════════════════════════════════════════════
# OBJETOS TOMADOS / ADAPTADOS DEL PROYECTO 3
# ═══════════════════════════════════════════════════════════════════════════


def draw_perro_detallado(color):
    global ang_perro3, dir_perro3
    ang_perro3 += dir_perro3
    if ang_perro3 >= 60:  dir_perro3 = -12.0
    if ang_perro3 <= -60: dir_perro3 =  12.0
    glPushMatrix()
    glPushMatrix(); glTranslate(0, 0.21, 0.3); draw_sphere(0.08, color); glPopMatrix()
    glPushMatrix(); glTranslate(-0.02, 0.27, 0.3); draw_piramide(0.021, 0.2, 0.05, 0, color); glPopMatrix()
    glPushMatrix(); glTranslate( 0.02, 0.27, 0.3); draw_piramide(0.021, 0.2, 0.05, 0, color); glPopMatrix()
    glPushMatrix(); glTranslate(0, 0.15, 0); draw_cilindro_tapado(0, 360, 0.06, 0.3, color); glPopMatrix()
    for sx, sz, sign in [(0.06,0,-1),(-0.06,0.3,-1),(-0.06,0,1),(0.06,0.3,1)]:
        glPushMatrix(); glTranslate(sx, 0.15, sz); glRotate(90,1,0,0); glRotate(sign*ang_perro3,1,0,0)
        draw_cilindro_tapado(0,360,0.03,0.15,color); glPopMatrix()
    glPopMatrix()

def draw_pelicano(color):
    """¡VERSIÓN ACTUALIZADA! Pelícano con alas 'menos pros' (planas)."""
    global ang_peli, dir_peli
    ang_peli += dir_peli
    if ang_peli >= 30:  dir_peli = -10.0
    if ang_peli <= -30: dir_peli =  10.0
    glPushMatrix(); glRotate(90, 0, 1, 0)
    # Cabeza (esfera simple)
    glPushMatrix(); glTranslate(0,0.43,0); draw_sphere(0.06, color); glPopMatrix()
    # Pico
    glPushMatrix(); glTranslate(0,0.43,0.05); glRotate(110,1,0,0); draw_piramide(0.15,0.05,0.05,0,(1.0,0.5,0.0)); glPopMatrix()
    # Cuello
    glPushMatrix(); glTranslate(0,0.43,0); glRotate(140,1,0,0); draw_cilindro_tapado(0,360,0.04,0.25,color); glPopMatrix()
    # Cuerpo
    glPushMatrix(); glTranslate(0,0.17,-0.25); draw_sphere(0.12, color); glPopMatrix()
    
    # ALAS "MENOS PROS" (Tablas planas 2D que aletean como cartón)
    glPushMatrix(); glTranslate(0,0.17,-0.25); glRotate(ang_peli,0,0,1); glTranslate(0.2,0,0)
    draw_rectangulo(0.2, 0.1, color); glPopMatrix()
    
    glPushMatrix(); glTranslate(0,0.17,-0.25); glRotate(-ang_peli,0,0,1); glTranslate(-0.2,0,0)
    draw_rectangulo(0.2, 0.1, color); glPopMatrix()
    
    glPopMatrix()

def draw_pollo(color):
    global ang_pollo, dir_pollo
    ang_pollo += dir_pollo
    if ang_pollo >= 30:  dir_pollo = -10.0
    if ang_pollo <= -30: dir_pollo =  10.0
    glPushMatrix(); glRotate(90,0,1,0); glScale(0.75,0.75,0.75)
    glPushMatrix(); glTranslate(0,0.17,-0.25); draw_sphere(0.12,color); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.15,-0.16); draw_cube(0.05,0.2,0.1,(1,0.5,0),(1,0.5,0),(1,0.5,0)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.15,-0.25); glRotate(ang_pollo,0,0,1); glTranslate(0.17,0,0)
    draw_cube(0.05,0.2,0.2,color,color,color); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.17,-0.25); glRotate(180-ang_pollo,0,0,1); glRotate(180,0,1,0); glTranslate(-0.15,0,0)
    draw_cube(0.05,0.2,0.2,color,color,color); glPopMatrix()
    glPopMatrix()

def draw_pez_animado(color):
    global ang_pez
    ang_pez += 3.0
    if ang_pez > 360: ang_pez -= 360
    col_ang = math.sin(math.radians(ang_pez)) * 30
    glPushMatrix(); glRotate(180,0,1,0)
    glPushMatrix(); glTranslate(0,0.1,0); draw_sphere(0.1,color); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.12,0); glRotate(col_ang,0,1,0); glTranslate(0,0,0.2); glRotate(-90,1,0,0)
    draw_piramide(0.1,0.05,0.2,0,color); glPopMatrix()
    glPushMatrix(); glTranslate(0.05,0.14,-0.07); draw_sphere(0.025,(0,0,0)); glPopMatrix()
    glPopMatrix()

def draw_barquito(color, color2):
    glPushMatrix(); glRotate(90,0,1,0)
    glPushMatrix(); glTranslate(0,0.5,0); draw_cilindro_tapado(180,360,0.5,0.5,color); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.5,0); glRotate(270,1,0,0); draw_cilindro_tapado(0,360,0.04,1.2,(0.45,0.25,0.08)); glPopMatrix()
    glPushMatrix(); glTranslate(0,1.2,0); draw_piramide(0.8,0.5,0.02,0,(0.95,0.95,0.95)); glPopMatrix()
    glPopMatrix()

def draw_camion(color):
    glPushMatrix()
    glPushMatrix(); glTranslate(0,0.5,0); draw_cube(0.7,2.0,0.7,color,color,color); glPopMatrix()
    for tx,tz in [(0.8,-0.4),(0.8,0.2),(-0.8,-0.4),(-0.8,0.2)]:
        glPushMatrix(); glTranslate(tx,0.25,tz); draw_cilindro_tapado(0,360,0.25,0.1,(0.15,0.15,0.15)); glPopMatrix()
    for tz in [-0.2, 0.2]:
        glPushMatrix(); glTranslate(-1,0.7,tz); draw_sphere(0.05,(1,1,0)); glPopMatrix()
    glPopMatrix()

def draw_moto(color):
    glPushMatrix()
    glPushMatrix(); glTranslate(0,0.3,0); draw_cube(0.2,1.0,0.3,color,color,color); glPopMatrix()
    glPushMatrix(); glTranslate(-0.4,0.5,0); draw_cube(0.35,0.4,0.3,color,color,color); glPopMatrix()
    for tx,tz in [(0.3,-0.1),(0.3,0.1),(-0.3,-0.1),(-0.3,0.1)]:
        glPushMatrix(); glTranslate(tx,0.2,tz); draw_cilindro_tapado(0,360,0.15,0.1,(0.1,0.1,0.1)); glPopMatrix()
    glPushMatrix(); glTranslate(-0.6,0.65,0); draw_sphere(0.1,(0,1,1)); glPopMatrix()
    glPopMatrix()

def draw_cocodrilo(color, ang_boca, dir_b):
    ang_boca += dir_b
    if ang_boca >= 30:  dir_b = -10.0
    if ang_boca <= -30: dir_b =  10.0
    glPushMatrix()
    draw_cube(0.15, 0.4, 1.2, color, color, color)
    glPushMatrix(); glTranslate(0, 0.08, 0.65)
    draw_cube(0.12, 0.35, 0.4, color, color, color)
    glPushMatrix(); glTranslate(0, -0.06, 0.2); draw_cube(0.06, 0.3, 0.22, color, color, color); glPopMatrix()
    glPushMatrix(); glTranslate(0, 0.06, 0); glRotate(ang_boca, 1, 0, 0)
    glTranslate(0, 0, 0.2); draw_cube(0.06, 0.3, 0.22, color, color, color); glPopMatrix()
    glPopMatrix()
    for sx, sz in [( 0.25,  0.3),( 0.25, -0.3),(-0.25,  0.3),(-0.25, -0.3)]:
        glPushMatrix(); glTranslate(sx, 0, sz); glRotate(270,1,0,0)
        draw_cilindro_tapado(0,360,0.05,0.2,color); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.05,-0.6); draw_piramide(0.1,0.25,0.8,0,color); glPopMatrix()
    glPopMatrix()
    return ang_boca, dir_b

def draw_abeja():
    global mov_abeja, dir_abeja
    mov_abeja += dir_abeja
    if mov_abeja <= 0: dir_abeja =  0.05
    if mov_abeja >= 2: dir_abeja = -0.05
    glPushMatrix(); glTranslate(0, mov_abeja, 0); glRotate(270,0,1,0); glScale(0.1,0.1,0.1)
    glTranslate(0,10,0)
    glPushMatrix(); glTranslate(0.56,0,0); draw_sphere(1,(1,1,0)); glPopMatrix()
    glPushMatrix(); glRotate(180,1,0,1); draw_cilindro_tapado(0,360,1.001,1,(0,0,0)); glPopMatrix()
    for tz in [-0.5,0.5]:
        glPushMatrix(); glTranslate(1.3,0.4,tz); draw_sphere(0.2,(0,0,0)); glPopMatrix()
    glPushMatrix(); glTranslate(0.3,0,0.5); glRotate(45,1,1,1); glTranslate(0.3,0.6,-0.5)
    draw_cilindro_tapado(0,360,0.7,0.7,(1,1,1)); glPopMatrix()
    glPopMatrix()

def draw_caracol(color1, color2):
    glPushMatrix()
    glTranslate(0,0.07,0); draw_cilindro_tapado(0,360,0.07,0.2,color2)
    glTranslate(0,0,0.13); glRotate(-45,1,0,0); draw_cilindro_tapado(0,360,0.07,0.2,color2)
    glPopMatrix()
    for sx in [-0.03, 0.03]:
        glPushMatrix(); glTranslate(sx,0.15,0.24); glRotate(270,1,0,0)
        draw_cilindro_tapado(0,360,0.02,0.2,color2); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.11,0.05); draw_sphere(0.1,color1); glPopMatrix()

def draw_nube_grande(color=(0.9,0.9,0.95)):
    glPushMatrix()
    for dx_,dy_,dz_ in [(0,1,-0.5),(0,1,0.5),(0,2,0),(-1,1,-0.5),(-1,1,0.5),(-1,2,0),(-2,1,0),(1,1,-0.5),(1,1,0.5),(1,2,0),(2,1,0)]:
        glPushMatrix(); glTranslate(dx_,dy_,dz_); draw_sphere(1,color); glPopMatrix()
    glPopMatrix()

def draw_sol_luna_animado():
    global angulo_dia, r_cielo, g_cielo, b_cielo
    if 0 < angulo_dia < math.radians(70):
        g_cielo = min(1.0, g_cielo + 0.002); b_cielo = min(1.0, b_cielo + 0.002)
    elif math.radians(110) < angulo_dia < math.radians(180):
        g_cielo = max(0.1, g_cielo - 0.002); b_cielo = max(0.1, b_cielo - 0.002)
    elif math.radians(180) < angulo_dia < math.radians(360):
        g_cielo = max(0.0, g_cielo - 0.001); b_cielo = max(0.0, b_cielo - 0.001)
    glPushMatrix(); glTranslate(-15,0,0); glRotate(90,1,0,0)
    mover_en_circulo(lambda: draw_sphere(4,(0.75,0.75,0.75)), 55, angulo_dia, 0, 0, 0)
    mover_en_circulo(lambda: draw_sphere(7,(1.0,0.92,0.2)), 55, angulo_dia, math.pi, 0, 0)
    glPopMatrix()
    angulo_dia += 0.005
    if angulo_dia >= math.radians(360): angulo_dia = 0.0

def draw_ola_agua(t):
    h_ola = abs(math.sin(t*1.5)) * 0.3
    glPushMatrix(); glTranslate(0, h_ola, 0)
    draw_rectangulo(50, 0.3, (0.2, 0.6, 0.95))
    glPopMatrix()

def draw_casa_proyecto3(color1, color2, color3, color4):
    glPushMatrix()
    draw_cube(2,2,2,color1,color2,color3)
    draw_piramide(2,2,2,2,color4)
    for vx in [-0.5, 0.5]:
        glPushMatrix(); glTranslate(vx,1.5,1.002); glRotate(90,1,0,0)
        draw_rectangulo(0.25,0.25,(0,0.75,0.75)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0.5,1.002); glRotate(90,1,0,0)
    draw_rectangulo(0.25,0.5,(0.55,0.25,0.08)); glPopMatrix()
    glPopMatrix()

# -------------------------------------------------------------------------
# ENSAMBLAJE DE LA CIUDAD
# -------------------------------------------------------------------------
def draw_mundo(): 
    global ang_coco1, dir_coco1, ang_coco2, dir_coco2, r_cielo, g_cielo, b_cielo
    t = glfw.get_time()

    glClearColor(r_cielo, g_cielo, b_cielo, 1.0)
    glPushMatrix(); glTranslatef(0,-0.05,0); draw_rectangulo(80,80,(0.08,0.42,0.82)); glPopMatrix()
    draw_ola_agua(t)

    glPushMatrix(); draw_rectangulo(9,9,(0.48,0.82,0.38)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,-22); draw_rectangulo(16,12,(0.38,0.78,0.28)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0, 22); draw_rectangulo(16,12,(0.92,0.82,0.55)); glPopMatrix()
    glPushMatrix(); glTranslate(24,0,0);  draw_rectangulo(14,16,(0.42,0.78,0.32)); glPopMatrix()
    glPushMatrix(); glTranslate(-24,0,0); draw_rectangulo(14,16,(0.42,0.78,0.32)); glPopMatrix()

    glPushMatrix(); glTranslate(0,0,-10); glRotate(90,0,1,0); draw_puente_ac(); glPopMatrix()
    glPushMatrix(); glTranslate(0,0, 10); glRotate(90,0,1,0); draw_puente_ac(); glPopMatrix()
    glPushMatrix(); glTranslate( 10,0,0); draw_puente_ac(); glPopMatrix()
    glPushMatrix(); glTranslate(-10,0,0); draw_puente_ac(); glPopMatrix()

    glPushMatrix(); glTranslate(0,0.01,0); draw_rectangulo(5,5,(0.70,0.70,0.70)); glPopMatrix()
    draw_fuente_ac(t)

    for lx,lz in [(-4,4),(4,4),(4,-4),(-4,-4)]:
        glPushMatrix(); glTranslate(lx,0,lz); draw_lampara_jardin(); glPopMatrix()

    for bx,bz,br in [(-3,0,0),(3,0,180),(0,-3,90),(0,3,-90)]:
        glPushMatrix(); glTranslate(bx,0,bz); glRotate(br,0,1,0); draw_banco_madera(); glPopMatrix()
    for mx,mz,mc in [(-2,2,(0.9,0.2,0.3)),(2,2,(0.2,0.7,0.9)),(2,-2,(0.9,0.8,0.1)),(-2,-2,(0.6,0.2,0.9))]:
        glPushMatrix(); glTranslate(mx,0,mz); draw_maceta(mc); glPopMatrix()

    glPushMatrix(); glTranslate(-6,0,0); draw_buzon(); glPopMatrix()
    glPushMatrix(); glTranslate( 6,0,0); draw_buzon(); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,7); draw_letrero_pueblo(); glPopMatrix()

    mover_en_circulo(lambda: draw_perro_caminando((0.8,0.4,0.2),t), 5.5, t*0.6, 0,    90)
    mover_en_circulo(lambda: draw_perro_caminando((0.2,0.2,0.2),t), 6.5, t*0.4, 3.14,-90)
    mover_en_circulo(lambda: draw_gato_caminando((0.9,0.9,0.9),t),  7.5, t*0.5, 1.5,  90)
    mover_en_circulo(lambda: draw_conejo_caminando((0.9,0.6,0.6),t),4.5, t*0.7, 4.5, -90)

    mover_en_circulo(lambda: draw_perro_detallado((0.6,0.3,0.15)), 10.0, t*0.55, 1.0, 90)

    for jx,jz in [(-7,0),(7,0),(0,-7),(0,7),(-6,4),(6,-4),(-5,-5),(5,5)]:
        glPushMatrix(); glTranslate(jx,0,jz); draw_jardin_flores(); glPopMatrix()

    for mi,(mx,mz) in enumerate([(-3,3),(3,-3),(0,4),(-4,0)]):
        glPushMatrix(); glTranslate(mx,1.2,mz); draw_mariposa(t,mi*1.5); glPopMatrix()

    glPushMatrix(); glTranslate(-2,0,2); glScale(3,3,3); draw_abeja(); glPopMatrix()

    for cx,cz in [(3,4),(-3,-4),(4,-3)]:
        glPushMatrix(); glTranslate(cx,0,cz); draw_caracol((0.6,0.3,0.5),(0.9,0.8,0.6)); glPopMatrix()

    glPushMatrix(); glTranslate(-5,0,-5); draw_pozo(); glPopMatrix()

    glPushMatrix(); glTranslate(0,0,-22)
    glPushMatrix(); glTranslate(0,0,-5); glScale(0.6,0.6,0.6); draw_edificio_reloj(t); glPopMatrix()
    glPushMatrix(); glTranslate(9,0,-2); glScale(0.65,0.65,0.65); draw_museo(); glPopMatrix()

    glPushMatrix(); glTranslate(-8,0, 2); draw_casa_zen((0.9,0.9,0.9),(0.8,0.3,0.3)); glPopMatrix()
    glPushMatrix(); glTranslate(-8,0, 4); draw_cerca(); glPopMatrix()
    glPushMatrix(); glTranslate(-8,0,-5); draw_casa_colorida((0.9,0.9,0.2),(0.2,0.6,0.9)); glPopMatrix()
    glPushMatrix(); glTranslate( 8,0, 2); draw_casa_zen((0.9,0.9,0.9),(0.3,0.8,0.3)); glPopMatrix()
    glPushMatrix(); glTranslate( 8,0, 4); draw_cerca(); glPopMatrix()
    glPushMatrix(); glTranslate( 8,0,-5); draw_casa_colorida((0.9,0.2,0.9),(0.2,0.9,0.6)); glPopMatrix()

    glPushMatrix(); glTranslate(-13,0,-3); glScale(0.7,0.7,0.7); draw_casa_proyecto3((0.75,0.6,0),(0.6,0.2,0.2),(0.53,0.81,1),(0.6,0.2,0.2)); glPopMatrix()
    glPushMatrix(); glTranslate( 13,0,-3); glScale(0.7,0.7,0.7); draw_casa_proyecto3((0.8,0.75,0.9),(0.5,0.2,0.5),(0.7,0.9,1),(0.5,0.2,0.5)); glPopMatrix()
    glPushMatrix(); glTranslate(-13,0, 5); glScale(0.7,0.7,0.7); draw_casa_proyecto3((0.6,0.8,0.5),(0.2,0.5,0.2),(0.53,0.81,1),(0.2,0.5,0.2)); glPopMatrix()
    glPushMatrix(); glTranslate( 13,0, 5); glScale(0.7,0.7,0.7); draw_casa_proyecto3((0.9,0.7,0.5),(0.6,0.3,0.1),(0.53,0.81,1),(0.6,0.3,0.1)); glPopMatrix()

    for tx,tz in [(-5,-8),(-10,-8),(5,-8),(10,-8),(-3,6),(3,6),(-8,0),(8,0)]:
        glPushMatrix(); glTranslate(tx,0,tz); draw_arbol_cerezo(); glPopMatrix()

    for lx,lz in [(-5,0),(5,0),(0,-9),(-9,0),(9,0)]:
        glPushMatrix(); glTranslate(lx,0,lz); draw_lampara_jardin(); glPopMatrix()

    mover_en_circulo(lambda: draw_camion((0.8,0.2,0.2)), 13.0, t*0.3, 0, 0)
    mover_en_circulo(lambda: draw_moto((0.2,0.2,0.9)),   11.0, t*0.5, 1.8, 0)
    mover_en_circulo(lambda: draw_auto((0.2,0.8,0.2)),   12.0, t*0.35,3.5, 0)
    glPopMatrix() 

    glPushMatrix(); glTranslate(0,0,22)
    glPushMatrix(); glTranslate(0,0,7); draw_faro(); glPopMatrix()
    glPushMatrix(); glTranslate(5,0,4); glRotate(90,0,1,0); draw_muelle(7.0); glPopMatrix()
    glPushMatrix(); glTranslate(d_barco,0,6); draw_barquito((0.55,0.28,0.1),(0.9,0.7,0.5)); glPopMatrix()

    for sx,sz in [(-5,2),(-2,3),(2,2),(6,1),(-7,3)]:
        glPushMatrix(); glTranslate(sx,0,sz); draw_sombrilla(); glPopMatrix()
    for tx,tz in [(-5,4),(-2,5),(2,4),(6,3),(-7,5)]:
        glPushMatrix(); glTranslate(tx,0,tz); draw_toalla_playa(); glPopMatrix()
    for sx,sz in [(-3,3),(0,3),(4,2)]:
        glPushMatrix(); glTranslate(sx,0,sz); draw_silla_playa(); glPopMatrix()

    glPushMatrix(); glTranslate(-4,0,5); draw_hoguera_playa(t); glPopMatrix()

    for x in range(-10,11,3):
        glPushMatrix(); glTranslate(x,0,-2); draw_palmera(); glPopMatrix()
    for x in [-9,-5,5,9]:
        glPushMatrix(); glTranslate(x,0,3); draw_palmera(); glPopMatrix()

    for px,pz,pr in [(-8,4,0.5),(-6,6,0.35),(7,5,0.4),(9,3,0.45),(-3,7,0.3)]:
        glPushMatrix(); glTranslate(px,0,pz); draw_piedra_decorativa(pr); glPopMatrix()

    mover_en_circulo(lambda: draw_pelicano((0.9,0.9,0.9)), 6.0, t*0.4, 0, 0, 4.0)
    mover_en_circulo(lambda: draw_pelicano((0.85,0.82,0.75)), 8.0, t*0.35, 2.0, 0, 5.0)

    for pi,(fx,fz) in enumerate([(2,8),(5,9),(-3,10),(0,11)]):
        glPushMatrix(); glTranslate(fx,-0.3,fz); draw_pez_animado([(0.9,0.4,0.1),(0.2,0.5,0.9),(0.9,0.9,0.2),(0.5,0.2,0.8)][pi%4]); glPopMatrix()

    glPushMatrix(); glTranslate(0,0,-1); draw_campana_entrada(); glPopMatrix()
    glPopMatrix()  

    glPushMatrix(); glTranslate(24,0,0)
    glPushMatrix(); glTranslate(0,0,-8); glScale(0.65,0.65,0.65); draw_tienda_nook(); glPopMatrix()
    draw_fogata(t)
    glPushMatrix(); glTranslate(0,0,-4);   draw_tienda_campana((0.9,0.9,0.2)); glPopMatrix()
    glPushMatrix(); glTranslate(3,0,2);    glRotate(-45,0,1,0); draw_tienda_campana((0.2,0.6,0.9)); glPopMatrix()
    glPushMatrix(); glTranslate(-3,0,2);   glRotate(45,0,1,0);  draw_tienda_campana((0.9,0.3,0.4)); glPopMatrix()
    glPushMatrix(); glTranslate(4,0,-4);   glRotate(-20,0,1,0); draw_tienda_campana((0.4,0.8,0.4)); glPopMatrix()

    glPushMatrix(); glTranslate(-1.5,0,0); draw_tronco_asiento(); glPopMatrix()
    glPushMatrix(); glTranslate( 1.5,0,0); draw_tronco_asiento(); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,3); draw_banco_madera(); glPopMatrix()

    glPushMatrix(); glTranslate(-5,0,8);  draw_huerto(); glPopMatrix()
    glPushMatrix(); glTranslate(4,0,8);   draw_huerto(); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,10);  draw_huerto(); glPopMatrix()

    for i in range(5):
        glPushMatrix(); glTranslate(i*3-6,0,-12); draw_arbol_manzanas(); glPopMatrix()

    glPushMatrix(); glTranslate(-5,0.002,4); glRotate(45,0,1,0)
    ang_coco1, dir_coco1 = draw_cocodrilo((0.1,0.4,0.1), ang_coco1, dir_coco1)
    glPopMatrix()
    glPushMatrix(); glTranslate(5,0.002,5); glRotate(-50,0,1,0)
    ang_coco2, dir_coco2 = draw_cocodrilo((0.15,0.5,0.15), ang_coco2, dir_coco2)
    glPopMatrix()

    mover_en_circulo(lambda: draw_pollo((0.9,0.85,0.7)), 4.0, t*0.8, 0, 0)
    mover_en_circulo(lambda: draw_pollo((0.7,0.5,0.3)),  3.0, t*0.9, 1.5, 0)

    for lx,lz in [(-4,0),(4,0),(0,-6),(0,5)]:
        glPushMatrix(); glTranslate(lx,0,lz); draw_lampara_jardin(); glPopMatrix()

    for mi,(mx,mz) in enumerate([(2,4),(-2,5),(0,2)]):
        glPushMatrix(); glTranslate(mx,1.5,mz); draw_mariposa(t,mi*2.1); glPopMatrix()

    for hx,hz,hc in [(6,6,(0.9,0.2,0.2)),(-6,6,(0.9,0.9,0.2))]:
        glPushMatrix(); glTranslate(hx,0,hz); draw_hongo(hc); glPopMatrix()
    glPopMatrix()  

    glPushMatrix(); glTranslate(-24,0,0)
    glPushMatrix(); glTranslate(0,0,-3);  draw_puesto_mercado((0.2,0.8,0.3)); glPopMatrix()
    glPushMatrix(); glTranslate(3,0,0);   glRotate(-90,0,1,0); draw_puesto_mercado((0.9,0.2,0.2)); glPopMatrix()
    glPushMatrix(); glTranslate(-3,0,0);  glRotate( 90,0,1,0); draw_puesto_mercado((0.2,0.4,0.9)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,4);   draw_puesto_mercado((0.9,0.8,0.1)); glPopMatrix()
    glPushMatrix(); glTranslate(-3,0,-6); glRotate(20,0,1,0);  draw_puesto_mercado((0.8,0.3,0.7)); glPopMatrix()

    glPushMatrix(); glTranslate(-5,0,-10); draw_casa_colorida((0.8,0.8,0.9),(0.4,0.4,0.5)); glPopMatrix()
    glPushMatrix(); glTranslate(-5,0,-8);  draw_cerca(); glPopMatrix()
    glPushMatrix(); glTranslate(5,0,-10);  draw_casa_zen((0.95,0.88,0.75),(0.55,0.38,0.18)); glPopMatrix()
    glPushMatrix(); glTranslate(-8,0,4);   draw_casa_zen((0.92,0.85,0.78),(0.6,0.4,0.15)); glPopMatrix()
    glPushMatrix(); glTranslate(0,0,10);   glScale(0.7,0.7,0.7); draw_casa_proyecto3((0.7,0.85,0.95),(0.2,0.4,0.75),(0.53,0.81,1),(0.2,0.4,0.75)); glPopMatrix()

    for ax,az in [(-4,-4),(-2,-4),(2,-4),(4,-4),(-4,4),(-2,4),(2,4),(4,4),(4,-2),(-4,-2),(-6,0),(6,0)]:
        glPushMatrix(); glTranslate(ax,0,az); draw_arbusto(); glPopMatrix()
    for hx,hz,hc in [(5,5,(0.9,0.2,0.2)),(-5,5,(0.9,0.9,0.2)),(5,-5,(0.2,0.4,0.9)),(-5,-5,(0.8,0.4,0.2))]:
        glPushMatrix(); glTranslate(hx,0,hz); draw_hongo(hc); glPopMatrix()

    for lx,lz in [(-6,-6),(6,-6),(6,6),(-6,6),(0,-8),(0,8)]:
        glPushMatrix(); glTranslate(lx,0,lz); draw_lampara_jardin(); glPopMatrix()

    glPushMatrix(); glTranslate(0,0,-8); draw_pozo(); glPopMatrix()

    mover_en_circulo(lambda: draw_auto((0.8,0.6,0.1)), 10.0, t*0.42, 0.7, 0)

    for jx,jz in [(-7,0),(7,0),(0,-12),(0,12)]:
        glPushMatrix(); glTranslate(jx,0,jz); draw_jardin_flores(); glPopMatrix()

    for cx,cz,c1,c2 in [(2,6,(0.8,0.2,0.4),(0.95,0.85,0.7)),(-2,-6,(0.3,0.6,0.2),(0.9,0.8,0.6))]:
        glPushMatrix(); glTranslate(cx,0,cz); draw_caracol(c1,c2); glPopMatrix()
    glPopMatrix()  

    draw_sol_luna_animado()

    for i in range(8):
        glPushMatrix()
        glScale(1.5,1.0,1.5)
        glTranslatef(-60 + ((t*1.5 + i*18) % 120), 12, -40 + i*12)
        draw_nube_grande((0.9,0.9,0.95))
        glPopMatrix()

    colores_globo = [(0.9,0.2,0.2),(0.2,0.9,0.2),(0.2,0.2,0.9),(0.9,0.9,0.2),(0.9,0.2,0.9),(0.2,0.9,0.9)]
    for i in range(12):
        x = -55 + ((t*4 + i*11) % 110)
        glPushMatrix(); glTranslatef(x,0,-25+i*6)
        draw_globo(t, colores_globo[i%len(colores_globo)], i)
        draw_regalo(t, i)
        glPopMatrix()

    mover_en_circulo(lambda: draw_pelicano((0.85,0.82,0.72)), 30.0, t*0.2, 0.5, 0, 8.0)
    mover_en_circulo(lambda: draw_pelicano((0.9,0.88,0.80)),  25.0, t*0.25,2.5, 0, 10.0)

# -------------------------------------------------------------------------
# BUCLE PRINCIPAL CON CÁMARA (MEDIAPIPE TASKS)
# -------------------------------------------------------------------------
def main(): 
    global moverx, movery, moverz, dx, dz, girar 
    try: window = init_glfw() 
    except Exception as e: print(f"Error al inicializar GLFW: {e}"); return 

    setup_opengl(); setup_lights() 
    frame_count, fps_timer = 0, glfw.get_time() 

    try: 
        cap = cv2.VideoCapture(0); cv2.waitKey(2000) 

        while not glfw.window_should_close(window): 
            ret, frame = cap.read() 
            if not ret: break 
            
            frame2 = frame.copy() 
            h, w, _ = frame.shape 
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            detection_result = hand_landmarker.detect(mp_image)

            vista_aerea = False 
            estado_txt  = "Modo Normal"
            color_txt   = (0, 255, 0) # Verde

            if detection_result.hand_landmarks: 
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # REVISAR QUÉ GESTO HACE EL USUARIO
                vista_aerea = es_paz(hand_landmarks)
                avanzar     = es_palma_abierta(hand_landmarks)
                retroceder  = es_puno(hand_landmarks)
                
                # POSICIÓN DE LA MUÑECA PARA EL VOLANTE
                muneca_x = hand_landmarks[0].x

                # Dibujar esqueleto en la cámara (Líneas y Nodos)
                for conexion in HAND_CONNECTIONS:
                    p1 = hand_landmarks[conexion[0]]
                    p2 = hand_landmarks[conexion[1]]
                    cx1, cy1 = int(p1.x * w), int(p1.y * h)
                    cx2, cy2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame2, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2) # Líneas Verdes
                
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame2, (cx, cy), 4, (0, 0, 255), -1) # Puntos Rojos
                
                # LÓGICA DE MOVIMIENTO
                if vista_aerea:
                    estado_txt = "DRON ACTIVADO"
                    color_txt = (255, 0, 255) # Rosa
                else:
                    if avanzar:
                        moverx += dx * 0.8
                        moverz += dz * 0.8
                        estado_txt = "AVANZANDO"
                        color_txt = (255, 255, 0) # Celeste
                    elif retroceder:
                        moverx -= dx * 0.8
                        moverz -= dz * 0.8
                        estado_txt = "RETROCEDIENDO"
                        color_txt = (0, 0, 255) # Rojo
                    
                    # VOLANTE: Si la mano está en la izq o der de la cámara
                    if muneca_x < 0.4:
                        girar -= 0.06
                        estado_txt += " + GIRANDO IZQ"
                    elif muneca_x > 0.6:
                        girar += 0.06
                        estado_txt += " + GIRANDO DER"
                     
            # Mostrar estado en la cámara
            cv2.putText(frame2, estado_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_txt, 3)
            cv2.imshow("Camara de Control", frame2) 
            if cv2.waitKey(1) & 0xFF == ord('q'): break 

            if key.is_pressed('up') or key.is_pressed('w'): moverx += dx*0.8; moverz += dz*0.8  
            if key.is_pressed('down') or key.is_pressed('s'): moverx -= dx*0.8; moverz -= dz*0.8  
            if key.is_pressed('left') or key.is_pressed('a'): girar -= 0.08 
            if key.is_pressed('right') or key.is_pressed('d'): girar += 0.08 

            dx, dz = math.sin(girar), -math.cos(girar)     
            glfw.poll_events() 
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: break 

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 
            glMatrixMode(GL_PROJECTION); glLoadIdentity() 
            gluPerspective(60, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 400.0)  
            glMatrixMode(GL_MODELVIEW); glLoadIdentity() 
            
            # CÁMARA CON EFECTO ZOOM
            if vista_aerea:
                # Dron a 45 de altura para dar efecto de "Zoom In"
                gluLookAt(moverx, 30.0, moverz, moverx, 0.0, moverz - 0.1, 0, 1, 0) 
            else:
                gluLookAt(moverx, movery, moverz, moverx+dx, movery, moverz+dz, 0, 1, 0) 

            draw_mundo() 
            glfw.swap_buffers(window) 

            frame_count += 1 
            current_time = glfw.get_time() 
            if current_time - fps_timer >= 1.0: 
                fps = frame_count / (current_time - fps_timer) 
                glfw.set_window_title(window, f"{WINDOW_TITLE} - FPS: {fps:.1f}") 
                frame_count, fps_timer = 0, current_time 

    except Exception as e: 
        print(f"Error en el loop: {e}") 
    finally: 
        cap.release(); cv2.destroyAllWindows(); glfw.terminate() 

if __name__ == "__main__": 
    main()