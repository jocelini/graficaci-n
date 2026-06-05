import os
import urllib.request
import ssl
import glfw 
import cv2 
import numpy as np 
from OpenGL.GL import * 
from OpenGL.GLU import * 
import math 
import keyboard as key  
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Descargando el modelo necesario de MediaPipe, espera unos segundos...")
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Descarga completada exitosamente.")
    except Exception as e:
        print(f"Error en la descarga automatica: {e}")

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

WINDOW_WIDTH = 800 
WINDOW_HEIGHT = 600 
WINDOW_TITLE = "Proyecto 2: ciudad 3D" 

moverx, movery, moverz, dx, dz, girar = 0, 1.5, 0, 0, 0, 0 

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

# -------------------------------------------------------------------------
# FUNCIONES PRIMITIVAS DE DIBUJO
# -------------------------------------------------------------------------
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
    """Buzón rojo con poste y banderita."""
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.07, 1.0, (0.85, 0.85, 0.85)); glPopMatrix()
    glTranslate(0, 0.9, 0); draw_cube(0.5, 0.55, 0.45, (0.9, 0.15, 0.15), (0.9, 0.15, 0.15), (0.9, 0.15, 0.15))
    glTranslate(0, 0.5, 0); draw_piramide(0.15, 0.55, 0.45, 0, (0.75, 0.1, 0.1))
    glTranslate(0.3, -0.3, 0); draw_cube(0.3, 0.05, 0.2, (0.9, 0.8, 0.1), (0.9, 0.8, 0.1), (0.9, 0.8, 0.1))
    glPopMatrix()

def draw_banco_madera():
    """Banco de madera estilo parque."""
    glPushMatrix()
    color_m = (0.55, 0.33, 0.13)
    draw_cube(0.1, 1.4, 0.5, color_m, color_m, color_m)
    glTranslate(0, 0.35, -0.22); draw_cube(0.35, 1.4, 0.08, color_m, color_m, color_m)
    for px in [-0.6, 0.6]:
        for pz in [-0.18, 0.18]:
            glPushMatrix(); glTranslate(px, -0.3, pz); draw_cube(0.3, 0.08, 0.08, color_m, color_m, color_m); glPopMatrix()
    glPopMatrix()

def draw_maceta(color_flor):
    """Maceta de barro con flor adentro."""
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.22, 0.3, (0.7, 0.38, 0.22)); glPopMatrix()
    glTranslate(0, 0.3, 0); draw_flor(color_flor, (1.0, 0.9, 0.3))
    glPopMatrix()

def draw_tienda_nook():
    """Tienda de Tom Nook con techo y ventanas."""
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
    """Museo con columnas y cúpula."""
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
    """Muelle de madera hacia el agua."""
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
    """Piedra gris redondeada."""
    glPushMatrix(); glScale(1.0, 0.6, 0.85)
    draw_sphere(radio, (0.62, 0.62, 0.65))
    glPopMatrix()

def draw_mariposa(t, offset):
    """Mariposa animada que aletea."""
    aleteo = math.sin(t * 10 + offset) * 25
    glPushMatrix()
    glTranslate(0, 0.05 * math.sin(t * 3 + offset), 0)
    glRotate(aleteo, 0, 0, 1); draw_sphere(0.08, (0.9, 0.4, 0.9))
    glRotate(-2 * aleteo, 0, 0, 1); draw_sphere(0.08, (0.9, 0.4, 0.9))
    glPopMatrix()

def draw_bote_pesca():
    """Bote de madera pequeño."""
    glPushMatrix()
    color_b = (0.55, 0.28, 0.1)
    draw_cube(0.45, 1.8, 0.9, color_b, (0.6, 0.32, 0.12), color_b)
    glTranslate(0, 0.45, 0); draw_cube(0.15, 1.6, 0.75, (0.88, 0.82, 0.72), (0.88, 0.82, 0.72), (0.88, 0.82, 0.72))
    glTranslate(0, 0.15, -0.2); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.04, 1.2, (0.45, 0.25, 0.08))
    glTranslate(0.2, 0, 0.8); draw_cube(0.3, 0.35, 0.02, (0.9, 0.2, 0.2), (0.9, 0.2, 0.2), (0.9, 0.2, 0.2))
    glPopMatrix()

def draw_campana_entrada():
    """Arco con campana, entrada al pueblo."""
    glPushMatrix()
    for px in [-0.8, 0.8]:
        glPushMatrix(); glTranslate(px, 0, 0); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.1, 2.5, (0.5, 0.3, 0.1)); glPopMatrix()
    glTranslate(0, 2.4, 0); draw_cube(0.15, 1.6, 0.15, (0.5, 0.3, 0.1), (0.5, 0.3, 0.1), (0.5, 0.3, 0.1))
    glTranslate(0, -0.4, 0); draw_sphere(0.3, (0.88, 0.75, 0.2))
    glTranslate(0, -0.25, 0); draw_sphere(0.07, (0.6, 0.5, 0.1))
    glPopMatrix()

def draw_arbol_cerezo():
    """Árbol de sakura rosado."""
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.18, 1.8, (0.45, 0.28, 0.18)); glRotate(-270, 1, 0, 0)
    glTranslate(0, 1.8, 0)
    draw_sphere(0.9, (0.98, 0.72, 0.78))
    glTranslate(0.6, 0.3, 0.3); draw_sphere(0.65, (0.95, 0.65, 0.72))
    glTranslate(-1.2, 0.1, -0.5); draw_sphere(0.7, (1.0, 0.78, 0.82))
    glPopMatrix()

def draw_letrero_pueblo():
    """Letrero de bienvenida al pueblo."""
    glPushMatrix()
    for px in [-0.7, 0.7]:
        glPushMatrix(); glTranslate(px, 0, 0); glRotate(270, 1, 0, 0)
        draw_cilindro_tapado(0, 360, 0.08, 1.6, (0.45, 0.28, 0.1)); glPopMatrix()
    glTranslate(0, 1.3, 0); draw_cube(0.45, 1.4, 0.1, (0.85, 0.62, 0.28), (0.9, 0.68, 0.32), (0.85, 0.62, 0.28))
    for sx in [-0.5, 0.5]:
        glPushMatrix(); glTranslate(sx, 0.1, 0.06); draw_sphere(0.12, (0.95, 0.3, 0.4)); glPopMatrix()
    glPopMatrix()

def draw_jardin_flores():
    """Parche de flores variadas."""
    colores = [(0.95, 0.2, 0.3), (0.95, 0.7, 0.1), (0.5, 0.2, 0.9), (0.2, 0.8, 0.4)]
    posiciones = [(-0.5,0,0), (0.5,0,0.3), (0,0,-0.5), (-0.3,0,0.6), (0.6,0,-0.3), (-0.6,0,0.5)]
    for i, (fx, fy, fz) in enumerate(posiciones):
        glPushMatrix(); glTranslate(fx, fy, fz)
        draw_flor(colores[i % len(colores)], (1.0, 0.9, 0.3))
        glPopMatrix()

def draw_pozo():
    """Pozo de piedra con tejado."""
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

# -------------------------------------------------------------------------
# ENSAMBLAJE DE LA CIUDAD
# -------------------------------------------------------------------------
def draw_mundo(): 
    t = glfw.get_time()
    
    # 1. PISO BASE (RIO)
    glPushMatrix(); glTranslatef(0, -0.05, 0); draw_rectangulo(50, 50, (0.1, 0.5, 0.9)); glPopMatrix() 

    # 2. LAS 5 ISLAS BASE
    glPushMatrix(); glTranslate(0, 0, 0); draw_rectangulo(8, 8, (0.5, 0.85, 0.4)); glPopMatrix()       
    glPushMatrix(); glTranslate(0, 0, -20); draw_rectangulo(14, 10, (0.4, 0.8, 0.3)); glPopMatrix()    
    glPushMatrix(); glTranslate(0, 0, 20); draw_rectangulo(14, 10, (0.95, 0.85, 0.6)); glPopMatrix()   
    glPushMatrix(); glTranslate(22, 0, 0); draw_rectangulo(12, 14, (0.45, 0.8, 0.35)); glPopMatrix()   
    glPushMatrix(); glTranslate(-22, 0, 0); draw_rectangulo(12, 14, (0.45, 0.8, 0.35)); glPopMatrix()  

    # 3. PUENTES DE CONEXIÓN
    glPushMatrix(); glTranslate(0, 0, -10); glRotate(90, 0, 1, 0); draw_puente_ac(); glPopMatrix() 
    glPushMatrix(); glTranslate(0, 0, 10); glRotate(90, 0, 1, 0); draw_puente_ac(); glPopMatrix()  
    glPushMatrix(); glTranslate(10, 0, 0); draw_puente_ac(); glPopMatrix()                         
    glPushMatrix(); glTranslate(-10, 0, 0); draw_puente_ac(); glPopMatrix()                        

    # =====================================================================
    # ZONA OESTE (MERCADO, ARBUSTOS, CERCAS)
    # =====================================================================
    glPushMatrix()
    glTranslate(-22, 0, 0)
    glPushMatrix(); glTranslate(0, 0, -3); draw_puesto_mercado((0.2, 0.8, 0.3)); glPopMatrix() 
    glPushMatrix(); glTranslate(3, 0, 0); glRotate(-90, 0, 1, 0); draw_puesto_mercado((0.9, 0.2, 0.2)); glPopMatrix() 
    glPushMatrix(); glTranslate(-3, 0, 0); glRotate(90, 0, 1, 0); draw_puesto_mercado((0.2, 0.4, 0.9)); glPopMatrix() 
    
    pos_arbustos_mercado = [(-4, -4), (-2, -4), (2, -4), (4, -4), (-4, 4), (-2, 4), (2, 4), (4, 4), (4, -2), (-4, -2)]
    for ax, az in pos_arbustos_mercado:
        glPushMatrix(); glTranslate(ax, 0, az); draw_arbusto(); glPopMatrix()

    glPushMatrix(); glTranslate(-5, 0, -10); draw_casa_colorida((0.8, 0.8, 0.9), (0.4, 0.4, 0.5)); glPopMatrix()
    glPushMatrix(); glTranslate(-5, 0, -8.5); draw_cerca(); glPopMatrix()
    glPopMatrix()

    # =====================================================================
    # ZONA ESTE (CAMPAMENTO Y HUERTO)
    # =====================================================================
    glPushMatrix()
    glTranslate(22, 0, 0)
    
    draw_fogata(t)
    glPushMatrix(); glTranslate(0, 0, -3); draw_tienda_campana((0.9, 0.9, 0.2)); glPopMatrix() 
    glPushMatrix(); glTranslate(3, 0, 2); glRotate(-45, 0, 1, 0); draw_tienda_campana((0.2, 0.6, 0.9)); glPopMatrix() 
    glPushMatrix(); glTranslate(-1.5, 0, 0); draw_tronco_asiento(); glPopMatrix()
    glPushMatrix(); glTranslate(1.5, 0, 0); draw_tronco_asiento(); glPopMatrix()
    
    glPushMatrix(); glTranslate(-4, 0, 8); draw_huerto(); glPopMatrix()
    glPushMatrix(); glTranslate(4, 0, 8); draw_huerto(); glPopMatrix()
    
    for i in range(4):
        glPushMatrix(); glTranslate(i*3 - 4, 0, -10); draw_arbol_manzanas(); glPopMatrix()
    glPopMatrix()

    # =====================================================================
    # ZONA SUR (PLAYA)
    # =====================================================================
    glPushMatrix()
    glTranslate(0, 0, 22)
    glPushMatrix(); glTranslate(0, 0, 6); draw_faro(); glPopMatrix()
    
    glPushMatrix(); glTranslate(-5, 0, 2); draw_sombrilla(); glPopMatrix()
    glPushMatrix(); glTranslate(-5, 0, 4); draw_toalla_playa(); glPopMatrix()
    glPushMatrix(); glTranslate(5, 0, 2); draw_sombrilla(); glPopMatrix()
    glPushMatrix(); glTranslate(5, 0, 4); draw_toalla_playa(); glPopMatrix()

    for x in range(-10, 11, 4):
        glPushMatrix(); glTranslate(x, 0, -2); draw_palmera(); glPopMatrix()
    glPopMatrix()

    # =====================================================================
    # ZONA NORTE (EL AYUNTAMIENTO Y CASAS)
    # =====================================================================
    glPushMatrix()
    glTranslate(0, 0, -22)
    
    # Edificio del reloj central (En lugar de la cascada)
    glPushMatrix(); glTranslate(0, 0, -4); glScale(0.6, 0.6, 0.6); draw_edificio_reloj(t); glPopMatrix()
    
    # Casas Zen y Coloridas alrededor
    glPushMatrix(); glTranslate(-8, 0, 2); draw_casa_zen((0.9,0.9,0.9), (0.8,0.3,0.3)); glPopMatrix()
    glPushMatrix(); glTranslate(-8, 0, 3.5); draw_cerca(); glPopMatrix()
    glPushMatrix(); glTranslate(-8, 0, -5); draw_casa_colorida((0.9,0.9,0.2), (0.2,0.6,0.9)); glPopMatrix()
    
    glPushMatrix(); glTranslate(8, 0, 2); draw_casa_zen((0.9,0.9,0.9), (0.3,0.8,0.3)); glPopMatrix()
    glPushMatrix(); glTranslate(8, 0, 3.5); draw_cerca(); glPopMatrix()
    glPushMatrix(); glTranslate(8, 0, -5); draw_casa_colorida((0.9,0.2,0.9), (0.2,0.9,0.6)); glPopMatrix()
    glPopMatrix()

    # =====================================================================
    # PLAZA CENTRAL (FUENTE Y JAURÍA DE MASCOTAS)
    # =====================================================================
    glPushMatrix(); glTranslate(0, 0.01, 0); draw_rectangulo(4, 4, (0.7, 0.7, 0.7)); glPopMatrix() 
    glPushMatrix(); glTranslate(0, 0, 0); draw_fuente_ac(t); glPopMatrix()
    
    # Los 4 animalitos rodeando la plaza a diferentes velocidades y distancias
    mover_en_circulo(lambda: draw_perro_caminando((0.8, 0.4, 0.2), t), 5.5, t*0.6, 0, 90) # Perro café
    mover_en_circulo(lambda: draw_perro_caminando((0.2, 0.2, 0.2), t), 6.5, t*0.4, 3.14, -90) # Perro negro
    mover_en_circulo(lambda: draw_gato_caminando((0.9, 0.9, 0.9), t), 7.5, t*0.5, 1.5, 90) # Gatito blanco (encimoso)
    mover_en_circulo(lambda: draw_conejo_caminando((0.9, 0.6, 0.6), t), 4.5, t*0.7, 4.5, -90) # Conejo rosado

    # =====================================================================
    # CIELO LOKO (Sol, Lluvia de Globos)
    # =====================================================================
    glPushMatrix(); glRotate(t * 10, 0, 0, 1); glTranslate(40, 0, -25); draw_sphere(5.0, (1.0, 0.9, 0.2)); glPopMatrix() 
    for i in range(15): 
        glPushMatrix(); glTranslatef(-40 + ((t * 2 + i * 10) % 80), 18, -30 + (i*5)); draw_nube(); glPopMatrix()
    
    colores_globo = [(0.9,0.2,0.2), (0.2,0.9,0.2), (0.2,0.2,0.9), (0.9,0.9,0.2)]
    for i in range(8): 
        x = -40 + ((t * 4 + i * 10) % 80)
        glPushMatrix(); glTranslatef(x, 0, -20 + i*8); draw_globo(t, colores_globo[i%4], i); draw_regalo(t, i); glPopMatrix()

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
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks: 
                hand_landmarks = detection_result.hand_landmarks[0]
                
                xml, yml = int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h)
                xil, yil = int(hand_landmarks[5].x * w), int(hand_landmarks[5].y * h)
                xbl, ybl = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame2, (cx, cy), 5, (0, 255, 0), -1)
                
                mov_z = yil - ybl 
                giro = xil - xbl  
                umbral = 40 
                
                if mov_z > umbral:   moverx += dx * 0.6; moverz += dz * 0.6
                elif mov_z < -umbral: moverx -= dx * 0.6; moverz -= dz * 0.6
                if giro > umbral:     girar += 0.08
                elif giro < -umbral:  girar -= 0.08
                     
            cv2.imshow("mundo3D", frame2) 
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
            gluPerspective(60, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 200.0)  
            glMatrixMode(GL_MODELVIEW); glLoadIdentity() 
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