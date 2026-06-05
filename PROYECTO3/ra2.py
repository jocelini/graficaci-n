
from __future__ import annotations
import os, sys, ssl, math, time, urllib.request, random
import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# --- LIBRERÍA NUEVA PARA TUS FUENTES CUSTOM ---
from PIL import Image, ImageDraw, ImageFont

# ── Descarga del modelo MediaPipe ─────────────────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e: print(f"Error descarga: {e}")

_mp_base    = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
_mp_opts    = vision.HandLandmarkerOptions(base_options=_mp_base, num_hands=2)
hand_landmarker = vision.HandLandmarker.create_from_options(_mp_opts)

# ── ArUco ─────────────────────────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
MARKER_ID       = 0
MARKER_LENGTH_M = 0.10          
ZNEAR, ZFAR     = 0.01, 200.0

_aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
_aruco_params = cv2.aruco.DetectorParameters()
if hasattr(cv2.aruco, "ArucoDetector"): _aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)
else: _aruco_detector = None

def detect_aruco(gray):
    if _aruco_detector is not None: corners, ids, _ = _aruco_detector.detectMarkers(gray)
    else: corners, ids, _ = cv2.aruco.detectMarkers(gray, _aruco_dict, parameters=_aruco_params)
    if ids is None: return None
    matches = np.where(ids.flatten() == MARKER_ID)[0]
    return corners[int(matches[0])] if len(matches) > 0 else None

def default_K(w, h):
    f = float(max(w, h))
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)

def estimate_pose_aruco(corners, K, dist):
    s = MARKER_LENGTH_M / 2.0
    obj = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)
    ip  = np.asarray(corners[0] if corners.ndim==3 else corners, dtype=np.float32).reshape(-1,2)
    flags = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2,"SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, ip, K, dist, flags=flags)
    return (rvec, tvec) if ok else (None, None)

def projection_from_K(K, w, h, znear, zfar):
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0]=2*fx/w;  P[1,1]=2*fy/h
    P[0,2]=(w-2*cx)/w; P[1,2]=(2*cy-h)/h
    P[2,2]=-(zfar+znear)/(zfar-znear); P[2,3]=-1
    P[3,2]=-2*zfar*znear/(zfar-znear)
    return P

def modelview_from_pose(rvec, tvec):
    R,_ = cv2.Rodrigues(rvec)
    M   = np.eye(4, dtype=np.float64)
    M[:3,:3]=R; M[:3,3]=tvec.flatten()
    return (np.diag([1.,-1.,-1.,1.]) @ M).T.astype(np.float32)

def get_landmark_3d(lm, camera_matrix, frame_w, frame_h, rvec, tvec, idx):
    if rvec is None or tvec is None: return None
    px, py = lm[idx].x * frame_w, lm[idx].y * frame_h
    K_inv = np.linalg.inv(camera_matrix)
    ray_cam = K_inv @ np.array([px, py, 1.0])
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    n = R[:, 2] 
    denom = n @ ray_cam
    if abs(denom) < 1e-6: return None
    lam = (n @ t) / denom
    p_cam = lam * ray_cam
    return R.T @ (p_cam - t)

# ── Textura de fondo ───────────────────────────────────────────────────────
_tex_id = None; _tex_buf = None
def upload_bg(frame, w, h):
    global _tex_id, _tex_buf
    rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)
    if _tex_buf is None or _tex_buf.shape[:2]!=(h,w): _tex_buf = np.empty((h,w,3), dtype=np.uint8)
    np.copyto(_tex_buf, rgb)
    if _tex_id is None: _tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,_tex_buf)

def draw_bg_quad(w, h):
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,w,0,h,-1,1)
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, _tex_id); glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(0,0); glTexCoord2f(1,0); glVertex2f(w,0)
    glTexCoord2f(1,1); glVertex2f(w,h); glTexCoord2f(0,1); glVertex2f(0,h)
    glEnd()
    glDisable(GL_TEXTURE_2D); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST)

# -------------------------------------------------------------------------
# VARIABLES GLOBALES DE LA CIUDAD DESTRUCTIBLE
# -------------------------------------------------------------------------
entidades_mundo = []
baldosas_borradas = set()
objeto_en_mano = None

tipos_disponibles = ["perro", "gato", "casa", "arbol", "hongo", "auto", "fuente"]
random.seed(123)
for _ in range(60):
    entidades_mundo.append({
        "id": random.randint(1000, 99999),
        "tipo": random.choice(tipos_disponibles),
        "x": random.uniform(-25, 25),
        "z": random.uniform(-25, 25),
        "color": (random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0.3, 1))
    })

# -------------------------------------------------------------------------
# OPENGL GEOMETRÍA
# -------------------------------------------------------------------------
def init_glfw():
    if not glfw.init(): raise Exception("Error GLFW") 
    window = glfw.create_window(1024, 768, "Realidad aumentada", None, None) 
    if not window: glfw.terminate(); raise Exception("Error Ventana") 
    glfw.make_context_current(window); glfw.swap_interval(1)
    return window 

def setup_opengl(): 
    glClearColor(0.4, 0.75, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS) 
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) 
    glLightfv(GL_LIGHT0, GL_POSITION, (10, 20, 10, 0)) 
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1)) 

def normal_tri(v1, v2, v3): 
    ux, uy, uz = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
    vx, vy, vz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
    nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx 
    l = math.sqrt(nx*nx + ny*ny + nz*nz) 
    return (0,1,0) if l < 1e-8 else (nx/l, ny/l, nz/l) 

def draw_rectangulo(lado1, lado2, color):  
    glBegin(GL_QUADS); glColor3f(*color); glNormal3f(0, 1, 0)  
    glVertex3f(-lado1, 0, lado2); glVertex3f(lado1, 0, lado2) 
    glVertex3f(lado1, 0, -lado2); glVertex3f(-lado1, 0, -lado2) 
    glEnd() 

def draw_cube(alto, ancho, largo, c1, c2, c3): 
    ancho, largo = ancho/2, largo/2 
    glBegin(GL_QUADS) 
    glColor3f(*c1); glNormal3f(0,0,1); glVertex3f(-ancho, 0, largo); glVertex3f(ancho, 0, largo); glVertex3f(ancho, alto, largo); glVertex3f(-ancho, alto, largo) 
    glNormal3f(0,0,-1); glVertex3f(-ancho, 0, -largo); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, alto, -largo); glVertex3f(-ancho, alto, -largo) 
    glNormal3f(-1,0,0); glVertex3f(-ancho, 0, -largo); glVertex3f(-ancho, 0, largo); glVertex3f(-ancho, alto, largo); glVertex3f(-ancho, alto, -largo) 
    glNormal3f(1,0,0); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, 0, largo); glVertex3f(ancho, alto, largo); glVertex3f(ancho, alto, -largo) 
    glColor3f(*c2); glNormal3f(0,1,0); glVertex3f(-ancho, alto, -largo); glVertex3f(ancho, alto, -largo); glVertex3f(ancho, alto, largo); glVertex3f(-ancho, alto, largo) 
    glColor3f(*c3); glNormal3f(0,-1,0); glVertex3f(-ancho, 0, -largo); glVertex3f(ancho, 0, -largo); glVertex3f(ancho, 0, largo); glVertex3f(-ancho, 0, largo) 
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

# -------------------------------------------------------------------------
# FUNCIONES ESPECÍFICAS DE OBJETOS
# -------------------------------------------------------------------------
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

def draw_casa_zen(color_pared, color_techo):
    glPushMatrix()
    draw_cube(1.2, 2.2, 2.0, color_pared, color_pared, color_pared) 
    glTranslate(0, 1.2, 0); draw_cube(0.2, 2.6, 2.4, color_techo, color_techo, color_techo)
    glTranslate(0, 0.2, 0); draw_piramide(1.0, 2.2, 2.0, 0, color_techo)
    glPopMatrix()

def draw_arbol_cerezo():
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.18, 1.8, (0.45, 0.28, 0.18)); glRotate(-270, 1, 0, 0)
    glTranslate(0, 1.8, 0); draw_sphere(0.9, (0.98, 0.72, 0.78))
    glTranslate(0.6, 0.3, 0.3); draw_sphere(0.65, (0.95, 0.65, 0.72))
    glPopMatrix()

def draw_hongo(color_sombrero):
    glPushMatrix()
    glPushMatrix(); glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 0.1, 0.3, (0.9, 0.9, 0.8)); glPopMatrix()
    glTranslate(0, 0.3, 0); draw_sphere(0.25, color_sombrero)
    glPopMatrix()

def draw_auto(color): 
    glPushMatrix(); glTranslate(0,0.001,0) 
    glPushMatrix(); glTranslate(0,0.4,0); draw_cube(0.3,1.2,0.6,color,color,color); glPopMatrix() 
    glPushMatrix(); glTranslate(0,0.7,0); draw_cube(0.2,0.8,0.4,(0.9,0.9,1.0),(0.9,0.9,1.0),(0.9,0.9,1.0)); glPopMatrix() 
    for x, z in [(0.3, -0.2), (0.3, 0.2), (-0.3, -0.2), (-0.3, 0.2)]: 
        glPushMatrix(); glTranslate(x,0.2,z); draw_cilindro_tapado(0,360,0.2,0.1,(0.1,0.1,0.1)); glPopMatrix() 
    glPopMatrix() 

def draw_fuente_ac(t):
    glPushMatrix()
    glRotate(270, 1, 0, 0); draw_cilindro_tapado(0, 360, 2.5, 0.3, (0.7, 0.7, 0.7))
    glTranslate(0, 0, 0.3); draw_cilindro_tapado(0, 360, 2.2, 0.1, (0.1, 0.7, 0.9))
    glTranslate(0, 0, 0.1); draw_cilindro_tapado(0, 360, 0.6, 1.0, (0.8, 0.8, 0.8))
    glPopMatrix()
    glPushMatrix(); glTranslate(0, 2.0, 0); h = abs(math.sin(t*8))*0.4; draw_sphere(0.3 + h, (0.8, 0.9, 1.0)); glPopMatrix()

def draw_nube_grande():
    glPushMatrix()
    for dx,dy,dz in [(0,0,0), (1,0,0), (-1,0,0), (0.5,0.5,0.5), (-0.5,0.5,-0.5)]:
        glPushMatrix(); glTranslate(dx,dy,dz); draw_sphere(1,(1.0,1.0,1.0)); glPopMatrix()
    glPopMatrix()

def render_entidad(ent, t):
    tipo = ent["tipo"]
    c = ent["color"]
    if tipo == "perro": draw_perro_caminando(c, t)
    elif tipo == "gato": draw_gato_caminando(c, t)
    elif tipo == "casa": draw_casa_zen((0.9,0.9,0.9), c)
    elif tipo == "arbol": draw_arbol_cerezo()
    elif tipo == "hongo": draw_hongo(c)
    elif tipo == "auto": draw_auto(c)
    elif tipo == "fuente": draw_fuente_ac(t)

# -------------------------------------------------------------------------
# LÓGICA DE DIBUJO DE LA CIUDAD DESTRUCTIBLE
# -------------------------------------------------------------------------
def draw_mundo(): 
    t = glfw.get_time()

    # 1. DIBUJAR PISO VOXEL 
    for ix in range(-12, 13):
        for iz in range(-12, 13):
            if (ix, iz) not in baldosas_borradas:
                glPushMatrix()
                glTranslate(ix * 2, 0, iz * 2)
                draw_rectangulo(0.95, 0.95, (0.48, 0.82, 0.38)) 
                glPopMatrix()

    # 2. DIBUJAR ENTIDADES (CON FÍSICA DE GRAVEDAD Y REBOTE)
    for ent in entidades_mundo:
        # Si el objeto fue aventado, tiene coordenada "y"
        if "y" in ent and ent["y"] > 0:
            ent["vy"] -= 0.6  # La gravedad jala al objeto hacia abajo
            ent["y"] += ent["vy"] 
            
            # Choca contra el piso
            if ent["y"] <= 0: 
                ent["y"] = 0 
                if ent["vy"] < -2.0: # Si cayó muy rápido, ¡rebota!
                    ent["vy"] = abs(ent["vy"]) * 0.4
                    ent["y"] = 0.5 
                else:
                    ent["vy"] = 0

        glPushMatrix()
        # Usamos la altura 'y' que tenga el objeto
        glTranslate(ent["x"], ent.get("y", 0.0), ent["z"])
        render_entidad(ent, t)
        glPopMatrix()

    # 3. CIELO Y NUBES
    glPushMatrix()
    glTranslate(0, 15, -15) 
    glRotate(t*10, 0,1,0); draw_sphere(3, (1.0, 0.9, 0.2))
    glPopMatrix()

    for i in range(5):
        glPushMatrix()
        glTranslate(-15 + i*7, 10, -10 + i*4)
        draw_nube_grande()
        glPopMatrix()

# --- LÓGICA ANTI-CONFUSIÓN Y ESQUELETO ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),         # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),    # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17)                                 # Base de la palma
]

def es_apuntando(lm):
    """Devuelve True SOLO si el índice está arriba, los demás cerrados, y el pulgar lejos."""
    indice_up  = lm[8].y < lm[6].y
    medio_down = lm[12].y > lm[10].y
    anular_down= lm[16].y > lm[14].y
    menique_down=lm[20].y > lm[18].y
    
    # Esto asegura que no estés haciendo un pellizco al mismo tiempo
    dist_pulgar_indice = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
    
    return indice_up and medio_down and anular_down and menique_down and dist_pulgar_indice > 0.08
def es_palma_abierta(lm):
    """Verifica si los 4 dedos están extendidos para hacer el Swipe."""
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and 
            lm[16].y < lm[14].y and lm[20].y < lm[18].y)

# -------------------------------------------------------------------------
# BUCLE PRINCIPAL — REALIDAD AUMENTADA MODO DIOS (SÚPER ESTABLE)
# -------------------------------------------------------------------------
def main(): 
    global objeto_en_mano

    try: window = init_glfw() 
    except Exception as e: print(f"Error al inicializar GLFW: {e}"); return 

    setup_opengl()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.waitKey(2000) 

    # --- CARGAR FUENTES CUSTOM ---
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    
    ruta_watermelon = os.path.join(ruta_actual, "Watermelon Sugar.ttf")
    ruta_Mb = os.path.join(ruta_actual, "Mabook.ttf") # Cambiado a ruta_Mb como pediste

    # 1. Letra del sistema/HUD (Watermelon Sugar)
    try:
        font_hud = ImageFont.truetype(ruta_watermelon, 45)
    except IOError:
        print(f"\n[ERROR FATAL] No encontré: {ruta_watermelon}")
        font_hud = ImageFont.load_default()

    # 2. Letra de monitos (Mabook)
    try:
        font_comentarios = ImageFont.truetype(ruta_Mb, 40)
    except IOError:
        print(f"\n[ERROR FATAL] No encontré: {ruta_Mb}")
        font_comentarios = ImageFont.load_default()

    escala_ciudad = 0.005 
    rotacion_ciudad = 0.0     
    prev_mano_x = None        
    prev_mano_y = None        
    perspectiva_plana = True 
    tecla_p_presionada = False

    last_rvec, last_tvec = None, None
    tiempo_ultimo_qr = 0

    comentario_texto = ""
    comentario_pos = (0, 0)
    comentario_tiempo = 0

    # Variables de Suavizado Matemático (Anti-Temblor)
    suavizado_x, suavizado_y, suavizado_z = None, None, None

    while not glfw.window_should_close(window): 
        ret, frame = cap.read() 
        if not ret: break 
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        K = default_K(w, h)
        dist = np.zeros((5, 1))

        corners = detect_aruco(gray)
        if corners is not None:
            rv, tv = estimate_pose_aruco(corners, K, dist)
            if rv is not None:
                last_rvec, last_tvec = rv, tv
                tiempo_ultimo_qr = time.time()
        
        if time.time() - tiempo_ultimo_qr > 0.5:
            last_rvec, last_tvec = None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = hand_landmarker.detect(mp_image)

        estado_hud = "MODO EDITOR: Buscando QR..." if last_rvec is None else "QR listo y detectado edita tu ciudad con tus manos :)"
        color_hud = (255, 50, 50) if last_rvec is None else (50, 255, 50) 
        anchor_3d = None

        if detection_result.hand_landmarks: 
            for hand_landmarks in detection_result.hand_landmarks:
                # DIBUJAR LÍNEAS DEL ESQUELETO (Topología Visual)
                for conexion in HAND_CONNECTIONS:
                    p1 = hand_landmarks[conexion[0]]
                    p2 = hand_landmarks[conexion[1]]
                    cx1, cy1 = int(p1.x * w), int(p1.y * h)
                    cx2, cy2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2) # Líneas Verdes
                
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) # Puntos Rojos

            # --- LÓGICA DE 1 MANO (Zoom, Rotación, Destrucción y Agarre) ---
            lm1 = detection_result.hand_landmarks[0]
            pinch_dist = math.hypot(lm1[8].x - lm1[4].x, lm1[8].y - lm1[4].y)
            mano_x, mano_y = lm1[0].x, lm1[0].y # Usamos la muñeca para el swipe

            # A) SWIPE: Palma Abierta (Zoom y Rotación)
            if es_palma_abierta(lm1):
                if prev_mano_x is not None and prev_mano_y is not None:
                    # Swipe Horizontal = Rotar ciudad
                    delta_x = mano_x - prev_mano_x
                    rotacion_ciudad += delta_x * 150.0 
                    
                    # Swipe Vertical = Zoom
                    delta_y = mano_y - prev_mano_y
                    escala_ciudad -= delta_y * 0.05 
                    escala_ciudad = max(0.001, min(escala_ciudad, 0.03))
                    
                    estado_hud = "SWIPE: ROTANDO/ZOOM"
                prev_mano_x, prev_mano_y = mano_x, mano_y
            
            # B) ACCIONES: Láser y Pellizco
            else:
                prev_mano_x, prev_mano_y = None, None # Reseteamos el swipe
                
                if last_rvec is not None:
                    anchor_raw = get_landmark_3d(lm1, K, w, h, last_rvec, last_tvec, 8)
                    if anchor_raw is not None:
                        if suavizado_x is None:
                            suavizado_x, suavizado_y, suavizado_z = anchor_raw
                        else:
                            suavizado_x = suavizado_x * 0.8 + anchor_raw[0] * 0.2
                            suavizado_y = suavizado_y * 0.8 + anchor_raw[1] * 0.2
                            suavizado_z = suavizado_z * 0.8 + anchor_raw[2] * 0.2
                        
                        anchor_3d = [suavizado_x, suavizado_y, suavizado_z]
                        
                        # Matemáticas para que el láser coincida aunque la ciudad esté rotada
                        cx_b, cz_b = anchor_3d[0] / escala_ciudad, -anchor_3d[1] / escala_ciudad
                        ang = math.radians(-rotacion_ciudad)
                        cx_city = cx_b * math.cos(ang) - cz_b * math.sin(ang)
                        cz_city = cx_b * math.sin(ang) + cz_b * math.cos(ang)

                        if es_apuntando(lm1) and pinch_dist > 0.08:
                            bx, bz = int(round(cx_city / 2.0)), int(round(cz_city / 2.0))
                            if (bx, bz) not in baldosas_borradas:
                                baldosas_borradas.add((bx, bz)) 
                                estado_hud = "LASER: BORRANDO SUELO!"
                                comentario_texto = "Oh no, mi casa :("
                                comentario_pos = (int(lm1[8].x * w) + 30, int(lm1[8].y * h) - 40)
                                comentario_tiempo = time.time()

                        elif pinch_dist < 0.05:
                            if objeto_en_mano is None:
                                for ent in entidades_mundo:
                                    if math.hypot(ent["x"] - cx_city, ent["z"] - cz_city) < 4.0:
                                        objeto_en_mano = ent
                                        entidades_mundo.remove(ent)
                                        comentario_texto = "¡AHHHHHH!"
                                        comentario_pos = (int(lm1[8].x * w) + 30, int(lm1[8].y * h) - 40)
                                        comentario_tiempo = time.time()
                                        break
                            
                            if objeto_en_mano is not None:
                                estado_hud = f"AGARRANDO: {objeto_en_mano['tipo'].upper()}"
                                anchor_3d[2] += 0.06 
                        else: 
                            if objeto_en_mano is not None:
                                objeto_en_mano["x"], objeto_en_mano["z"] = cx_city, cz_city
                                
                                # ¡MAGIA! Le damos una altura inicial para que "caiga" al suelo
                                objeto_en_mano["y"] = 25.0 
                                objeto_en_mano["vy"] = -1.0 
                                
                                entidades_mundo.append(objeto_en_mano)
                                objeto_en_mano = None
                                
                                # Opcional: Le ponemos texto divertido de caída
                                estado_hud = "¡LO VAS A LANZAR!"
                                comentario_texto = "¡WIIIIII!"
                                comentario_pos = (int(lm1[8].x * w) + 30, int(lm1[8].y * h) - 40)
                                comentario_tiempo = time.time()
                                
                            suavizado_x, suavizado_y, suavizado_z = None, None, None

        # 3. DIBUJAR TEXTOS CON PILLOW
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Dibujar HUD (Watermelon Sugar)
        draw.text((22, 52), estado_hud, font=font_hud, fill=(0, 0, 0)) 
        draw.text((20, 50), estado_hud, font=font_hud, fill=color_hud)

        # Dibujar Comentario del Monito (Mabook)
        if time.time() - comentario_tiempo < 1.5 and comentario_texto != "":
            cx, cy = comentario_pos
            cx = max(10, min(cx, w - 250))
            cy = max(10, min(cy, h - 60))
            
            draw.text((cx+2, cy+2), comentario_texto, font=font_comentarios, fill=(0, 0, 0)) 
            draw.text((cx, cy), comentario_texto, font=font_comentarios, fill=(255, 255, 255)) 
            
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 4. SUBIR A OPENGL
        glViewport(0, 0, w, h)
        upload_bg(frame, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_bg_quad(w, h)

# 4. DIBUJAR LA CIUDAD 3D
        if last_rvec is not None:
            P = projection_from_K(K, w, h, ZNEAR, ZFAR)
            MV = modelview_from_pose(last_rvec, last_tvec)

            glMatrixMode(GL_PROJECTION); glLoadMatrixf(P)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glMultMatrixf(MV)

            glPushMatrix()
            glScalef(escala_ciudad, escala_ciudad, escala_ciudad) 
            if perspectiva_plana: glRotatef(90, 1, 0, 0) 
            glRotatef(rotacion_ciudad, 0, 1, 0) # <--- LÍNEA 1
            draw_mundo() 
            glPopMatrix()
            

            if objeto_en_mano is not None and anchor_3d is not None:
                glPushMatrix()
                glTranslatef(anchor_3d[0], anchor_3d[1], anchor_3d[2])
                glScalef(escala_ciudad * 2.5, escala_ciudad * 2.5, escala_ciudad * 2.5) 
                if perspectiva_plana: glRotatef(90, 1, 0, 0) 
                glRotatef(rotacion_ciudad, 0, 1, 0) # <--- LÍNEA 2
                render_entidad(objeto_en_mano, glfw.get_time())
                glPopMatrix()

        glfw.swap_buffers(window) 
        glfw.poll_events() 
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: break 
        
        if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
            if not tecla_p_presionada:
                perspectiva_plana = not perspectiva_plana  
                tecla_p_presionada = True
        else: tecla_p_presionada = False

    cap.release(); cv2.destroyAllWindows(); glfw.terminate() 

if __name__ == "__main__": 
    main()