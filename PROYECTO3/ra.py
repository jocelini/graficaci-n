#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   REALIDAD AUMENTADA — ArUco + OpenGL + MediaPipe Hands          ║
║   Proyecto Final Graficación — Enero-Junio 2026                  ║
╠══════════════════════════════════════════════════════════════════╣
║  El objeto 3D vive sobre el marcador ArUco.                      ║
║  Interacción 100% Gestual y Espacial                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Controles de Gestos (1 Mano):                                   ║
║    🤏 Pellizco (Índice y Pulgar) → Agarra la Esfera o el Cubo    ║
║    ✊ Medio Puño                 → Agarra el Helado de Fresa     ║
║    ✌️ Amor y Paz (Dedos en V)    → Teletransporte al marcador    ║
║    💨 Swipe Izq/Der (Rápido)     → Cambia de Objeto              ║
╠══════════════════════════════════════════════════════════════════╣
║  Controles de Gestos (2 Manos):                                  ║
║    🔍 Juntar/Separar Índices     → Zoom (Aumentar/Reducir Escala)║
╚══════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
import math
import time
import os
import urllib.request
import ssl
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import (
    GLU_FILL, GLU_SMOOTH,
    gluNewQuadric, gluQuadricDrawStyle, gluQuadricNormals,
    gluSphere, gluCylinder, gluDisk,
)

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ════════════════════════════════════════════════════════════════════════════
# FUENTE PERSONALIZADA — Super Beatpop
# ════════════════════════════════════════════════════════════════════════════
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).resolve().parent / "Super Beatpop.ttf"
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        if FONT_PATH.is_file():
            _font_cache[size] = ImageFont.truetype(str(FONT_PATH), size)
        else:
            print(f"⚠  No se encontró {FONT_PATH.name} — usando fuente por defecto")
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_text_beatpop(frame: np.ndarray, text: str, pos: tuple[int,int],
                     size: int = 24, color: tuple[int,int,int] = (255, 220, 80),
                     shadow: bool = True) -> None:
    font = get_font(size)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = pos
    if shadow:
        shadow_color = (30, 20, 10)
        draw.text((x + 2, y + 2), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)

def text_size_beatpop(text: str, size: int) -> tuple[int, int]:
    font = get_font(size)
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

# ════════════════════════════════════════════════════════════════════════════
# DESCARGA AUTOMÁTICA DEL MODELO DE IA
# ════════════════════════════════════════════════════════════════════════════
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Descargando el modelo de MediaPipe, espera unos segundos...")
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Descarga completada exitosamente.")
    except Exception as e:
        print(f"Error en la descarga automática: {e}")

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX     = 0
MARKER_LENGTH_M  = 0.10
ARUCO_DICT_ID    = cv2.aruco.DICT_4X4_50
MARKER_ID        = 0
MODEL_SCALE_INIT = 0.04
ZNEAR, ZFAR      = 0.01,100.0
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_NPZ  = SCRIPT_DIR / "camera ar.npz"

SPRING_K      = 8.0
DAMPING       = 0.70
BOUNCE_DECAY  = 0.6

# ════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL 
# ════════════════════════════════════════════════════════════════════════════
OBJECT_MODES  = ["sphere", "cube", "icecream"]
OBJECT_LABELS = ["Esferita", "Cubito", "nieve de fresa"]
object_idx    = 0
MODEL_SCALE   = MODEL_SCALE_INIT

obj_pos = np.array([0.0, 0.0, 0.06], dtype=np.float64)
obj_vel = np.array([0.0, 0.0, 0.0],  dtype=np.float64)

last_rvec = None
last_tvec = None
marker_lost_time = 0.0
prev_time = time.time()

# Variables para Gestos Temporales
last_swipe_time = 0.0
prev_wrist_x = None
prev_zoom_dist = None

# ════════════════════════════════════════════════════════════════════════════
# CÁMARA Y ARUCO
# ════════════════════════════════════════════════════════════════════════════
def load_calibration(w: int, h: int):
    if CALIB_NPZ.is_file():
        d = np.load(CALIB_NPZ)
        return d["camera matrix"], d["dist coeffs"]
    f = float(max(w, h))
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64), np.zeros((5, 1), dtype=np.float64)

def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params), dictionary
    return None, dictionary

def detect_marker(gray, detector, dictionary):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=cv2.aruco.DetectorParameters())
    if ids is None or len(ids) == 0: return None, None
    matches = np.where(ids.flatten() == MARKER_ID)[0]
    if len(matches) == 0: return None, None
    return corners[int(matches[0])], ids[int(matches[0])]

def estimate_pose(corners, camera_matrix, dist_coeffs):
    s = MARKER_LENGTH_M / 2.0
    obj_pts = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)
    ip = np.asarray(corners[0] if corners.ndim==3 else corners, dtype=np.float32).reshape(-1,2)
    flags = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, ip, camera_matrix, dist_coeffs, flags=flags)
    if not ok: return None, None
    return rvec, tvec

# ════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE — NUEVA LÓGICA (AMOR Y PAZ + AGARRES + SWIPE + ZOOM)
# ════════════════════════════════════════════════════════════════════════════
def init_hands():
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    # ¡AUMENTAMOS A 2 MANOS PARA PERMITIR EL ZOOM!
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)

def is_peace_sign(lm) -> bool:
    index_up  = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down

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

def draw_hand_skeleton(frame, hand_landmarks, w, h):
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    for p1, p2 in connections:
        cv2.line(frame, (int(hand_landmarks[p1].x*w), int(hand_landmarks[p1].y*h)), (int(hand_landmarks[p2].x*w), int(hand_landmarks[p2].y*h)), (100,255,100), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,100,100), -1)

# ════════════════════════════════════════════════════════════════════════════
# FÍSICA Y ANCLAJE MANUAL
# ════════════════════════════════════════════════════════════════════════════
def update_physics(dt, anchor_3d=None, teleport=False, marker_visible=True, current_mode="sphere"):
    global obj_pos, obj_vel
    HOME_POS = np.array([0.0, 0.0, MODEL_SCALE * 0.5])

    if teleport:
        obj_pos = HOME_POS.copy()
        obj_vel = np.zeros(3)
        return

    if anchor_3d is not None:
        target_pos = anchor_3d.copy()
        if current_mode == "icecream": target_pos[2] += MODEL_SCALE * 1.5  
        else: target_pos[2] += MODEL_SCALE * 0.5  
        obj_pos = obj_pos + (target_pos - obj_pos) * 0.4
        obj_vel = np.zeros(3)
        return

    displacement = HOME_POS - obj_pos
    spring_force = SPRING_K * displacement
    if not marker_visible: spring_force *= 2.0

    obj_vel += spring_force * dt
    obj_vel *= DAMPING
    obj_pos += obj_vel * dt

    min_z = MODEL_SCALE * 0.5
    if obj_pos[2] < min_z:
        obj_pos[2] = min_z
        obj_vel[2] = abs(obj_vel[2]) * BOUNCE_DECAY

# ════════════════════════════════════════════════════════════════════════════
# OPENGL — MATRICES Y GEOMETRÍA
# ════════════════════════════════════════════════════════════════════════════
def projection_from_k(K, w, h, znear, zfar):
    fx,fy, cx,cy = K[0,0],K[1,1], K[0,2],K[1,2]
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0], P[1,1], P[0,2], P[1,2] = 2.0*fx/w, 2.0*fy/h, (w-2.0*cx)/w, (2.0*cy-h)/h
    P[2,2], P[2,3], P[3,2] = -(zfar+znear)/(zfar-znear), -1.0, -2.0*zfar*znear/(zfar-znear)
    return P

def modelview_from_pose(rvec, tvec):
    R,_ = cv2.Rodrigues(rvec)
    M = np.eye(4, dtype=np.float64)
    M[:3,:3], M[:3,3] = R, tvec.flatten()
    return (np.diag([1.,-1.,-1.,1.]) @ M).T.astype(np.float32)

_quadric = None
def get_quadric():
    global _quadric
    if _quadric is None:
        _quadric = gluNewQuadric()
        gluQuadricDrawStyle(_quadric, GLU_FILL)
        gluQuadricNormals(_quadric, GLU_SMOOTH)
    return _quadric

def draw_sphere_obj(scale):
    glColor3f(0.35, 0.75, 1.0)
    gluSphere(get_quadric(), scale, 36, 18)

def draw_cube_obj(scale):
    s = scale * 0.7
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.3, 0.3); glNormal3f(0, 0, 1); glVertex3f(-s, -s, s); glVertex3f(s, -s, s); glVertex3f(s, s, s); glVertex3f(-s, s, s)
    glColor3f(0.3, 0.8, 0.3); glNormal3f(0, 0, -1); glVertex3f(-s, -s, -s); glVertex3f(-s, s, -s); glVertex3f(s, s, -s); glVertex3f(s, -s, -s)
    glColor3f(0.3, 0.3, 0.8); glNormal3f(0, 1, 0); glVertex3f(-s, s, -s); glVertex3f(-s, s, s); glVertex3f(s, s, s); glVertex3f(s, s, -s)
    glColor3f(0.8, 0.8, 0.3); glNormal3f(0, -1, 0); glVertex3f(-s, -s, -s); glVertex3f(s, -s, -s); glVertex3f(s, -s, s); glVertex3f(-s, -s, s)
    glColor3f(0.8, 0.3, 0.8); glNormal3f(1, 0, 0); glVertex3f(s, -s, -s); glVertex3f(s, s, -s); glVertex3f(s, s, s); glVertex3f(s, -s, s)
    glColor3f(0.3, 0.8, 0.8); glNormal3f(-1, 0, 0); glVertex3f(-s, -s, -s); glVertex3f(-s, -s, s); glVertex3f(-s, s, s); glVertex3f(-s, s, -s)
    glEnd()

def draw_icecream(scale):
    q = get_quadric()
    cone_h, cone_r, ball_r, ball_z = scale * 2.2, scale * 0.85, scale * 1.05, scale * 2.2 * 0.95
    glPushMatrix(); glColor3f(0.82, 0.60, 0.25); gluCylinder(q, 0.0, cone_r, cone_h, 12, 4)
    glTranslatef(0, 0, cone_h); glColor3f(0.75, 0.52, 0.18); gluDisk(q, 0, cone_r, 12, 2); glPopMatrix()
    glDisable(GL_LIGHTING); glColor3f(0.55, 0.35, 0.10); glLineWidth(1.2)
    for i in range(8):
        ang = math.radians(i * 45); x = math.cos(ang) * cone_r * 0.95; y = math.sin(ang) * cone_r * 0.95
        glBegin(GL_LINES); glVertex3f(0, 0, 0); glVertex3f(x, y, cone_h); glEnd()
    glEnable(GL_LIGHTING)
    glPushMatrix(); glTranslatef(0, 0, ball_z); glColor3f(0.95, 0.35, 0.50); gluSphere(q, ball_r, 28, 14)
    glColor3f(0.35, 0.18, 0.05)
    for cx_, cy_, cz_ in [(0.4, 0.3, 0.85), (-0.5, 0.2, 0.82), (0.1, -0.55, 0.83), (0.6, -0.1, 0.79), (-0.2, 0.6, 0.78), (0.3, 0.55, 0.77), (-0.45, -0.4, 0.80)]:
        glPushMatrix(); glTranslatef(cx_ * ball_r * 0.6, cy_ * ball_r * 0.6, cz_ * ball_r); glRotatef(np.random.uniform(0, 360), 0, 0, 1); gluCylinder(q, 0.003, 0.003, 0.012, 6, 1); glPopMatrix()
    glPopMatrix()

def draw_object(mode, scale, offset):
    glPushMatrix()
    glTranslatef(float(offset[0]),float(offset[1]),float(offset[2]))
    glRotatef(glfw.get_time() * 40, 0, 0, 1)
    
    if mode=="sphere": draw_sphere_obj(scale)
    elif mode=="cube": draw_cube_obj(scale)
    else:              draw_icecream(scale)
    glPopMatrix()

def setup_lighting():
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, (0.3,0.5,1.0,0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.0,1.0,0.95,1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.25,0.25,0.25,1.0))
    glEnable(GL_NORMALIZE)

# ════════════════════════════════════════════════════════════════════════════
# OPENGL — TEXTURA Y RENDER
# ════════════════════════════════════════════════════════════════════════════
_tex_id = None
def upload_frame_texture(frame_bgr, w, h):
    global _tex_id
    rgb = cv2.flip(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 0)
    if _tex_id is None: _tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,rgb)

def draw_background_quad(w, h):
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,w,0,h,-1,1)
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, _tex_id); glColor3f(1,1,1)
    glBegin(GL_QUADS); glTexCoord2f(0,0); glVertex2f(0,0); glTexCoord2f(1,0); glVertex2f(w,0); glTexCoord2f(1,1); glVertex2f(w,h); glTexCoord2f(0,1); glVertex2f(0,h); glEnd()
    glDisable(GL_TEXTURE_2D); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST)

def draw_hud_cv(frame, marker_visible, is_holding, is_peace, is_swiping, is_zooming, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy(); cv2.rectangle(overlay, (0,0), (w,72), (0,0,0), -1); cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    put_text_beatpop(frame, OBJECT_LABELS[object_idx], (12, 8), size=32, color=(255, 220, 60))
    
    fps_text = f"FPS {fps:.0f}"
    tw, _ = text_size_beatpop(fps_text, 20)
    put_text_beatpop(frame, fps_text, (w - tw - 12, 14), size=20, color=(160, 255, 160))

    if marker_visible:
        put_text_beatpop(frame, "QR DETECTADO", (12, 46), size=18, color=(80, 255, 80))
    else:
        put_text_beatpop(frame, "buscando QR rebote activo", (12, 46), size=18, color=(80, 180, 255))

    # Mensajes de gestos
    if is_peace:
        msg = "¡upsi, reiniciaste!"
        tw, th = text_size_beatpop(msg, 48)
        put_text_beatpop(frame, msg, ((w - tw)//2, h - th - 50), size=48, color=(0, 255, 200), shadow=True)
    elif is_zooming:
        msg = "¡ZOOM! "
        tw, th = text_size_beatpop(msg, 48)
        put_text_beatpop(frame, msg, ((w - tw)//2, h - th - 50), size=48, color=(100, 255, 255), shadow=True)
    elif is_swiping:
        msg = "¡deslizaste jiji! "
        tw, th = text_size_beatpop(msg, 48)
        put_text_beatpop(frame, msg, ((w - tw)//2, h - th - 50), size=48, color=(255, 255, 100), shadow=True)
    elif is_holding:
        msg = "agarrando"
        tw, th = text_size_beatpop(msg, 48)
        put_text_beatpop(frame, msg, ((w - tw)//2, h - th - 50), size=48, color=(255, 100, 200), shadow=True)

    controls = ["T/deslizar: cambiar", "+/-/Zoom: escala", "ESC: salir"]
    for i, txt in enumerate(controls):
        put_text_beatpop(frame, txt, (12, h - len(controls)*22 - 8 + i*22), size=16, color=(200, 200, 200), shadow=False)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    global object_idx, MODEL_SCALE, last_rvec, last_tvec, marker_lost_time, prev_time
    global last_swipe_time, prev_wrist_x, prev_zoom_dist

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): sys.exit(1)
    ret, probe = cap.read()
    if not ret: sys.exit(1)

    cam_h, cam_w = probe.shape[:2]
    K, dist_coeffs = load_calibration(cam_w, cam_h)
    detector, dictionary = make_aruco_detector()
    landmarker = init_hands()

    if not glfw.init(): sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(cam_w, cam_h, "RA", None, None)
    if not window: glfw.terminate(); sys.exit(1)
    glfw.make_context_current(window); glfw.swap_interval(1)

    def on_key(win, key, _sc, action, _mods):
        global object_idx, MODEL_SCALE
        if action != glfw.PRESS: return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q): glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_T: object_idx = (object_idx + 1) % len(OBJECT_MODES)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):   MODEL_SCALE *= 1.1
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT): MODEL_SCALE /= 1.1

    glfw.set_key_callback(window, on_key); glEnable(GL_DEPTH_TEST)
    fps_counter, fps_display, fps_timer = 0, 0.0, time.time()
    get_font(32); get_font(20); get_font(18); get_font(48); get_font(16)

    while not glfw.window_should_close(window):
        now = time.time(); dt = min(now - prev_time, 0.05); prev_time = now

        fps_counter += 1
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer); fps_counter = 0; fps_timer = now

        ret, frame = cap.read()
        if not ret: continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, _ = detect_marker(gray, detector, dictionary)
        marker_visible = False
        rvec_use, tvec_use = last_rvec, last_tvec

        if corners is not None:
            rv, tv = estimate_pose(corners, K, dist_coeffs)
            if rv is not None:
                last_rvec, last_tvec, rvec_use, tvec_use = rv, tv, rv, tv
                marker_visible, marker_lost_time = True, now

        marker_stale = (now - marker_lost_time) > 30.0

        is_holding = False
        is_peace = False
        is_swiping = False
        is_zooming = False
        anchor_3d = None
        
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result    = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for lm in result.hand_landmarks:
                draw_hand_skeleton(frame, lm, w, h)
            
            # --- LÓGICA DE 1 MANO (Agarrar, Paz, Swipe) ---
            if len(result.hand_landmarks) == 1:
                lm = result.hand_landmarks[0]
                is_peace = is_peace_sign(lm)
                
                # Swipe (Cambiar Objeto)
                wrist_x = lm[0].x
                if prev_wrist_x is not None and not is_peace:
                    vx = (wrist_x - prev_wrist_x) / dt
                    # Si mueve la mano muy rápido horizontalmente
                    if abs(vx) > 2.0 and (now - last_swipe_time) > 1.0:
                        if vx > 0: object_idx = (object_idx + 1) % len(OBJECT_MODES)
                        else:      object_idx = (object_idx - 1) % len(OBJECT_MODES)
                        last_swipe_time = now
                        is_swiping = True
                prev_wrist_x = wrist_x
                prev_zoom_dist = None # Reseteamos el zoom
                
                # Agarre
                if rvec_use is not None and not is_peace and not is_swiping:
                    current_mode = OBJECT_MODES[object_idx]
                    if current_mode in ["sphere", "cube"]:
                        pinch_dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
                        if pinch_dist < 0.12:  
                            is_holding = True
                            p_index = get_landmark_3d(lm, K, w, h, rvec_use, tvec_use, 8)
                            p_thumb = get_landmark_3d(lm, K, w, h, rvec_use, tvec_use, 4)
                            if p_index is not None and p_thumb is not None:
                                anchor_3d = (p_index + p_thumb) / 2.0

                    elif current_mode == "icecream":
                        is_gripping = lm[12].y > lm[9].y
                        if is_gripping:
                            is_holding = True
                            anchor_3d = get_landmark_3d(lm, K, w, h, rvec_use, tvec_use, 5)

            # --- LÓGICA DE 2 MANOS (Zoom) ---
            elif len(result.hand_landmarks) >= 2:
                lm1 = result.hand_landmarks[0]
                lm2 = result.hand_landmarks[1]
                
                # Distancia entre los dedos índices de ambas manos
                dist = math.hypot(lm1[8].x - lm2[8].x, lm1[8].y - lm2[8].y)
                
                if prev_zoom_dist is not None:
                    delta = dist - prev_zoom_dist
                    # Filtro para que no tiemble con movimientos chiquitos
                    if abs(delta) > 0.01:
                        MODEL_SCALE += delta * 0.15  # Velocidad del zoom
                        MODEL_SCALE = max(0.01, min(MODEL_SCALE, 0.3)) # Límite de tamaño
                        is_zooming = True
                
                prev_zoom_dist = dist
                prev_wrist_x = None # Reseteamos el swipe

        update_physics(dt, anchor_3d=anchor_3d, teleport=is_peace, marker_visible=marker_visible, current_mode=OBJECT_MODES[object_idx])
        
        draw_hud_cv(frame, marker_visible, is_holding, is_peace, is_swiping, is_zooming, fps_display)

        glViewport(0,0,w,h)
        upload_frame_texture(frame,w,h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_background_quad(w,h)

        if rvec_use is not None and not marker_stale:
            P  = projection_from_k(K,w,h,ZNEAR,ZFAR)
            MV = modelview_from_pose(rvec_use,tvec_use)
            glMatrixMode(GL_PROJECTION); glLoadMatrixf(P)
            glMatrixMode(GL_MODELVIEW);  glLoadIdentity(); glMultMatrixf(MV)
            setup_lighting()
            draw_object(OBJECT_MODES[object_idx], MODEL_SCALE, obj_pos)

        glfw.swap_buffers(window)
        glfw.poll_events()

    if landmarker: landmarker.close()
    cap.release()
    glfw.terminate()
    print("adiosito")

if __name__ == "__main__":
    main()