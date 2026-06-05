

import time, math, os
import numpy as np
import cv2

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
W, H     = 800, 600
FPS      = 30
DURATION = 30.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH  = os.path.join(SCRIPT_DIR, "HIROMISAKE.ttf")

CX, CY  = W // 2, int(H * 0.50)
CORE_R  = 75

def hex2bgr(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

PALETTES = {
    "intro": [hex2bgr(c) for c in ['FF7B00', 'FF8800', 'FF9500', 'FFA200', 'FFAA00', 'FFB700', 'FFC300', 'FFD000', 'FFDD00', 'FFEA00']],
    "agua":  [hex2bgr(c) for c in ['03045E', '0077B6', '00B4D8', '90E0EF', 'CAF0F8']],
    "fuego": [hex2bgr(c) for c in ['03071E', '370617', '6A040F', '9D0208', 'D00000', 'DC2F02', 'E85D04', 'F48C06', 'FAA307', 'FFBA08']],
    "aire":  [hex2bgr(c) for c in ['003049', 'D62828', 'F77F00', 'FCBF49', 'EAE2B7']],
    "tierra":[hex2bgr(c) for c in ['132A13', '31572C', '4F772D', '90A955', 'ECF39E']]
}

# ══════════════════════════════════════════════════════════════════
# UTILIDADES Y BLINDAJE DE TIPOS DE OPENCV
# ══════════════════════════════════════════════════════════════════
def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))
def smooth(a, b, x):
    x = clamp((x - a) / (b - a) if b != a else 0.0)
    return x * x * (3 - 2 * x)
def col_alpha(color, a): return tuple(int(c * clamp(a)) for c in color)
def pulse(t, freq=1.5, lo=0.5, hi=1.0): return lo + (hi - lo) * (0.5 + 0.5 * math.sin(t * freq * math.pi * 2))

def pt(x, y):
    """Fuerza a enteros para que OpenCV no marque error de axes/types"""
    return (int(float(x)), int(float(y)))

def get_3d_orbit_target(t, angle_deg, radius, speed):
    t_anim = t * speed
    x_local = radius * math.cos(t_anim)
    y_local = (radius * 0.25) * math.sin(t_anim) 
    rad = math.radians(angle_deg)
    tx = CX + x_local * math.cos(rad) - y_local * math.sin(rad)
    ty = CY + x_local * math.sin(rad) + y_local * math.cos(rad)
    return tx, ty

def draw_text_centered(img, text, cx, cy, size, color, alpha=1.0, glow=True):
    if alpha <= 0: return
    try:
        from PIL import ImageFont, ImageDraw, Image, ImageFilter
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = font.getbbox(text)
        tx, ty = cx - (bbox[2]-bbox[0])//2 - bbox[0], cy - (bbox[3]-bbox[1])//2 - bbox[1]
        rgb = (color[2], color[1], color[0])
        
        if glow:
            glow_layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow_layer)
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]: gd.text((tx+dx, ty+dy), text, font=font, fill=(*rgb, int(150 * alpha)))
            pil = Image.alpha_composite(pil.convert("RGBA"), glow_layer.filter(ImageFilter.GaussianBlur(radius=5))).convert("RGB")
        
        draw = ImageDraw.Draw(pil)
        draw.text((tx+3, ty+3), text, font=font, fill=(0, 0, 0, int(255 * alpha)))
        draw.text((tx, ty), text, font=font, fill=(*rgb, int(255 * alpha)))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        scale = size / 40.0
        cv2.putText(img, text, pt(cx-50, cy), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(img, text, pt(cx-50, cy), cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════
# CINEMÁTICA INVERSA
# ══════════════════════════════════════════════════════════════════
class ProceduralChain:
    def __init__(self, origin, num_joints, link_length):
        self.joints = [np.array(origin, dtype=np.float32) for _ in range(num_joints)]
        self.angles = [0.0] * num_joints
        self.link_length = link_length
        self.num_joints = num_joints

    def resolve(self, target_pos):
        self.joints[0] = np.array(target_pos, dtype=np.float32)
        for i in range(1, self.num_joints):
            prev, curr = self.joints[i - 1], self.joints[i]
            dx, dy = prev[0] - curr[0], prev[1] - curr[1]
            angle = math.atan2(dy, dx)
            curr[0] = prev[0] - math.cos(angle) * self.link_length
            curr[1] = prev[1] - math.sin(angle) * self.link_length
            self.angles[i] = angle
        if self.num_joints > 1:
            dx, dy = self.joints[0][0] - self.joints[1][0], self.joints[0][1] - self.joints[1][1]
            self.angles[0] = math.atan2(dy, dx)

CHAINS = {
    "agua": ProceduralChain((CX, CY), 18, 12),
    "fuego": ProceduralChain((CX, CY), 16, 15),
    "tierra": ProceduralChain((CX, CY), 12, 18),
    "aire": ProceduralChain((CX, CY), 20, 10)
}

def draw_fluid_element(img, chain, widths, fill_color, stroke_color, alpha=1.0):
    if alpha <= 0: return
    left_side, right_side = [], []
    for i in range(chain.num_joints):
        pos, angle = chain.joints[i], chain.angles[i]
        w = (widths[i] if i < len(widths) else widths[-1]) * alpha
        left_side.append([pos[0] + math.cos(angle - math.pi/2) * w, pos[1] + math.sin(angle - math.pi/2) * w])
        right_side.append([pos[0] + math.cos(angle + math.pi/2) * w, pos[1] + math.sin(angle + math.pi/2) * w])
    right_side.reverse()
    pts = np.array(left_side + right_side, np.int32)
    cv2.fillPoly(img, [pts], col_alpha(fill_color, alpha), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, col_alpha(stroke_color, alpha), 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════
# CURVAS PARAMÉTRICAS (Para rúbrica, muy sutiles en fondo)
# ══════════════════════════════════════════════════════════════════
def get_lissajous(size): 
    ts = np.linspace(0, 2*math.pi, 200)
    return np.column_stack((CX + size * np.sin(3 * ts), CY + size * np.sin(2 * ts)))

def get_lemniscate(size): 
    ts = np.linspace(0, 2*math.pi, 200)
    x = size * np.cos(ts) / (1 + np.sin(ts)**2)
    y = size * np.sin(ts) * np.cos(ts) / (1 + np.sin(ts)**2)
    return np.column_stack((CX + x, CY + y))

def get_hypotrochoid(size): 
    R, r, d = 5.0, 3.0, 5.0
    ts = np.linspace(0, 10*math.pi, 300)
    x = (R - r) * np.cos(ts) + d * np.cos((R - r)/r * ts)
    y = (R - r) * np.sin(ts) - d * np.sin((R - r)/r * ts)
    return np.column_stack((CX + x * (size/8), CY + y * (size/8)))

def get_polar_rose(size): 
    k = 4
    ts = np.linspace(0, 2*math.pi, 300)
    r = size * np.cos(k * ts)
    return np.column_stack((CX + r * np.cos(ts), CY + r * np.sin(ts)))

def draw_magic_seal_warp(img, t, curve_pts, color, rot_speed, scale_speed, alpha=1.0):
    if alpha <= 0: return
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.polylines(mask, [np.int32(curve_pts)], True, color, 2, cv2.LINE_AA)
    scale = 1.0 + 0.15 * math.sin(t * scale_speed)
    angle = t * rot_speed * 180 / math.pi
    M = cv2.getRotationMatrix2D(pt(CX, CY), angle, scale)
    warped = cv2.warpAffine(mask, M, (W, H))
    cv2.addWeighted(img, 1.0, warped, alpha, 0, img)

# ══════════════════════════════════════════════════════════════════
# FONDOS, TRANSICIONES DE CIELO, MONTAÑAS PROCEDURALES Y APPA
# ══════════════════════════════════════════════════════════════════
def draw_bg_palette(img, palette_name):
    pal = PALETTES[palette_name]
    c_top, c_bottom = pal[0], pal[-2]
    gradient = np.zeros((2, 1, 3), dtype=np.uint8)
    gradient[0, 0], gradient[1, 0] = c_top, c_bottom
    img[:] = cv2.resize(gradient, (W, H), interpolation=cv2.INTER_LINEAR)
    return c_top 

def draw_bg_transition(img, t_local, duration):
    """Hace la transición fluida del cielo de la paleta 'aire' a 'intro'"""
    alpha = clamp(t_local / duration)
    c_top_aire, c_bot_aire = PALETTES["aire"][0], PALETTES["aire"][-2]
    c_top_intro, c_bot_intro = PALETTES["intro"][0], PALETTES["intro"][-2]

    c_top = tuple(int(a*(1-alpha) + b*alpha) for a, b in zip(c_top_aire, c_top_intro))
    c_bot = tuple(int(a*(1-alpha) + b*alpha) for a, b in zip(c_bot_aire, c_bot_intro))

    gradient = np.zeros((2, 1, 3), dtype=np.uint8)
    gradient[0, 0], gradient[1, 0] = c_top, c_bot
    img[:] = cv2.resize(gradient, (W, H), interpolation=cv2.INTER_LINEAR)

def draw_mountains(img, t, alpha=1.0, progress=1.0, color_scheme="tierra"):
    """Montañas procedurales que crecen hasta 1/3 de la pantalla"""
    if alpha <= 0 or progress <= 0: return
    ov = img.copy()
    base_y = H
    max_h = H // 3  
    
    for layer in range(2):
        pts = [[W, base_y], [0, base_y]]
        offset = layer * 100
        
        # Color según la escena (Tierra o Final oscuro)
        if color_scheme == "tierra":
            color = PALETTES["tierra"][2] if layer == 0 else PALETTES["tierra"][0]
            outline = col_alpha((0,0,0), 0.3)
        else:
            color = (35, 25, 45) if layer == 0 else (20, 15, 25) # Siluetas oscuras
            outline = color
            
        speed = 0.5 if layer == 0 else 1.0
        
        for x in range(0, W + 20, 20):
            noise = math.sin(x * 0.01 + offset) * 30 + \
                    math.sin(x * 0.03 + t * 0.5 * speed) * 15 + \
                    math.sin(x * 0.005) * 50
            
            layer_h = max_h if layer == 1 else max_h * 1.3
            y = base_y - (layer_h + noise - 40) * progress
            pts.append([int(x), int(y)])
        
        pts.append([W, base_y])
        pts_array = np.array(pts, np.int32)
        cv2.fillPoly(ov, [pts_array], color, cv2.LINE_AA)
        cv2.polylines(ov, [pts_array], False, outline, 2, cv2.LINE_AA)
        
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_appa(img, t, alpha=1.0):
    """Dibuja a Appa volando en la parte inferior"""
    if alpha <= 0: return
    ov = img.copy()
    
    x = int((t * 80) % (W + 400) - 200)
    y = int(H - 120 + math.sin(t * 3.0) * 15) 
    
    c_fur = (240, 245, 245)  
    c_skin = (50, 70, 90)    
    c_belly = (190, 200, 200)
    
    cv2.ellipse(ov, pt(x - 55, y + 2), pt(22, 8), 0, 0, 360, c_skin, -1, cv2.LINE_AA)
    for px in [x - 22, x, x + 22]:
        cv2.ellipse(ov, pt(px, y + 18), pt(7, 12), 0, 0, 360, c_skin, -1, cv2.LINE_AA)
    
    cv2.ellipse(ov, pt(x, y), pt(45, 22), 0, 0, 360, c_fur, -1, cv2.LINE_AA)
    cv2.ellipse(ov, pt(x, y + 10), pt(40, 12), 0, 0, 180, c_belly, -1, cv2.LINE_AA)
    
    cv2.circle(ov, pt(x + 50, y - 5), 16, c_fur, -1, cv2.LINE_AA)
    cv2.ellipse(ov, pt(x + 48, y - 18), pt(4, 10), -30, 0, 360, c_skin, -1, cv2.LINE_AA)
    cv2.ellipse(ov, pt(x + 58, y - 15), pt(4, 10), 30, 0, 360, c_skin, -1, cv2.LINE_AA)
    
    arrow_pts = np.array([[x+65, y-5], [x+50, y-12], [x+50, y+2]], np.int32)
    cv2.fillPoly(ov, [arrow_pts], c_skin, cv2.LINE_AA)
    cv2.circle(ov, pt(x + 60, y - 2), 2, (0, 0, 0), -1, cv2.LINE_AA)
    
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_stars(img, t, n=400):
    rng = np.random.default_rng(42)
    xs, ys = rng.integers(0, W, n), rng.integers(0, H, n)
    tw = 0.3 + 0.7 * np.sin(t * 1.8 + np.arange(n) * 1.1)
    for i in range(n):
        cv2.circle(img, pt(xs[i], ys[i]), 1, (int(220*tw[i]),)*3, -1, cv2.LINE_AA)

def draw_clouds(img, t, y_base, color, speed=0.35, alpha=0.45, seed=0):
    if alpha <= 0: return
    rng = np.random.default_rng(seed)
    n = 6
    xs0, ys0 = rng.integers(-100, W + 100, n), rng.integers(-20, 30, n)
    ws, hs = rng.integers(100, 250, n), rng.integers(30, 80, n)
    ov = img.copy()
    for i in range(n):
        bx, by = int((xs0[i] + t * speed * 40) % (W + 300) - 150), int(y_base + ys0[i])
        for dx, dy, rw, rh in [(0,0,ws[i],hs[i]), (-ws[i]//3,-hs[i]//3,ws[i]//2,hs[i]//2), (ws[i]//3,-hs[i]//4,ws[i]//2,hs[i]//2)]:
            ex, ey = max(0, bx+dx), max(0, by+dy)
            if ex < W and ey < H:
                cv2.ellipse(ov, pt(ex, ey), pt(max(1, rw/2.0), max(1, rh/2.0)), 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

# ══════════════════════════════════════════════════════════════════
# NÚCLEO MINIMALISTA Y ELEMENTOS
# ══════════════════════════════════════════════════════════════════
def draw_core(img, t, alpha=1.0):
    if alpha <= 0: return
    cx, cy = CX, CY
    p      = pulse(t, 0.8, 0.5, 1.0)
 
    # Capas de aura difusa concéntricas — efecto glow suave
    for r_aura, strength in [
        (int(CORE_R*2.8), 0.022),
        (int(CORE_R*2.2), 0.038),
        (int(CORE_R*1.7), 0.058),
        (int(CORE_R*1.3), 0.080),
        (int(CORE_R*1.0), 0.105),
        (int(CORE_R*0.7), 0.135),
    ]:
        ov = img.copy()
        cv2.circle(ov, pt(cx,cy), r_aura, (255,255,255), -1, cv2.LINE_AA)
        cv2.addWeighted(ov, strength*alpha, img, 1-strength*alpha, 0, img)

def draw_agua(img, t, intensity=1.0):
    if intensity <= 0: return
    draw_magic_seal_warp(img, t, get_lissajous(90), PALETTES["agua"][3], rot_speed=-0.5, scale_speed=2.0, alpha=intensity * 0.15)
    target_radius = smooth(0, 2, t) * (CORE_R * 2.2) 
    tx, ty = get_3d_orbit_target(t, -45, target_radius, speed=4.5)
    CHAINS["agua"].resolve((tx, ty))
    widths = [8, 12, 14, 15, 14, 12, 10, 8, 6, 5, 4, 3, 2, 1, 1]
    draw_fluid_element(img, CHAINS["agua"], widths, PALETTES["agua"][2], PALETTES["agua"][4], intensity)
    
    white_pts = []
    for i in range(CHAINS["agua"].num_joints):
        pos, angle = CHAINS["agua"].joints[i], CHAINS["agua"].angles[i]
        offset = 10 * math.sin(t*5 + i*0.5)
        white_pts.append([pos[0] + math.cos(angle - math.pi/2)*offset, pos[1] + math.sin(angle - math.pi/2)*offset])
    cv2.polylines(img, [np.array(white_pts, np.int32)], False, col_alpha((255,255,255), intensity), 2, cv2.LINE_AA)

    if intensity > 0.05: draw_text_centered(img, "AGUA", CX + 260, CY - 220, 40, PALETTES["agua"][3], alpha=intensity)

def draw_fuego(img, t, intensity=1.0):
    if intensity <= 0: return
    draw_magic_seal_warp(img, t, get_lemniscate(100), PALETTES["fuego"][6], rot_speed=0.6, scale_speed=2.5, alpha=intensity * 0.15)
    target_radius = smooth(0, 2, t) * (CORE_R * 2.2)
    tx, ty = get_3d_orbit_target(t, 45, target_radius, speed=5.0)
    CHAINS["fuego"].resolve((tx, ty))
    widths = [10, 15, 18, 14, 11, 8, 5, 3, 2, 1]
    draw_fluid_element(img, CHAINS["fuego"], widths, PALETTES["fuego"][5], PALETTES["fuego"][8], intensity)
    
    helix_pts = []
    for i in range(CHAINS["fuego"].num_joints):
        pos, angle = CHAINS["fuego"].joints[i], CHAINS["fuego"].angles[i]
        offset = 18 * math.sin(t * 12 - i * 0.8) 
        hx, hy = pos[0] + math.cos(angle - math.pi/2) * offset, pos[1] + math.sin(angle - math.pi/2) * offset
        helix_pts.append([hx, hy])
    cv2.polylines(img, [np.array(helix_pts, np.int32)], False, col_alpha(PALETTES["fuego"][4], intensity), 2, cv2.LINE_AA)

    if intensity > 0.05: draw_text_centered(img, "FUEGO", CX - 260, CY + 220, 40, PALETTES["fuego"][7], alpha=intensity)

def draw_tierra(img, t, intensity=1.0):
    if intensity <= 0: return
    draw_magic_seal_warp(img, t, get_hypotrochoid(110), PALETTES["tierra"][3], rot_speed=-0.3, scale_speed=1.5, alpha=intensity * 0.15)
    target_radius = smooth(0, 2, t) * (CORE_R * 2.2)
    tx, ty = get_3d_orbit_target(t, 90, target_radius, speed=4.0)
    CHAINS["tierra"].resolve((tx, ty))
    widths = [12, 18, 17, 14, 10, 7, 4, 2]
    draw_fluid_element(img, CHAINS["tierra"], widths, PALETTES["tierra"][1], PALETTES["tierra"][3], intensity)
    
    for i in range(1, CHAINS["tierra"].num_joints, 2):
        pos = CHAINS["tierra"].joints[i]
        offset_x = 15 * math.cos(t * 5 + i)
        offset_y = 15 * math.sin(t * 5 + i)
        cv2.circle(img, pt(pos[0] + offset_x, pos[1] + offset_y), 6, col_alpha(PALETTES["tierra"][2], intensity), 2, cv2.LINE_AA)
        cv2.circle(img, pt(pos[0] + offset_x, pos[1] + offset_y), 2, col_alpha(PALETTES["tierra"][4], intensity), -1, cv2.LINE_AA)

    if intensity > 0.05: draw_text_centered(img, "TIERRA", CX + 260, CY + 220, 40, PALETTES["tierra"][3], alpha=intensity)

def draw_aire(img, t, intensity=1.0):
    if intensity <= 0: return
    draw_magic_seal_warp(img, t, get_polar_rose(110), PALETTES["aire"][3], rot_speed=0.4, scale_speed=1.8, alpha=intensity * 0.15)
    target_radius = smooth(0, 2, t) * (CORE_R * 2.2)
    tx, ty = get_3d_orbit_target(t, 0, target_radius, speed=5.5)
    CHAINS["aire"].resolve((tx, ty))
    widths = [8, 12, 14, 15, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1]
    draw_fluid_element(img, CHAINS["aire"], widths, col_alpha(PALETTES["aire"][4], 0.7*intensity), col_alpha((255, 255, 255), 0.9*intensity), intensity)
    
    if intensity > 0.05: draw_text_centered(img, "AIRE", CX - 260, CY - 220, 40, PALETTES["aire"][2], alpha=intensity)

# ══════════════════════════════════════════════════════════════════
# ESCENAS DEL TIMELINE
# ══════════════════════════════════════════════════════════════════
def scene_intro(img, t):
    draw_bg_palette(img, "intro")
    draw_stars(img, t)
    draw_clouds(img, t, int(H*0.75), PALETTES["intro"][3], speed=0.8, alpha=0.6)
    draw_core(img, t, alpha=smooth(1.2, 3.8, t))
    ft, fs = smooth(1.5, 3.8, t), smooth(2.5, 4.0, t)
    if ft > 0: draw_text_centered(img, "AVATAR", CX, CY - 40, 60, PALETTES["intro"][8], alpha=ft)
    if fs > 0: draw_text_centered(img, "simulacion cuatro elementos", CX, CY + 30, 30, (255,255,255), alpha=fs)

def scene_agua(img, t):
    bg_top = draw_bg_palette(img, "agua")
    draw_clouds(img, t, int(H*0.4), PALETTES["agua"][1], speed=1.0, alpha=smooth(0.0, 1.5, t - 4.0))
    draw_clouds(img, t, int(H*0.7), PALETTES["agua"][0], speed=1.5, alpha=smooth(0.0, 1.5, t - 4.0))
    draw_core(img, t)
    draw_agua(img, t - 4.0, intensity=smooth(0.0, 2.0, t - 4.0))

def scene_fuego(img, t):
    draw_bg_palette(img, "fuego")
    draw_core(img, t)
    draw_agua(img, t - 4.0, intensity=1.0)
    draw_fuego(img, t - 9.0, intensity=smooth(0.0, 2.0, t - 9.0))

def scene_tierra(img, t):
    draw_bg_palette(img, "tierra")
    # Montañas de tierra subiendo
    progress = smooth(0.0, 2.0, t - 14.0)
    draw_mountains(img, t, alpha=1.0, progress=progress, color_scheme="tierra")
    
    draw_core(img, t)
    draw_agua(img, t - 4.0, intensity=1.0)
    draw_fuego(img, t - 9.0, intensity=1.0)
    draw_tierra(img, t - 14.0, intensity=smooth(0.0, 2.0, t - 14.0))

def scene_aire(img, t):
    draw_bg_palette(img, "aire")
    draw_clouds(img, t, int(H*0.3), PALETTES["aire"][3], speed=1.8, alpha=smooth(0.0, 1.5, t - 19.0))
    draw_clouds(img, t, int(H*0.8), PALETTES["aire"][2], speed=2.5, alpha=smooth(0.0, 1.5, t - 19.0))
    
    # Montañas de tierra ocultándose
    progress = 1.0 - smooth(0.0, 1.5, t - 19.0)
    draw_mountains(img, t, alpha=progress, progress=progress, color_scheme="tierra")
    
    draw_core(img, t)
    draw_agua(img, t - 4.0, intensity=1.0)
    draw_fuego(img, t - 9.0, intensity=1.0)
    draw_tierra(img, t - 14.0, intensity=1.0)
    draw_aire(img, t - 19.0, intensity=smooth(0.0, 2.0, t - 19.0))

def scene_climax(img, t):
    # Transición suave del cielo de Aire a Intro
    t_local = t - 24.0
    draw_bg_transition(img, t_local, duration=4.0)
    
    # Núcleo y elementos girando limpios
    draw_core(img, t)
    draw_agua(img, t - 4.0, intensity=1.0)
    draw_fuego(img, t - 9.0, intensity=1.0)
    draw_tierra(img, t - 14.0, intensity=1.0)
    draw_aire(img, t - 19.0, intensity=1.0)
    draw_text_centered(img, "SIMULACION CUATRO ELEMENTOS", CX, 60, 45, (255,255,255), alpha=1.0)

def scene_final(img, t):
    fade = smooth(0.1, 1.6, t - 28.0)
    draw_bg_palette(img, "intro")
    
    # Montañas finales (siluetas oscuras) y Appa
    draw_mountains(img, t, alpha=fade, progress=1.0, color_scheme="final")
    draw_appa(img, t, alpha=fade)
    
    draw_core(img, t, alpha=0.7)
    draw_agua(img, t - 4.0, intensity=0.8)
    draw_fuego(img, t - 9.0, intensity=0.8)
    draw_tierra(img, t - 14.0, intensity=0.8)
    draw_aire(img, t - 19.0, intensity=0.8)
    
    draw_text_centered(img, "GRAFICACION POR COMPUTADORA", CX, 50, 30, PALETTES["intro"][8], alpha=fade)
    draw_text_centered(img, "Demo Procedural", CX, 90, 22, (255,255,255), alpha=fade)

# ══════════════════════════════════════════════════════════════════
# POST-EFECTOS Y GESTIÓN DE TIMELINE
# ══════════════════════════════════════════════════════════════════
_vig_mask = None
def post_vignette(img, strength=0.62):
    global _vig_mask
    if _vig_mask is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = (xx - W*0.5) / (W*0.5), (yy - H*0.5) / (H*0.5)
        _vig_mask = np.clip(1.0 - strength*(nx*nx + ny*ny), 0, 1)[..., None]
    return (img.astype(np.float32) * _vig_mask).astype(np.uint8)

def post_bloom(img, strength=0.32): return cv2.addWeighted(img, 1.0, cv2.GaussianBlur(img, (0, 0), 15), strength, 0)
def post_grain(img, t, strength=3):
    noise = np.random.default_rng(int(t * FPS) % 9999).integers(-strength, strength, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

SCENES = [
    ( 0,  4,  scene_intro),
    ( 4,  9,  scene_agua),
    ( 9,  14, scene_fuego),
    (14,  19, scene_tierra),
    (19,  24, scene_aire),
    (24,  28, scene_climax),
    (28,  30, scene_final),
]
TRANSITION = 1.0

def timeline(t, bufA, bufB):
    sid = next((i for i, (s, e, _) in enumerate(SCENES) if s <= t < e), len(SCENES)-1)
    s_start, s_end, fn = SCENES[sid]
    fn(bufA, t)
    dur, t_in = s_end - s_start, t - s_start
    if sid < len(SCENES) - 1 and t_in >= dur - TRANSITION:
        SCENES[sid+1][2](bufB, t)
        a = smooth(dur - TRANSITION, dur, t_in)
        return cv2.addWeighted(bufA, 1 - a, bufB, a, 0)
    return bufA.copy()

def main():
    os.makedirs("renders", exist_ok=True)
    bufA, bufB = np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("renders/demo_avatar_final.mp4", fourcc, FPS, (W, H))

    total_frames = int(DURATION * FPS)
    print("Renderizando a 800x600 - Appa y Montañas Procedurales...")
    t0 = time.perf_counter()
    for i in range(total_frames):
        t = i / FPS
        frame = timeline(t, bufA, bufB)
        frame = post_bloom(frame, 0.30)
        frame = post_vignette(frame, 0.60)
        frame = post_grain(frame, t, 3)
        writer.write(frame)
        
        cv2.imshow("Los Cuatro Elementos", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        
        if i % FPS == 0:
            eta = (time.perf_counter() - t0) / (i+1) * (total_frames - i)
            print(f"  [{int(100*i/total_frames):3d}%]  t={t:.1f}s  ETA: {eta:.0f}s")
            
    writer.release()
    cv2.destroyAllWindows()
    print(" Video finalizado exitosamente.")

if __name__ == "__main__":
    main()