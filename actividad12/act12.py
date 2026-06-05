import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import math

modo_camara = 1 

def draw_cube():
    glBegin(GL_QUADS); glColor3f(0.2, 0.8, 0.2)
    for v in [(0.5,1,0.5), (-0.5,1,0.5), (-0.5,0,0.5), (0.5,0,0.5),
              (0.5,1,-0.5), (-0.5,1,-0.5), (-0.5,0,-0.5), (0.5,0,-0.5)]: glVertex3f(*v)
    glEnd()

def key_callback(window, key, scancode, action, mods):
    global modo_camara
    if action == glfw.PRESS and glfw.KEY_1 <= key <= glfw.KEY_5: modo_camara = key - 48

def main():
    if not glfw.init(): return
    window = glfw.create_window(800, 600, "Camaras", None, None)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); gluPerspective(45, 800/600, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glLoadIdentity()

        t = glfw.get_time()
        obj_x, obj_y, obj_z = math.cos(t)*5, 0, math.sin(t)*5
        dir_x, dir_z = -math.sin(t), math.cos(t)

        if modo_camara == 1: gluLookAt(obj_x - dir_x*4, 3, obj_z - dir_z*4, obj_x, obj_y, obj_z, 0, 1, 0)
        elif modo_camara == 2: gluLookAt(0, 10, 15, obj_x, obj_y, obj_z, 0, 1, 0)
        elif modo_camara == 3: gluLookAt(obj_x, 15, obj_z+0.001, obj_x, obj_y, obj_z, 0, 1, 0)
        elif modo_camara == 4: gluLookAt(obj_x, 0.8, obj_z, obj_x+dir_x, 0.8, obj_z+dir_z, 0, 1, 0)
        elif modo_camara == 5: gluLookAt(obj_x+math.cos(t*0.5)*8, 4, obj_z+math.sin(t*0.5)*8, obj_x, obj_y, obj_z, 0, 1, 0)

        glPushMatrix(); glTranslatef(obj_x, obj_y, obj_z)
        glRotatef(math.degrees(math.atan2(dir_x, dir_z)), 0, 1, 0)
        draw_cube(); glPopMatrix()
        glfw.swap_buffers(window); glfw.poll_events()

if __name__ == "__main__": main()