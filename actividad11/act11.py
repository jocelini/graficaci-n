import glfw
from OpenGL.GL import *

def main():
    if not glfw.init(): return
    window = glfw.create_window(500, 500, "Cuadrado GLFW", None, None)
    glfw.make_context_current(window)
    glOrtho(-1, 1, -1, 1, -1, 1) 
    
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0)
        glVertex2f(-0.5, 0.5); glVertex2f(0.5, 0.5)
        glVertex2f(0.5, -0.5); glVertex2f(-0.5, -0.5)
        glEnd()
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()

if __name__ == "__main__": main()