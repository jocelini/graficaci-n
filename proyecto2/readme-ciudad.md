# 🏝️ Ciudad entorno 3D

> **Proyecto 3** — Simulación de una isla viva con ayuntamiento, mascotas y control por gestos de mano en tiempo real.

---

## 📋 Descripción

Aplicación de gráficas 3D en tiempo real que renderiza un mundo dividido en **cinco islas temáticas** interconectadas por puentes. El jugador puede explorar el entorno usando el **teclado** o mediante **gestos de la mano** capturados por la webcam gracias a **MediaPipe Hand Landmarker**.

---

## 🗂️ Tabla de Contenidos

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Controles](#-controles)
- [Arquitectura del Código](#-arquitectura-del-código)
- [Zonas del Mundo](#-zonas-del-mundo)
- [Referencia de Funciones](#-referencia-de-funciones)

---

## ✨ Características

- Mundo 3D procedural con **5 zonas temáticas** (Plaza Central, Norte, Sur, Este, Oeste)
- **Control por gestos de mano** en tiempo real con MediaPipe Tasks
- **Control por teclado** como alternativa
- Animaciones continuas: mascotas orbitando la plaza, reloj animado, fuente pulsante, globos y nubes flotantes
- Iluminación con OpenGL Legacy (GL_LIGHT0)
- Contador de FPS en tiempo real en la barra de título
- Descarga automática del modelo de IA si no está presente

---

## 📦 Requisitos

| Librería | Propósito |
|----------|-----------|
| `glfw` | Creación de ventana y contexto OpenGL |
| `PyOpenGL` | Renderizado 3D (OpenGL + GLU) |
| `opencv-python` | Captura de webcam y visualización de landmarks |
| `mediapipe` | Detección de mano y landmarks |
| `numpy` | Operaciones matriciales auxiliares |
| `keyboard` | Lectura de teclas en tiempo real |

### Versión de Python recomendada
Python **3.9 – 3.11**

---

## ⚙️ Instalación

```bash
# 1. Clonar o descargar el proyecto
# 2. Instalar dependencias
pip install glfw PyOpenGL PyOpenGL_accelerate opencv-python mediapipe numpy keyboard

# 3. Ejecutar
python cd2.py
```

> **Nota:** El modelo `hand_landmarker.task` se descarga automáticamente (~8 MB) la primera vez que se ejecuta el programa.

---

## 🚀 Uso

Al ejecutar `cd2.py` se abrirán **dos ventanas**:

1. **Ventana OpenGL** — El mundo 3D en perspectiva primera persona.
2. **Ventana OpenCV** — Vista de la webcam con los landmarks de la mano superpuestos.

Mueve la cámara usando teclado o gestos (ver sección de Controles). Presiona `ESC` o `Q` para salir.

---

## 🎮 Controles

### Teclado

| Tecla | Acción |
|-------|--------|
| `W` / `↑` | Avanzar |
| `S` / `↓` | Retroceder |
| `A` / `←` | Girar a la izquierda |
| `D` / `→` | Girar a la derecha |
| `ESC` | Salir |

### Gestos de Mano (Webcam)

La cámara detecta la posición relativa entre los landmarks del **índice** (punto 5 y 8):

| Gesto | Acción |
|-------|--------|
| Dedo índice apuntando **abajo** (punta por debajo de la articulación) | Avanzar |
| Dedo índice apuntando **arriba** | Retroceder |
| Dedo índice inclinado a la **derecha** | Girar a la derecha |
| Dedo índice inclinado a la **izquierda** | Girar a la izquierda |

> El umbral de detección de gestos es de **40 píxeles** de diferencia.

---

## 🏗️ Arquitectura del Código

```
cd2.py
├── Descarga automática del modelo (MODEL_PATH / MODEL_URL)
├── Inicialización (init_glfw, setup_opengl, setup_lights)
├── Primitivas de dibujo
│   ├── draw_rectangulo()
│   ├── draw_cube()
│   ├── draw_piramide()
│   ├── draw_sphere()
│   └── draw_cilindro_tapado()
├── Objetos compuestos
│   ├── Vegetación (draw_flor, draw_hongo, draw_arbusto, draw_palmera, draw_arbol_manzanas)
│   ├── Estructuras (draw_casa_colorida, draw_casa_zen, draw_edificio_reloj, draw_faro, draw_puente_ac)
│   ├── Mercado (draw_puesto_mercado, draw_comida_mercado)
│   ├── Campamento (draw_tienda_campana, draw_fogata, draw_tronco_asiento, draw_huerto)
│   ├── Playa (draw_sombrilla, draw_toalla_playa)
│   ├── Mascotas animadas (draw_perro_caminando, draw_gato_caminando, draw_conejo_caminando)
│   └── Ambiente (draw_nube, draw_globo, draw_regalo, draw_fuente_ac, draw_reloj_animado)
├── draw_mundo()  ← Ensambla todo el mundo
└── main()        ← Bucle principal (OpenGL + MediaPipe + teclado)
```

---

## 🗺️ Zonas del Mundo

### 🏛️ Plaza Central (origen)
Corazón del mundo. Contiene una **fuente animada** y cuatro **mascotas** que orbitan a diferentes velocidades:
- Perro café (radio 5.5)
- Perro negro (radio 6.5, sentido contrario)
- Gato blanco (radio 7.5)
- Conejo rosado (radio 4.5)

### 🏙️ Zona Norte — Ayuntamiento y Casas
Contiene el **edificio del reloj** con un reloj animado en tiempo real, casas zen y casas coloridas a ambos lados, con cercas decorativas.

### 🏖️ Zona Sur — Playa
Zona costera con un **faro**, sombrillas, toallas y una fila de palmeras.

### 🏕️ Zona Este — Campamento y Huerto
Área de acampado con **fogata animada**, tiendas de campaña de colores, troncos como asientos, huertos y árboles de manzanas.

### 🛒️ Zona Oeste — Mercado
Mercado con **puestos de comida** (frutas y verduras), arbustos decorativos y casas vecinas.

### ☁️ Cielo
Sol rotante, **15 nubes** que se desplazan horizontalmente y **8 globos** con regalos animados que flotan y oscilan.

---

## 📖 Referencia de Funciones

### Primitivas

| Función | Descripción |
|---------|-------------|
| `normal_tri(v1, v2, v3)` | Calcula la normal de un triángulo para iluminación correcta |
| `draw_rectangulo(lado1, lado2, color)` | Dibuja un plano horizontal (suelo o piso) |
| `draw_cube(alto, ancho, largo, color1, color2, color3)` | Cubo con cara superior e inferior de colores distintos |
| `draw_piramide(alto, ancho, largo, altura, color)` | Pirámide con base rectangular a una altura dada |
| `draw_sphere(radius, color)` | Esfera con sombreado suave (GLU) |
| `draw_cilindro_tapado(ang_in, ang_fin, radius, largo, color)` | Cilindro con tapas (puede ser arco parcial) |
| `mover_en_circulo(fn, radio, ang, base, voltear, altura)` | Posiciona y orienta un objeto en trayectoria circular |

### Sistema de Cámara

La cámara usa una proyección en perspectiva con FOV de 60° y utiliza `gluLookAt` con las variables globales:

```python
moverx, movery, moverz  # Posición del ojo
dx, dz                   # Vector de dirección (calculado desde girar)
girar                    # Ángulo de rotación horizontal (radianes)
```

---

## 🤖 Integración MediaPipe

Se usa la API de **MediaPipe Tasks** (no la API legacy):

```python
base_options = mp_tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
```

Los landmarks clave usados son:
- **Punto 0** — Muñeca (punto de referencia)
- **Punto 5** — Articulación base del índice
- **Punto 8** — Punta del índice

---

## 📝 Notas Técnicas

- El rendimiento puede variar según la GPU y la resolución de la webcam. Se recomienda una GPU con soporte OpenGL 2.x o superior.
- El modelo de MediaPipe se descarga en modo `ssl._create_unverified_context` para evitar errores de certificado en algunos sistemas. Considera usar un contexto verificado en producción.
- La librería `keyboard` puede requerir permisos de superusuario en Linux (`sudo python cd2.py`).