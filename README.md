# 📡 DeepSignal: Automatic Modulation Classification (AMC) for SDR

**DeepSignal** es un proyecto de investigación enfocado en el reconocimiento automático de esquemas de modulación digital y analógica (AMC). Utiliza Inteligencia Artificial y arquitecturas de Redes Neuronales Convolucionales (CNN) sobre secuencias temporal-espaciales In-Phase & Quadrature (I/Q) tomadas a través de tecnología Radio Definida por Software (SDR, HackRF One / RTL-SDR).

*Proyecto Académico - Trabajo de Tesis / Grado Universitario.*

## 🧠 Arquitectura de Procesamiento
1. **Captura:** Recepción de Radio Frecuencia con hardware SDR.
2. **DSP (Preprocesamiento):** Filtrado y acondicionamiento (offset tuning, normalización de energía) convirtiendo las lecturas RF en tensores matemáticos.
3. **Inferencia Profunda (Deep Learning):** Clasificación probabilística del tipo de modulación extraída usando convolución temporal 1D.

---

## 🛠 Estructura Analítica del Repositorio

- `data/` : Contenedor para bases de datos (Ej: RadioML 2016.10A) o capturas custom (*Nota: Los archivos .dat o .npy generados se ignorarán por Git*).
- `models/` : Arquitecturas entrenadas exportadas en formato `.keras` ó `.h5`.
- `notebooks/` : Entorno Jupyter enfocado a data discovery, gráficos de constelaciones I/Q de las señales y validación de datasets.
- `src/` : Núcleo empaquetado del orquestador.
  - `capture.py`: API base que consume librerías del hardware SDR.
  - `model.py`: La definición, optimización y entrenamiento estricto del modelo Deep Learning.
  - `processing.py`: Filtrado matemático usando módulos de `scipy.signal`.
  - `main.py`: Entrada principal al programa de consola o en vivo.

---

## 🚀 Instalación y Despliegue

Sigue estos pasos cuidadosamente para inicializar ecosistemas reproducibles localmente o en un equipo externo.

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/deepsignal.git
cd deepsignal
```

### 2. Entorno Virtual de Python (Recomendado)
Activa un subsistema aislado para prevenir conflictos de versiones y drivers SDR:

**En Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**En Linux o Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependencias
Las librerías numéricas fundamentales se instalarán usando el administrador de paquetes estándar:
```bash
pip install -r requirements.txt
```

### 4. Pruebas de Arranque (Smoke Test)
Evalúa las capacidades operativas de la red neuronal mediante:
```bash
python src/model.py
```

---

## 📝 Soporte Técnico sobre SDR Hardware
Para asegurar la estabilidad técnica durante inferencias en vivo, recomendamos instalar los drivers a nivel Sistema Operativo de tu dispositivo receptor (Zadig en Windows para dispositivos base RTL2832U, u OSmoSDR en integraciones Linux/GNU Radio).