"""
DeepSignal - Automatic Modulation Classification (AMC) Model
Arquitectura de CNN para Procesamiento de Señales Digitales I/Q.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  # type: ignore
import logging
from pathlib import Path

# Configuración agnóstica al Sistema Operativo para la carpeta de modelos
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_pretrained_model(model_name="amc_model.h5"):
    """
    Carga un modelo preentrenado desde el directorio de modelos.
    """
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        logging.info(f"Cargando modelo existente desde: {model_path}")
        return models.load_model(model_path)
    return None


def save_model(model, model_name="amc_model.h5"):
    """
    Guarda el modelo entrenado en el directorio de modelos, creando
    la carpeta si no existe.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    model.save(model_path)
    logging.info(f"Modelo guardado exitosamente en: {model_path}")


def build_amc_cnn(input_shape=(128, 2), num_classes=11, l2_reg=1e-4):
    """
    Construye una Red Neuronal Convolucional (CNN) orientada a la
    clasificación de señales base I/Q.

    Args:
        input_shape (tuple): Dimensiones de las muestras de entrada. Por
                             defecto (128, 2) indica 128 instantes temporales
                             para los 2 canales espaciales (I y Q).
        num_classes (int): Cantidad de modulaciones posibles a clasificar
                           (e.g., BPSK, QPSK, 8PSK, 16QAM, 64QAM, FM, etc.).
        l2_reg (float): Parámetro para la regularización L2 evitando overfitting.

    Returns:
        model (tf.keras.Model): Modelo Keras compilado.
    """
    model = models.Sequential(name="DeepSignal_AMC_CNN")

    # --- Bloque Extractor de Características Témporo-Espaciales ---

    # Capa Conv1: Extrae patrones locales en la forma de onda
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=input_shape,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Capa Conv2: Combina abstracciones para identificar patrones de fase/amplitud
    model.add(
        layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Capa Conv3: Profundización semántica
    model.add(
        layers.Conv1D(
            filters=256,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # --- Bloque Clasificador ---

    model.add(layers.Flatten())

    model.add(
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))
    )
    model.add(
        layers.Dropout(0.5)
    )  # Dropout agresivo vital para datos de radiofrecuencia (ruido ruidoso)

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))

    # Salida categórica con Softmax
    model.add(layers.Dense(num_classes, activation="softmax", name="mod_prediction"))

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Prueba de Humo y Arquitectura Base
    print("Inicializando Arquitectura Híbrida DeepSignal AMC...")

    # Ej: Asumiendo una longitud de trama (window) de 128 muestras y 2 canales (I, Q)
    # y 11 modulaciones extraídas del estándar RadioML 2016.10A
    amc_model = build_amc_cnn(input_shape=(128, 2), num_classes=11)

    amc_model.summary()
    print("✓ Modelo configurado y listo para ser entrenado iterativamente.")
