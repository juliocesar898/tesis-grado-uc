import numpy as np
import logging
from scipy import signal


def normalize_iq(iq_data):
    """
    Normaliza los datos IQ crudos (arreglos complejos) a su máximo valor absoluto.
    Retorna un tensor [Tiempo, Canales] adecuado para la CNN.
    """
    logging.info("Normalizando datos IQ crudos.")
    iq_data = np.asarray(iq_data, dtype=complex)

    max_val = np.max(np.abs(iq_data))
    if max_val > 0:
        iq_data = iq_data / max_val

    iq_tensor = np.column_stack((np.real(iq_data), np.imag(iq_data)))

    # Validación de dimensiones: [Longitud_Muestras, 2 Canales (I/Q)]
    expected_shape = (len(iq_data), 2)
    if iq_tensor.shape != expected_shape:
        logging.error(
            f"Error de dimensiones en normalize_iq. Actual: {iq_tensor.shape}, Esperado: {expected_shape}"
        )
        raise ValueError("El tensor IQ normalizado no tiene las dimensiones correctas.")

    return iq_tensor


def spectrogram_stft(iq_data, fs=1.0):
    """
    Transforma los datos IQ en espectrogramas usando STFT.
    Apropiado para arquitecturas CNN 2D.
    """
    logging.info("Generando espectrograma usando STFT.")
    iq_data = np.asarray(iq_data, dtype=complex)

    f, t, Zxx = signal.stft(iq_data, fs=fs, nperseg=32)
    espectrograma = np.abs(Zxx)
    espectrograma_tensor = np.expand_dims(espectrograma, axis=-1)

    # Validación de dimensiones: [Frecuencias, Tiempos, Canal (1)]
    if len(espectrograma_tensor.shape) != 3 or espectrograma_tensor.shape[-1] != 1:
        logging.error(
            f"Error de dimensiones en spectrogram_stft. Actual: {espectrograma_tensor.shape}"
        )
        raise ValueError(
            "El espectrograma debe tener formato tridimensional [Frecuencia, Tiempo, Canales]."
        )

    return espectrograma_tensor
