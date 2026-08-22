import time
import numpy as np

try:
    import adi
    print(">> Biblioteca pyadi-iio cargada correctamente.")
    
    # 1. Enlace con la IP del Pluto+ por USB RNDIS
    sdr = adi.Pluto("ip:192.168.2.1")
    
    # 2. Configurar sintonizador para banda libre (ej. ISM 915 MHz)
    sdr.rx_lo = int(915e6)
    sdr.sample_rate = int(1e6)  # 1 MSPS
    sdr.rx_rf_bandwidth = int(1e6)
    sdr.rx_buffer_size = 1024
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 30
    
    print(">> Conectado al Pluto+ SDR exitosamente!")
    
    # 3. Capturar ráfaga
    print(">> Leyendo buffer...")
    samples = sdr.rx()
    print(f">> Muestras obtenidas: {len(samples)} complexes.")
    print(f">> Muestra 0 (Formato complejo I+jQ): {samples[0]}")
    
    # Liberar puerto
    del sdr
    print(">> Puerto liberado con éxito.")
    
except Exception as e:
    print(f"❌ Error en la prueba de hardware: {e}")