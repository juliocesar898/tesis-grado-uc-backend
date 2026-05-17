import matplotlib.pyplot as plt
import numpy as np


def visualize_iq(iq_data):
    """
    Recibe un arreglo de datos complejos IQ y genera sus representaciones gráficas.
    Incluye dominio temporal, diagrama de constelación vectorizado y espectro en frecuencia.
    """
    # Separar componentes en Fase (I) y Cuadratura (Q)
    i_channel = np.real(iq_data)
    q_channel = np.imag(iq_data)

    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    fig.subplots_adjust(hspace=0.45)

    # -------------------------------------------------------------------------
    # Gráfico 1: Dominio del Tiempo
    # -------------------------------------------------------------------------
    axs[0].plot(i_channel[:128], label="En Fase (I) - Real", color="blue", alpha=0.8)
    axs[0].plot(
        q_channel[:128], label="Cuadratura (Q) - Imag", color="orange", alpha=0.8
    )
    axs[0].set_title(
        "Dominio del Tiempo (Primeras 128 Muestras)", fontsize=12, fontweight="bold"
    )
    axs[0].set_xlabel("Muestra")
    axs[0].set_ylabel("Amplitud")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Gráfico 2: Diagrama de Constelación (I vs Q) - CON AJUSTES DE LABORATORIO
    # -------------------------------------------------------------------------
    axs[1].scatter(
        i_channel, q_channel, alpha=0.6, color="purple", marker=".", label="Muestras IQ"
    )
    axs[1].set_title(
        "Diagrama de Constelación (Plano Complejo)", fontsize=12, fontweight="bold"
    )
    axs[1].set_xlabel("En Fase (I)")
    axs[1].set_ylabel("Cuadratura (Q)")

    # Ejes cartesianos de referencia cruzando por el origen (0,0)
    axs[1].axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
    axs[1].axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.5)

    # Forzar límites simétricos para mantener la geometría de la modulación
    max_val = max(max(abs(i_channel)), max(abs(q_channel)), 1.2)
    axs[1].set_xlim(-max_val, max_val)
    axs[1].set_ylim(-max_val, max_val)

    # Etiquetas de texto informativas para identificar cuadrantes trigonométricos
    # Fijadas a las esquinas relativas (transAxes) para no ensuciar los datos

    # Q1 (Arriba a la Derecha)
    axs[1].text(
        0.98,
        0.96,
        "Q1 (+,+)",
        transform=axs[1].transAxes,
        fontsize=9,
        color="gray",
        alpha=0.7,
        fontweight="bold",
        ha="right",
        va="top",
    )
    # Q2 (Arriba a la Izquierda)
    axs[1].text(
        0.02,
        0.96,
        "Q2 (-,+)",
        transform=axs[1].transAxes,
        fontsize=9,
        color="gray",
        alpha=0.7,
        fontweight="bold",
        ha="left",
        va="top",
    )
    # Q3 (Abajo a la Izquierda)
    axs[1].text(
        0.02,
        0.04,
        "Q3 (-,-)",
        transform=axs[1].transAxes,
        fontsize=9,
        color="gray",
        alpha=0.7,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    # Q4 (Abajo a la Derecha)
    axs[1].text(
        0.98,
        0.04,
        "Q4 (+,-)",
        transform=axs[1].transAxes,
        fontsize=9,
        color="gray",
        alpha=0.7,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    axs[1].grid(True, linestyle=":", alpha=0.5)
    axs[1].axis(
        "equal"
    )  # Evita que los círculos o cuadrados se deformen por la escala de la pantalla

    # -------------------------------------------------------------------------
    # Gráfico 3: Dominio de la Frecuencia (PSD)
    # -------------------------------------------------------------------------
    # Fs se establece a 2.048 MHz (Frecuencia de muestreo estándar de RTL-SDR)
    axs[2].psd(
        iq_data,
        NFFT=1024 if len(iq_data) > 1024 else len(iq_data),
        Fs=2.048e6,
        color="green",
    )
    axs[2].set_title(
        "Densidad Espectral de Potencia (Dominio de la Frecuencia)",
        fontsize=12,
        fontweight="bold",
    )
    axs[2].set_xlabel("Frecuencia (Hz)")
    axs[2].set_ylabel("Potencia (dB / Hz)")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    # Mostrar la interfaz unificada de análisis
    plt.show()
