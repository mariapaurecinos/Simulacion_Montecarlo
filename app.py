import streamlit as st
import numpy as np
import pandas as pd

# Pega aquí la clase SimulacionSateliteMonteCarlo
from simulador_satelites import SimulacionSateliteMonteCarlo  # o pega la clase directamente arriba


def main():
    st.set_page_config(page_title="Simulación Monte Carlo - Satélites", layout="centered")
    st.title("Simulación Monte Carlo de la vida útil de un satélite")

    st.markdown(
        """
        Este simulador estima el **tiempo promedio de funcionamiento** de un sistema de satélites/paneles,
        asumiendo que el sistema falla cuando hay menos satélites operativos que los necesarios.
        
        - Cuando **no subes CSV**, los tiempos de falla se generan como Uniforme(1000, 5000).
        - Cuando **subes CSV**, se usa directamente la matriz de datos:
          - Filas = satélites/paneles  
          - Columnas = experimentos (escenarios)
        """
    )

    st.sidebar.header("Parámetros de la simulación")

    # Modo de datos
    modo_datos = st.sidebar.radio(
        "Fuente de datos",
        ("Generar aleatorio (Uniforme[1000,5000])", "Usar CSV (filas = satélites, columnas = experimentos)")
    )

    df = None
    n_iteraciones = None
    n_satelites_totales = None
    n_satelites_necesarios = None

    if modo_datos == "Generar aleatorio (Uniforme[1000,5000])":
        # Solo en este modo pedimos n_satelites_totales
        n_satelites_totales = st.sidebar.number_input(
            "Número total de satélites/paneles",
            min_value=1,
            max_value=1000,
            value=5,
            step=1
        )

        n_satelites_necesarios = st.sidebar.number_input(
            "Número de satélites necesarios para que el sistema funcione",
            min_value=1,
            max_value=int(n_satelites_totales),
            value=2,
            step=1
        )

        n_iteraciones = st.sidebar.number_input(
            "Número de iteraciones de Monte Carlo",
            min_value=1,
            max_value=1000000,
            value=1000,
            step=1
        )

    else:  # Usar CSV
        st.markdown(
            """
            ### Formato requerido del CSV
            
            - Cada **fila** representa un satélite/panel.  
            - Cada **columna** representa un experimento (escenario).  
            - Los valores deben ser los **tiempos de falla**.
            """
        )
        archivo = st.sidebar.file_uploader("Sube el archivo CSV", type=["csv"])
        if archivo is not None:
            df = pd.read_csv(archivo, header=None)  # sin encabezados
            n_satelites_totales = df.shape[0]
            n_iteraciones = df.shape[1]

            st.subheader("Vista previa del CSV")
            st.write(df.head())
            st.info(
                f"El CSV tiene {n_satelites_totales} filas (satélites) y {n_iteraciones} columnas (experimentos)."
            )

            n_satelites_necesarios = st.sidebar.number_input(
                "Número de satélites necesarios para que el sistema funcione",
                min_value=1,
                max_value=int(n_satelites_totales),
                value=min(2, n_satelites_totales),
                step=1
            )
        else:
            st.sidebar.info("Sube un CSV para usar esta opción.")

    if st.button("Ejecutar simulación"):
        try:
            if modo_datos == "Generar aleatorio (Uniforme[1000,5000])":
                simulador = SimulacionSateliteMonteCarlo(
                    n_iteraciones=int(n_iteraciones),
                    n_satelites_totales=int(n_satelites_totales),
                    n_satelites_necesarios=int(n_satelites_necesarios),
                    df=None
                )
            else:
                if df is None:
                    st.error("Seleccionaste 'Usar CSV', pero no se ha subido ningún archivo.")
                    return
                simulador = SimulacionSateliteMonteCarlo(
                    n_iteraciones=0,  # no se usa en modo CSV
                    n_satelites_totales=0,  # se ignora, se usa df.shape[0]
                    n_satelites_necesarios=int(n_satelites_necesarios),
                    df=df
                )

            promedio, dsv, outputs = simulador.ejecutar()

            st.subheader("Resultados de la simulación")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tiempo promedio de falla", f"{promedio} horas")
            with col2:
                st.metric("Desviación estándar", f"{dsv} horas")

            # Tabla solo para el caso aleatorio
            if modo_datos == "Generar aleatorio (Uniforme[1000,5000])":
                st.subheader("Primeras 10 iteraciones (datos simulados)")
                detalle_df = pd.DataFrame({
                    "iteración": np.arange(1, len(outputs) + 1),
                    "tiempo_falla_sistema": outputs
                })
                st.dataframe(detalle_df.head(10))

        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar la simulación: {e}")


if __name__ == "__main__":
    main()
