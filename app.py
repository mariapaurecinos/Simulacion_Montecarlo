import streamlit as st
import numpy as np
import pandas as pd


class SimulacionSateliteMonteCarlo:
    def __init__(
        self,
        n_iteraciones: int,
        n_satelites_totales: int,
        n_satelites_necesarios: int,
        df: pd.DataFrame | None = None
    ):
        self.df = df
        self.n_iteraciones = int(n_iteraciones)

        if df is None:
            # Modo aleatorio: el total viene del usuario
            assert n_satelites_totales >= 1
            self.n_satelites_totales = int(n_satelites_totales)
        else:
            # Modo CSV: el total viene del número de filas del CSV
            self.n_satelites_totales = df.shape[0]

        # Mismo chequeo en ambos modos
        assert 1 <= n_satelites_necesarios <= self.n_satelites_totales
        self.n_satelites_necesarios = int(n_satelites_necesarios)

    def _simular_desde_uniforme(self) -> list[float]:
        """Modo aleatorio: cada iteración genera tiempos ~ U(1000,5000) para todos los satélites."""
        outputs = []
        for _ in range(self.n_iteraciones):
            tiempos_falla = np.random.uniform(1000, 5000, self.n_satelites_totales)
            tiempos_falla = np.sort(tiempos_falla)[::-1]  # de mayor a menor
            idx = self.n_satelites_necesarios - 1         # panel que define la falla
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)
        return outputs

    def _simular_desde_df(self) -> list[float]:
        """
        Modo CSV:
        - filas = satélites
        - columnas = experimentos/observaciones
        - Usamos TODAS las columnas: cada columna es un experimento.
        """
        data = self.df.to_numpy()  # shape: (n_sats, n_obs)
        n_sats, n_obs = data.shape
        outputs = []

        for j in range(n_obs):  # recorremos cada columna (experimento)
            tiempos_falla = data[:, j]
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)

        return outputs

    def ejecutar(self) -> tuple[float, float]:
        if self.df is None:
            outputs = self._simular_desde_uniforme()
        else:
            outputs = self._simular_desde_df()

        promedio_tiempofalla = round(float(np.mean(outputs)), 2)
        dsv_est = round(float(np.std(outputs)), 2)
        return promedio_tiempofalla, dsv_est


# APP

def main():
    st.set_page_config(page_title="Simulación Monte Carlo - Satélites", layout="centered")
    st.title("Simulación Monte Carlo de la vida útil de un satélite")

    st.markdown(
        """
        Este simulador estima el **tiempo promedio de funcionamiento** de un sistema de satélites/paneles,
        asumiendo que el sistema falla cuando hay menos satélites operativos que los necesarios.
        
        - En modo **Generar aleatorio**, los tiempos de falla se generan como Uniforme(1000, 5000).
        - En modo **Usar CSV**, debes subir un archivo **sin encabezados** donde:
          - Filas = satélites  
          - Columnas = observaciones/experimentos  
        """
    )

    st.sidebar.header("Parámetros de la simulación")

    # Modo de datos
    modo_datos = st.sidebar.radio(
        "Fuente de datos",
        ("Generar aleatorio", "Usar CSV")
    )

    df = None
    n_iteraciones = None
    n_satelites_totales = None
    n_satelites_necesarios = None

    if modo_datos == "Generar aleatorio":
        # Aquí sí se usan los tres parámetros
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
        archivo = st.sidebar.file_uploader("Sube el archivo CSV", type=["csv"])
        if archivo is not None:
            df = pd.read_csv(archivo, header=None)
            n_satelites_totales = df.shape[0]

            st.subheader("Vista previa del CSV")
            st.write(df.head())
            st.info(
                f"El CSV tiene {n_satelites_totales} filas (satélites) "
                f"y {df.shape[1]} columnas (observaciones/experimentos)."
            )

            # En modo CSV, el input relevante es SOLO el número de satélites necesarios
            n_satelites_necesarios = st.sidebar.number_input(
                "Número de satélites necesarios para que el sistema funcione",
                min_value=1,
                max_value=int(n_satelites_totales),
                value=2,
                step=1
            )

            # n_iteraciones en este modo no se usa para el cálculo (se usan todas las columnas),
            # pero le pasamos un valor dummy (1) para cumplir la firma del constructor.
            n_iteraciones = 1
        else:
            st.sidebar.info("Sube un CSV para usar esta opción.")

    if st.button("Ejecutar simulación"):
        try:
            if modo_datos == "Generar aleatorio":
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
                    n_iteraciones=int(n_iteraciones),          # no afecta en modo CSV
                    n_satelites_totales=df.shape[0],           # se reemplaza por df.shape[0] en la clase
                    n_satelites_necesarios=int(n_satelites_necesarios),
                    df=df
                )

            promedio, dsv = simulador.ejecutar()

            st.subheader("Resultados de la simulación")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tiempo promedio de falla", f"{promedio} horas")
            with col2:
                st.metric("Desviación estándar", f"{dsv} horas")

        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar la simulación: {e}")


if __name__ == "__main__":
    main()
