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
        """
        n_iteraciones:
            Número de iteraciones de Monte Carlo.
        n_satelites_totales:
            En modo aleatorio: número total de satélites.
            En modo CSV: puedes pasar cualquier cosa, se sobreescribe con df.shape[0].
        n_satelites_necesarios:
            Número de satélites/paneles necesarios para que el sistema funcione.
        df:
            Si es None => se generan tiempos ~ Uniforme(1000,5000).
            Si NO es None => df tiene filas = satélites, columnas = observaciones/experimentos.
        """
        self.df = df
        self.n_iteraciones = int(n_iteraciones)

        if df is None:
            # Modo aleatorio
            assert n_satelites_totales >= 1
            self.n_satelites_totales = int(n_satelites_totales)
        else:
            # Modo CSV: el número de satélites lo define el CSV
            self.n_satelites_totales = df.shape[0]

        assert 1 <= n_satelites_necesarios <= self.n_satelites_totales
        self.n_satelites_necesarios = int(n_satelites_necesarios)

    def _simular_desde_uniforme(self) -> list[float]:
        """Caso df is None: generar datos aleatorios Uniforme(1000,5000)."""
        outputs = []
        for _ in range(self.n_iteraciones):
            tiempos_falla = np.random.uniform(1000, 5000, self.n_satelites_totales)
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)
        return outputs

    def _simular_desde_df(self) -> list[float]:
        """
        Caso df is not None:
        - filas = satélites
        - columnas = observaciones/experimentos para cada satélite
        - En cada iteración de Monte Carlo:
          - para cada satélite se elige aleatoriamente una columna (con reemplazo)
          - se calcula el tiempo de falla del sistema a partir de esos tiempos.
        """
        data = self.df.to_numpy()              # shape: (n_sats, n_obs)
        n_sats, n_obs = data.shape
        outputs = []

        for _ in range(self.n_iteraciones):
            # para cada satélite elegimos una columna al azar
            col_indices = np.random.randint(0, n_obs, size=n_sats)
            tiempos_falla = data[np.arange(n_sats), col_indices]
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


# ------------------- STREAMLIT APP -------------------

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
          - Columnas = observaciones/experimentos  
        """
    )

    st.sidebar.header("Parámetros de la simulación")

    # Modo de datos
    modo_datos = st.sidebar.radio(
        "Fuente de datos",
        ("Generar aleatorio (Uniforme[1000,5000])", "Usar CSV (filas = satélites, columnas = observaciones)")
    )

    df = None
    n_iteraciones = None
    n_satelites_totales = None
    n_satelites_necesarios = None

    if modo_datos == "Generar aleatorio (Uniforme[1000,5000])":
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
        st.markdown(
            """
            ### Formato requerido del CSV
            
            - Cada **fila** representa un satélite/panel.  
            - Cada **columna** representa un experimento/observación.  
            - Los valores deben ser los **tiempos de falla**.
            """
        )
        archivo = st.sidebar.file_uploader("Sube el archivo CSV", type=["csv"])
        if archivo is not None:
            df = pd.read_csv(archivo, header=None)
            n_satelites_totales = df.shape[0]

            st.subheader("Vista previa del CSV")
            st.write(df.head())
            st.info(
                f"El CSV tiene {n_satelites_totales} filas (satélites) "
                f"y {df.shape[1]} columnas (observaciones)."
            )

            # ÚNICO parámetro que se considera desde el usuario en este modo:
            n_iteraciones = st.sidebar.number_input(
                "Número de iteraciones de Monte Carlo",
                min_value=1,
                max_value=1000000,
                value=1000,
                step=1
            )

            # Para este problema concreto asumimos que el sistema necesita al menos 2 paneles operativos.
            n_satelites_necesarios = 2
            st.sidebar.write(f"Para el modo CSV se asume: satélites necesarios = {n_satelites_necesarios}")
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
                    n_iteraciones=int(n_iteraciones),
                    n_satelites_totales=df.shape[0],  # se ignora dentro, se usa df.shape[0]
                    n_satelites_necesarios=int(n_satelites_necesarios),  # aquí es 2
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
