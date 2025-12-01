import streamlit as st
import numpy as np
import pandas as pd

# ---- pega aquí la clase SimulacionSateliteMonteCarlo definida antes ----
import numpy as np
import pandas as pd

class SimulacionSateliteMonteCarlo:
    def __init__(self, n_iteraciones: int, n_satelites_totales: int, n_satelites_necesarios: int, df: pd.DataFrame | None = None):
        """
        n_iteraciones: número de corridas de Monte Carlo (solo se usa cuando df es None).
        n_satelites_totales: número total de satélites/paneles.
        n_satelites_necesarios: cuántos satélites/paneles deben estar operativos para que el sistema funcione.
        df: opcional, matriz de datos donde filas = satélites y columnas = experimentos.
        """
        assert n_satelites_totales >= 1, "Debe haber al menos 1 satélite."
        assert 1 <= n_satelites_necesarios <= n_satelites_totales, "Los satélites necesarios deben estar entre 1 y el total."
        
        self.n_iteraciones = n_iteraciones
        self.n_satelites_totales = n_satelites_totales
        self.n_satelites_necesarios = n_satelites_necesarios
        self.df = df

    def _simular_desde_uniforme(self) -> list[float]:
        """Caso df is None: generar datos aleatorios Uniforme(1000,5000)."""
        outputs = []
        for _ in range(self.n_iteraciones):
            # tiempos de falla para cada satélite/panel
            tiempos_falla = np.random.uniform(1000, 5000, self.n_satelites_totales)
            # ordenar de mayor a menor
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            # índice del satélite que define la falla del sistema
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)
        return outputs

    def _simular_desde_df(self) -> list[float]:
        """
        Caso df is not None: cada columna es un experimento, cada fila un satélite.
        """
        outputs = []
        # checamos que el número de filas coincida con el número de satélites
        if self.df.shape[0] != self.n_satelites_totales:
            raise ValueError(
                f"El CSV debe tener {self.n_satelites_totales} filas (una por satélite). "
                f"Actualmente tiene {self.df.shape[0]} filas."
            )

        for i in range(self.df.shape[1]):  # columnas = experimentos
            tiempos_falla = self.df.iloc[:, i].values
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)
        return outputs

    def ejecutar(self) -> tuple[float, float]:
        """
        Ejecuta la simulación y devuelve:
        - promedio del tiempo de falla
        - desviación estándar del tiempo de falla
        """
        if self.df is None:
            outputs = self._simular_desde_uniforme()
        else:
            outputs = self._simular_desde_df()

        promedio_tiempofalla = round(float(np.mean(outputs)), 2)
        dsv_est = round(float(np.std(outputs)), 2)
        return promedio_tiempofalla, dsv_est


class SimulacionSateliteMonteCarlo:
    def __init__(self, n_iteraciones: int, n_satelites_totales: int, n_satelites_necesarios: int, df: pd.DataFrame | None = None):
        assert n_satelites_totales >= 1, "Debe haber al menos 1 satélite."
        assert 1 <= n_satelites_necesarios <= n_satelites_totales, "Los satélites necesarios deben estar entre 1 y el total."
        
        self.n_iteraciones = n_iteraciones
        self.n_satelites_totales = n_satelites_totales
        self.n_satelites_necesarios = n_satelites_necesarios
        self.df = df

    def _simular_desde_uniforme(self) -> list[float]:
        outputs = []
        for _ in range(self.n_iteraciones):
            tiempos_falla = np.random.uniform(1000, 5000, self.n_satelites_totales)
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_falla[idx]
            outputs.append(tiempo_expected_simulacion)
        return outputs

    def _simular_desde_df(self) -> list[float]:
        outputs = []
        if self.df.shape[0] != self.n_satelites_totales:
            raise ValueError(
                f"El CSV debe tener {self.n_satelites_totales} filas (una por satélite). "
                f"Actualmente tiene {self.df.shape[0]} filas."
            )

        for i in range(self.df.shape[1]):
            tiempos_falla = self.df.iloc[:, i].values
            tiempos_falla = np.sort(tiempos_falla)[::-1]
            idx = self.n_satelites_necesarios - 1
            tiempo_expected_simulacion = tiempos_fallas[idx]
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
        - Cuando **subes CSV**, se usa directamente la matriz de datos.
        """
    )

    st.sidebar.header("Parámetros de la simulación")

    # Modo de datos
    modo_datos = st.sidebar.radio(
        "Fuente de datos",
        ("Generar aleatorio (Uniforme[1000,5000])", "Usar CSV (filas = satélites, columnas = experimentos)")
    )

    # Inputs comunes
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

    df = None
    n_iteraciones = None

    if modo_datos == "Generar aleatorio (Uniforme[1000,5000])":
        n_iteraciones = st.sidebar.number_input(
            "Número de iteraciones de Monte Carlo",
            min_value=1,
            max_value=1000000,
            value=1000,
            step=1
        )
    else:
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
            df = pd.read_csv(archivo, header=None)  # sin encabezados, puro numerito
            st.subheader("Vista previa del CSV")
            st.write(df.head())
            st.info(f"El CSV tiene {df.shape[0]} filas (satélites) y {df.shape[1]} columnas (experimentos).")
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
                    n_iteraciones=0,  # no se usa en este modo
                    n_satelites_totales=int(n_satelites_totales),
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
