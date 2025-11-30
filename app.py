import numpy as np
import pandas as pd
import streamlit as st

class Entrada:
    def __init__(self, variables_aleatorias: int, tamaño_muestra: int, criterio_sorteo: int):
        assert variables_aleatorias > 3, "número de variables aleatorias debe ser mayor a 3 para mejores resultados"
        assert tamaño_muestra > 0, "tamaño de muestra debe ser mayor a 0"
        assert criterio_sorteo <= variables_aleatorias, "El criterio de sorteo k debe ser menor o igual al número de variables aleatorias"
        self.variables_aleatorias = variables_aleatorias
        self.tamaño_muestra = tamaño_muestra
        self.criterio_sorteo = criterio_sorteo


class SimulacionMontecarlo(Entrada):
    def __init__(self, variables_aleatorias: int, tamaño_muestra: int, criterio_sorteo: int, df: pd.DataFrame):
        super().__init__(variables_aleatorias, tamaño_muestra, criterio_sorteo)
        self.df = df

    # --- MODIFICACIÓN 1: ahora recibe opcionalmente datos_hist (vector desde CSV) ---
    def simulacionmontecarlo(self, datos_hist=None):

        variables_aleatorias = self.variables_aleatorias
        tamaño_muestra = self.tamaño_muestra
        criterio_sorteo = self.criterio_sorteo

        columnas = ["experimento"] + [f"panel {i+1}" for i in range(variables_aleatorias)] + ["tiempo falla"]
        df = pd.DataFrame(columns=columnas)

        for experimento in range(tamaño_muestra):
            # --- MODIFICACIÓN 2: lógica para elegir fuente de datos ---
            if datos_hist is None:
                # Caso original: números aleatorios uniformes
                tiempos_fallas = np.random.uniform(1000, 5000, variables_aleatorias)
            else:
                # Nuevo caso: tomar tiempos al azar de los datos históricos (CSV)
                tiempos_fallas = np.random.choice(datos_hist, size=variables_aleatorias, replace=True)

            tiempo_falla = np.sort(tiempos_fallas)[variables_aleatorias - criterio_sorteo]
            fila = [experimento + 1] + list(tiempos_fallas) + [tiempo_falla]
            df.loc[len(df)] = fila

        self.df = df
        return df


def main():
    st.set_page_config(page_title="Simulación Montecarlo",
                       layout="wide",
                       initial_sidebar_state="expanded")
    
    st.title("Simulación Montecarlo")

    st.sidebar.header("Parámetros (usar los botones o dar click en el cuadro de texto para modificar)")

    # --- MODIFICACIÓN 3: selector de fuente de datos ---
    st.sidebar.subheader("Fuente de datos para los tiempos de falla")
    fuente_datos = st.sidebar.radio(
        "Selecciona cómo obtener los datos:",
        ("Generar números aleatorios", "Usar CSV de datos")
    )

    # --- MODIFICACIÓN 4: manejo del CSV y columna numérica ---
    datos_hist = None
    if fuente_datos == "Usar CSV de datos":
        archivo = st.sidebar.file_uploader(
            "Sube un archivo CSV con los datos históricos",
            type=["csv"]
        )
        if archivo is not None:
            try:
                df_csv = pd.read_csv(archivo)
                st.sidebar.write("Vista previa del CSV:")
                st.sidebar.dataframe(df_csv.head())

                columnas_numericas = df_csv.select_dtypes(include=[np.number]).columns.tolist()
                if not columnas_numericas:
                    st.sidebar.error("No se encontraron columnas numéricas en el CSV.")
                else:
                    columna_seleccionada = st.sidebar.selectbox(
                        "Selecciona la columna con los tiempos de falla / datos para simular",
                        columnas_numericas
                    )
                    datos_hist = df_csv[columna_seleccionada].dropna().values
                    st.sidebar.success(f"Se cargaron {len(datos_hist)} datos de '{columna_seleccionada}'.")
            except Exception as e:
                st.sidebar.error(f"Error al leer el CSV: {e}")
        else:
            st.sidebar.info("Sube un CSV para poder usar esta opción.")

    # --- MODIFICACIÓN 5: min_value alineado con la aserción (>3) ---
    variables_aleatorias = st.sidebar.number_input(
        label="Número de variables aleatorias",
        min_value=4,  # antes era 1, pero falla con la aserción (>3)
        max_value=1000000,
        value=5
    )
    
    tamaño_muestra = st.sidebar.number_input(
        label="Tamaño de muestra",
        min_value=1,
        max_value=1000001,
        value=6
    )
    
    criterio_sorteo = st.sidebar.number_input(
        label="Criterio de sorteo (número de páneles necesarios)",
        min_value=1,
        max_value=variables_aleatorias,
        value=2
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        ejecutar_simulacion = st.button("Ejecutar simulación")
    with col2:
        descargar_csv = st.button("Descargar csv")

    if ejecutar_simulacion:
        # --- MODIFICACIÓN 6: validación cuando se elige CSV pero no hay datos válidos ---
        if fuente_datos == "Usar CSV de datos" and (datos_hist is None or len(datos_hist) == 0):
            st.error("Seleccionaste 'Usar CSV de datos', pero no hay datos válidos cargados. Sube un CSV y elige una columna numérica.")
        else:
            st.info(
                f"Simulación con número de variables aleatorias = {variables_aleatorias}, "
                f"tamaño de muestra = {tamaño_muestra}, criterio de sorteo = {criterio_sorteo} "
                f"y fuente de datos = '{fuente_datos}'."
            )
            
            entrada = Entrada(variables_aleatorias, tamaño_muestra, criterio_sorteo)
            algoritmo = SimulacionMontecarlo(
                entrada.variables_aleatorias,
                entrada.tamaño_muestra,
                entrada.criterio_sorteo,
                pd.DataFrame()
            )
            
            # --- MODIFICACIÓN 7: pasamos datos_hist a la simulación ---
            resultado_df = algoritmo.simulacionmontecarlo(datos_hist=datos_hist)
            
            st.subheader("Resultados")
            st.dataframe(resultado_df)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Experimentos", len(resultado_df))
            with col2:
                st.metric("Tiempo de vida promedio", f"{resultado_df['tiempo falla'].mean():.3f}")

            st.session_state.resultado_df = resultado_df

    if descargar_csv and hasattr(st.session_state, 'resultado_df'):
        csv_data = st.session_state.resultado_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name="simulacionMontecarlo.csv"
        )
    elif descargar_csv:
        st.warning("Ejecutar primero")

if __name__ == "__main__":
    main()
