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

    def simulacionmontecarlo(self):

        variables_aleatorias = self.variables_aleatorias
        tamaño_muestra = self.tamaño_muestra
        criterio_sorteo = self.criterio_sorteo
        df = self.df

        columnas = ["experimento"] + [f"panel {i+1}" for i in range(variables_aleatorias)] + ["tiempo falla"]
        df = pd.DataFrame(columns = columnas)

        for experimento in range(tamaño_muestra):
            tiempos_fallas = np.random.uniform(1000, 5000, variables_aleatorias)
            tiempo_falla = np.sort(tiempos_fallas)[variables_aleatorias - criterio_sorteo]
            fila = [experimento + 1] + list(tiempos_fallas) + [tiempo_falla]
            df.loc[len(df)] = fila

        self.df = df
        return df

def main():
    st.set_page_config(page_title = "Simulación Montecarlo", 
                       layout = "wide", 
                       initial_sidebar_state = "expanded")
    
    st.title("Simulación Montecarlo")

    st.sidebar.header("Parámetros (usar los botones o dar click en el cuadro de texto para modificar)")
    
    variables_aleatorias = st.sidebar.number_input(
        label = "Número de variables aleatorias",
        min_value = 1,
        max_value = 1000000,
        value = 5
    )
    
    tamaño_muestra = st.sidebar.number_input(
        label  = "Tamaño de muestra",
        min_value = 1,
        max_value = 1000001,
        value = 6
    )
    
    criterio_sorteo = st.sidebar.number_input(
        label ="Criterio de sorteo (número de páneles necesarios)",
        min_value = 1,
        max_value = variables_aleatorias,
        value = 2
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        ejecutar_simulacion = st.button("Ejecutar simulación")
    with col2:
        descargar_csv = st.button("Descargar csv")

    if ejecutar_simulacion:
        st.info(f"Simulación con número de variables aleatorias = {variables_aleatorias}, tamaño de muestra = {tamaño_muestra} y criterio de sorteo = {criterio_sorteo}")
        
        entrada = Entrada(variables_aleatorias, tamaño_muestra, criterio_sorteo)
        algoritmo = SimulacionMontecarlo(entrada.variables_aleatorias, entrada.tamaño_muestra, entrada.criterio_sorteo, pd.DataFrame())
        
        resultado_df = algoritmo.simulacionmontecarlo()
        
        st.subheader("Resultados")
        st.dataframe(resultado_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Experimentos", len(resultado_df))
        with col2:
            st.metric("Tiempo de vida promedio", f"{resultado_df['tiempo falla'].mean():.3f}")

        st.session_state.resultado_df = resultado_df

    if descargar_csv and hasattr(st.session_state, 'resultado_df'):
        csv_data = st.session_state.resultado_df.to_csv(index = False)
        st.download_button(
            label = "Descargar CSV",
            data = csv_data,
            file_name = "simulacionMontecarlo.csv"
        )
    elif descargar_csv:
        st.warning("Ejecutar primero")

if __name__ == "__main__":
    main()
    