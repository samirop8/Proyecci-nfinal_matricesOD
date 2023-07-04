import pandas as pd
import numpy as np
from proyecciones import *
from tiempoEjecucion import TiempoEjecucion


if __name__ == '__main__':

    matriz_od = ProcesamientoMOD()

    matriz_od.set_ratios("./ficheros_excel/matriz_original_completa.xlsx")

    matriz_od.calculo_zonas_colindantes()

    pares_od = list(set(list(zip(matriz_od.df_ratios['origen'].to_numpy(), matriz_od.df_ratios['destino'].to_numpy(), matriz_od.df_ratios['periodo'].to_numpy()))))

    array_pares_od = np.array_split(pares_od, PROCESOS_DISPONIBLES)

    tiempo_ejecucion = TiempoEjecucion('Nueva version 12/01')

    matriz_od.ejecucion_paralela()

    tiempo_ejecucion.get_tiempo_ejecucion()

    df_final = pd.concat(resultados)

    df_final.to_csv("./resultado_proyecciones.csv", sep=";", index=False)


