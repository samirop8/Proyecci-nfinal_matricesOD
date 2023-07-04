from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import multiprocessing
from copy import deepcopy
from tiempoEjecucion import TiempoEjecucion
import itertools


PROCESOS_DISPONIBLES = multiprocessing.cpu_count()
resultados = []

def juntar_resultados(resultado):
    print("Ha finalizado un proceso, el resultado fue: \n", resultado)
    resultados.append(resultado)


class ProcesamientoMOD:

    def __init__(self):
        
        self.gdf_zonas_telefonia = gpd.GeoDataFrame()
        self.zonas = []
        self.gdf_zonas_colindantes = gpd.GeoDataFrame()
        self.zonas_colindantes = {}
        self.df_paradas = pd.DataFrame()
        self.df_ratios = pd.DataFrame()


    def set_zonificacion_telefonia(self, fichero_zonas_telefonia):
        """ Método setter para obtener la MOD de telefonía desde un .shp

        Args:
            fichero_zonas_telefonia (str): ruta del fichero con la MOD de telefonía

        Returns:
            GeoDataFrame: Geopandas dataframe con las zonas de telefonía
        """
        try:
            assert '.shp' in fichero_zonas_telefonia 
        except:
            print("Se debe de introducir un fichero de tipo .shp")
            exit(0)
        else:
            self.gdf_zonas_telefonia = gpd.read_file(fichero_zonas_telefonia)
            self.gdf_zonas_telefonia['ID'] = self.gdf_zonas_telefonia['ID'].astype('int64')

    def set_zonificacion_titsa(self, fichero_zonas_titsa):
        self.gdf_zonas = gpd.read_file(fichero_zonas_titsa)
        self.gdf_zonas['ID'] = self.gdf_zonas['ID'].astype('int64')



    def calculo_zona_parada(self):
        """ Método para situar cada una de las paradas de TITSA en la zona de la MOD de
            telefonía donde quedan ubicadas

        Args:

        Returns:
        """

        tiempo_ejecucion = TiempoEjecucion("ubicar cada parada")

        gdf_paradas = gpd.GeoDataFrame(self.df_paradas, geometry=gpd.points_from_xy(self.df_paradas.LONGITUD, self.df_paradas.LATITUD), crs=4326).to_crs(32628)

        self.gdf_zonas_colindantes = self.gdf_zonas_colindantes.to_crs(32628)

        self.gdf_zonas_colindantes_con_paradas = gpd.tools.sjoin(gdf_paradas, self.gdf_zonas_colindantes)

        self.gdf_zonas_colindantes_con_paradas = self.gdf_zonas_colindantes_con_paradas.reset_index()

        tiempo_ejecucion.get_tiempo_ejecucion()


    def set_ratios(self, fichero_ratios):
        """Método para cargar las los ratios

            Args:
                fichero_ratios: Ruta donde se encuentra el fichero de ratios

            Returns:
        """

        self.df_ratios = pd.read_excel(fichero_ratios)

        self.df_ratios["N_PASAJEROS"].fillna(0, inplace = True)
        self.df_ratios["ratio"].fillna(0, inplace = True)


    def get_ratios_tp_tel(self):
        """ Método getter para obtener el dataframe con los ratios

        Args:

        Returns:
            DataFrame: Pandas dataframe con los ratios
        """
        
        return self.df_ratios


    def calculo_zonas_colindantes(self):
        """ Método para obtener las zonas colindantes que tiene cada una de las zonas de
            la MOD de telefonía

        Args:

        Returns:
        """

        tiempo_ejecucion = TiempoEjecucion("calcular zonas colindantes")

        self.gdf_zonas_colindantes = self.gdf_zonas_telefonia

        self.gdf_zonas_colindantes["ZONAS_COL"] = None

        for indice, zona in self.gdf_zonas_colindantes.iterrows():   

            zonas_colindantes = self.gdf_zonas_colindantes[~self.gdf_zonas_colindantes.geometry.disjoint(zona.geometry)].ID.tolist()

            zonas_colindantes = [ zona for zona in zonas_colindantes if zona.ID != zona ]

            self.gdf_zonas_colindantes.at[indice, "ZONAS_COL"] = str(zonas_colindantes)

        self.gdf_zonas_colindantes.to_excel("./zonas_colindantes.xlsx")

        self.zonas = np.unique(self.gdf_zonas_colindantes['ID'].tolist())

        for zona in self.zonas:
            self.zonas_colindantes[zona] = list(map(int, (np.unique(self.gdf_zonas_colindantes[self.gdf_zonas_colindantes['ID'] == zona]['ZONAS_COL'])[0].replace("[", "").replace("]", "").split(", "))))

        tiempo_ejecucion.get_tiempo_ejecucion()


    def get_zonas_colindates(self):

        return self.zonas_colindantes
    

    def ejecucion_paralela(self):

        tiempo_ejecucion = TiempoEjecucion('Nueva version 31/01')

        df_resultado = pd.DataFrame()

        # Cambio del formato de las horas 'P19' --> 19
        self.df_ratios['periodo'] = self.df_ratios.periodo.str.slice(-2)
        self.df_ratios['periodo'] = pd.to_numeric(self.df_ratios['periodo'])

        array_pares_od_per = list(set(list(zip(self.df_ratios['origen'].to_numpy(), self.df_ratios['destino'].to_numpy(), self.df_ratios['periodo'].to_numpy(), self.df_ratios['N_PASAJEROS'].to_numpy(), self.df_ratios['N_pasajeros'].to_numpy()))))

        arrays = np.array_split(np.asarray(array_pares_od_per), PROCESOS_DISPONIBLES)

        # COMIENZA PARALELISMO
        
        pool = multiprocessing.Pool(processes=PROCESOS_DISPONIBLES)
    
        for i in range(PROCESOS_DISPONIBLES):
            pool.apply_async(self.algoritmo_reparto, args=(arrays[i], ), callback=juntar_resultados)
            print(arrays[i], "\n\n")
    
        pool.close()
        pool.join()

        # FINALIZA PARALELISMO

        print('PARALELISMO FINALIZADO')

        # Retornamos los dataframes y los concatenamos hasta que la cola esté vacia

        tiempo_ejecucion.get_tiempo_ejecucion()


    def algoritmo_reparto(self, pares_od):

        df_resultado = pd.DataFrame()

        for par_od in pares_od:
            
            origen = par_od[0]; destino = par_od[1]; hora = par_od[2]
            zonas_colindantes_origen = self.zonas_colindantes.get(origen)
            zonas_colindantes_origen.append(origen)
            zonas_colindantes_destino = self.zonas_colindantes.get(destino)
            zonas_colindantes_destino.append(destino)
            pares_a_tratar = [i for i in itertools.product(*[zonas_colindantes_origen, zonas_colindantes_destino, [hora]])]

            # Se obtienen todas las n-uplas en las que se tiene presente las zonas colindantes del origen y del destino
            df_aux = pd.DataFrame({'origen': list(zip(*pares_a_tratar))[0], 'destino': list(zip(*pares_a_tratar))[1], 'periodo': list(zip(*pares_a_tratar))[2]})
            

            # Se simplifica el nº de n-uplas a unicamente aquellas de los que se tenga registro de TEL
            df = self.df_ratios.merge(df_aux, on=['origen', 'destino', 'periodo'])

            df.drop_duplicates(inplace=True)

            df['origen_cent'] = np.full(len(df), origen)
            df['destino_cent'] = np.full(len(df), destino)

            sumatorio_tp = np.sum(df['N_PASAJEROS'])
            

            """
                Si el Sumatorio de TP es 0, indica que ni en la zona central ni en las zonas
                colindates hay movilidad de TP, por lo que no se pueden hacer proyecciones.
                Sin embargo, son ternas (O,D,H) a estudiar. 
            """ 

            if sumatorio_tp != 0:
            
            
                sumatorio_tel = np.sum(df['N_pasajeros'])
                df['porct_tel'] = df['N_pasajeros'].apply(lambda x: x / sumatorio_tel * 100)
                proyecciones = sumatorio_tp * (df['porct_tel'].to_numpy() /100)
                df['proyecciones_I'] = proyecciones

                proyeccion_separadas = list(map(lambda x: math.modf(x), proyecciones))

                df['enteros'] = list(list(zip(*proyeccion_separadas))[1])
                df['resto'] = list(list(zip(*proyeccion_separadas))[0])

                df = df.sort_values(by='resto', ascending=False)

                reparto_restos = int(sumatorio_tp - np.sum(df['enteros'].to_numpy()))

                valores = np.full(len(df['enteros']), 0)
                valores_h = np.full(reparto_restos, 1)

                df['restos_nv'] = np.concatenate((valores_h, np.full(len(valores)-len(valores_h), 0)), axis=0)
                df['proyecciones_II'] = list(df['enteros'] + df['restos_nv'])
                df_resultado = pd.concat([df_resultado, df])


        return df_resultado