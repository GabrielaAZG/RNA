El archivo CSV se carga desde una ruta específica en tu computadora. Antes de ejecutar el código, cambia la ruta del archivo (línea 14) por la ruta donde tú tengas el archivo iris.data guardado. Por ejemplo:

self.df = pd.read_csv(r"D:\Mis archivos\iris.data", names=column_names)

