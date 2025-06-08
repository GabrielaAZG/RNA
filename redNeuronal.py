#Importación de las librerías
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

#CLASE DATASET
class Dataset:
    def __init__(self):
        self.df = None #Se inicializa la clase Dataset con un DataFrame vacío

    def leer_CSV(self):
        column_names = ['c1', 'c2', 'c3', 'c4', 'clase']
        self.df = pd.read_csv(r"C:\Users\Gabriela Alva\Downloads\iris\iris.data", names=column_names) #Se lee el archivo CSV, asignandole nombres a las columnas

    def dividirDataset(self):
        setosa = self.df[self.df['clase'] == 'Iris-setosa'].sample(n=25, random_state=1) #Se toman 25 muestras aleatorias de setosa, versicolor y virginica(se eligen aleatoriamente sin repetición)
        versicolor = self.df[self.df['clase'] == 'Iris-versicolor'].sample(n=25, random_state=1)
        virginica = self.df[self.df['clase'] == 'Iris-virginica'].sample(n=25, random_state=1)

        df_entrenamiento = pd.concat([setosa, versicolor, virginica]) #Se juntan las 25 muestras seleccionados aleatoriamente de setosa, versicolor y virginica.
        df_prueba = self.df.drop(df_entrenamiento.index) #Al DataFrame original se le eliminan las muestras elegidas para el entrenamiento (para quedarnos con lo restante que será para las pruebas)

        df_entrenamiento = df_entrenamiento.sample(frac=1, random_state=1).reset_index(drop=True) #Se mezclan los datos tanto del conjunto de pruebas como el de entrenamiento y se reinician los índices.
        df_prueba = df_prueba.sample(frac=1, random_state=1).reset_index(drop=True)

        return df_entrenamiento, df_prueba #Se devuelven los DataFrame de entrenamiento y de prueba.

class RNA:
    def __init__(self, x_train, y_train, no_epocas, no_capas, n, x_test): #Se inicializan los parametros
        self.x_train = x_train
        self.y_train = y_train
        self.no_epocas = no_epocas
        self.no_capas = no_capas
        self.n = n
        self.no_neuronas() #Al crear un objeto de la clase automáticamente se harán los métodos que solicita el número de neuronas y el que inicializa los pesos.
        self.inicializarPesos()
        self.x_test = x_test
        self.y_pred = []

    def no_neuronas(self):
        self.neuronas = []
        self.neuronas.append(len(self.x_train[0])) #Se guarda el número de entradas en la capa de entrada (en este caso, 4 características)
        for i in range(self.no_capas):
            no_neuronas = int(input(f"Número de neuronas capa {i + 1}: ")) #Se le solicita al usuario el número de neuronas que tendrá cada capa oculta
            self.neuronas.append(no_neuronas)
        self.neuronas.append(3) #La capa se salida tendrá 3 (una por cada clase del Dataset)

    def inicializarPesos(self):
        self.lista_W = [] #Lista matricial donde se irán almacenando los pesos.
        for i in range(self.no_capas + 1):
            W = np.random.uniform(-1, 1, size=(self.neuronas[i + 1], self.neuronas[i])) #Se generan pesos aleatorios entre -1 y 1. Las filas = neuronas[i+1] * columnas = neuronas[i]
            self.lista_W.append(W) #Se van agregando las matrices de pesos a la lista

    def sigmoide(self, z):  # Es la activacion
        return 1 / (1 + np.exp(-z)) #Funcion sigmoide

    def perceptron_ml(self, W, X):  # ml es multilayer
        Z = np.dot(W, X) + 1 #Cálculo de la salida de una capa. Producto punto de pesos por la entrada, más un sesgo de +1
        return self.sigmoide(Z) #Se evalúa la pre-activación en la función sigmoide

    def propagacion_adelante(self, X, lista_W):#Ocupa como parámetros el vector de características (valores de entrada de la red) y la lista de matrices de pesos
        self.lista_a = [] #Lista donde se almacenaran las activaciones que se generen en cada capa
        for i in range(self.no_capas + 1): #Se hace el proceso tantas capas haya +1
            a = self.perceptron_ml(lista_W[i], X) #Se invoca al metodo que hace la pre-activación y la activación (se le manda la matriz de pesos perteneciente)
            if np.isscalar(a) or np.array(a).shape == ():  # Si es escalar o sin forma
                self.lista_a.append(np.array([a]))  # Lo convertimos en array([valor])
            else:
                self.lista_a.append(np.array(a)) #De otra forma solo se agregan las activaciones a la lista de activaciones
            X = a #Las activaciones resultantes se vuelven los nuevos valores de entrada
        return X #El último valor de X se devuelve (la salida final)

    def error_cuadratico_medio(self, y_real): #Cálculo del error cuadrático medio entre la salida real y la predicción
        error = 0.5 * np.sum((self.lista_a[-1] - y_real) ** 2)
        print(error)

    def propagacion_atras(self, y_real):
        self.lista_deltas = [] #Se inicializa la lista que almacena los deltas (gardientes del error)
        for i in reversed(range(0, self.no_capas + 1)): #Se hace de la última capa hacia la primera
            if i == self.no_capas:  # Errores en la ultima capa
                eUltimaCapa = (np.array(self.lista_a[i] - y_real)).reshape(-1, 1) #Se calcula la diferencia de las activaciones de la capa actual y la etiqueta real y convierte el resultado a una columna, sin importar la cantidad de filas
                derivadas_sigmoide = ((self.lista_a[i]) * (1 - self.lista_a[i])).reshape(-1, 1) #Se calcula lo que dan las derivadas de la sigmoide y convierte el resultado a una columna, sin importar la cantidad de filas
                error_salida = derivadas_sigmoide * eUltimaCapa #Se hace el producto de HADAMARD con eL y los resultados de la derivada de la sigmoide
                self.lista_deltas.insert(0, error_salida) #Se agregan los deltas de la ultima capa al inicio de la lista
            else:  # Errores en las demas capas
                WT = self.lista_W[i + 1].transpose()  # Matriz transpuesta de la matriz de pesos de la capa siguiente (+1)
                e = np.dot(WT, self.lista_deltas[0]) #Producto punto de la matriz transpuesta y de los deltas de la capa siguiente (+1)
                derivada_sigma = (self.lista_a[i] * (1 - self.lista_a[i])).reshape(-1, 1) #Se calcula lo que dan las derivadas de la sigmoide y se convierte el resultado a una columna, sin importar la cantidad de filas
                delta = derivada_sigma * e #Se hace el producto de HADAMARD con eL y los resultados de la derivada de la sigmoide
                self.lista_deltas.insert(0, delta) #Se agregan los deltas de la capa actual al inicio de la lista

    def actualizacion_pesos(self, x):
        for i in reversed(range(0, self.no_capas + 1)): #Por cada capa se actualizan los pesos
            if i == 0:  # Cuando es la ultima capa se toma el valor de X
                xMatriz = x.reshape(1, -1) #Se transponen las entradas X
                delta_a = np.dot(self.lista_deltas[i], xMatriz) #Se hace el producto punto de los deltas de la capa actual por la entradas X
                deltaW = self.n * delta_a #Se hace la multiplicación por la tasa de aprendizaje
                self.lista_W[i] = self.lista_W[i] - deltaW #Se hace la resta de la matriz de pesos de la capa actual menos el resultado anterior
            else:
                aMatriz = self.lista_a[i - 1].reshape(1, -1) #Se hace uso de las activaciones de la capa anterior (-1) y se transponen (se obtiene una fila con varias columnas)
                delta_a = np.dot(self.lista_deltas[i], aMatriz) #Se hace el producto punto de los deltas de la capa actual por las activaciones
                deltaW = self.n * delta_a  #Se hace la multiplicación por la tasa de aprendizaje
                self.lista_W[i] = self.lista_W[i] - deltaW #Se hace la resta de la matriz de pesos de la capa actual menos el resultado anterior

    def entrenamientoRNA(self):
        for j in range(self.no_epocas): #Se ejecuta por un número de épocas en específico
            for i in range(len(self.x_train)): #Se hace con cada una de las intancias del conjunto de entrenamiento
                self.propagacion_adelante(self.x_train[i], self.lista_W) #Se invoca la propagacion hacia adelante
                self.error_cuadratico_medio(self.y_train[i]) #Se calcula el error cuadrático medio
                self.propagacion_atras(self.y_train[i]) #Se hace la propagación hacia atrás
                self.actualizacion_pesos(self.x_train[i]) #Se actualizan los pesos
        return self.lista_W #Se retornan los últimos pesos

    def pruebasRNA(self, W_rna):
        for i in range(len(self.x_test)): #Se hace por cada instancia del conjunto de prueba
            x = self.propagacion_adelante(self.x_test[i], W_rna) #Se invoca la propagación hacia adelante
            self.y_pred.append(x) #Se almacenan en una lista las etiquetas predichas por la propagación hacia adelante
        return self.y_pred


class Metricas:
    def __init__(self, y_real, y_pred): #Se inicializan los parametros
        self.y_real = y_real
        self.y_pred = y_pred

    def proceso(self):

        matriz = confusion_matrix(self.y_real, self.y_pred) #Se hace el cálculo de las métricas de evaluación
        reporte = classification_report(self.y_real, self.y_pred)

        print("Matriz de confusión:\n", matriz)
        print("Reporte de clasificación:\n", reporte)


#LEER DATASET, REALIZAR LA DIVISION DE ENTRENAMIENTO Y PRUEBA, ESCALAR, Y CAMBIO DE VARIABLES CATEGORICAS A NUMERICAS
datos = Dataset() #Se carga y divide el dataset
datos.leer_CSV()
train, test = datos.dividirDataset()
#print(train)
#print(test)

encoder = LabelEncoder() #Conversion de las variables categoricas a numeros
train['clase'] = encoder.fit_transform(train['clase'])
test['clase'] = encoder.transform(test['clase'])


scaler = MinMaxScaler() #Normalizacion de los datos entre 0 y 1
X_train = scaler.fit_transform(train[['c1', 'c2', 'c3', 'c4']])
X_test = scaler.transform(test[['c1', 'c2', 'c3', 'c4']])

y_train = train['clase'].values #Separacion de las 'y' de las 'x'
y_test = test['clase'].values

onehot = OneHotEncoder(sparse_output=False) #Conversion de las clases (0, 1 y 2) en vectores binarios
y_train_encoded = onehot.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = onehot.transform(y_test.reshape(-1, 1))

#SE CREA OBJETO DE LA CLASE RNA
rna1 = RNA(X_train, y_train_encoded, 500, 2, 0.2, X_test)
pesosResultantes = rna1.entrenamientoRNA() #ENTRENAMIENTO #Devueleve el modelo

y_pred = rna1.pruebasRNA(pesosResultantes) #PRUEBAS
y_pred_classes = np.argmax(y_pred, axis=1) #Se convierten las 'y' predichas y reales a variables numericas
y_real_classes = np.argmax(y_test_encoded, axis=1)

#CALCULO DE LAS METRICAS
metricas_rna = Metricas(y_pred_classes, y_real_classes) #Se manda las 'y' reales del conjunto de pruebas y las 'y' predichas por la red neuronal (propagacion hacia adelante)
metricas_rna.proceso()
























