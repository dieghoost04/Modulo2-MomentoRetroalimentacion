import numpy as np  
import os
import matplotlib.pyplot as plt  

# Variable en la que se almacenaran los errores
__errors__ = []

# Dataset que contiene dos columnas [ YearsExperience, Salary ]
archivo_csv = 'Salary_dataset.csv'
samples_g = []
y = []

# Almacenamos las columnas en dos listas
with open(archivo_csv, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  
        columns = line.strip().split(',')
        sample_row = [float(col) for col in columns[1:-1]] 
        samples_g.append(sample_row)  
        y.append(float(columns[-1]))  

samples = samples_g.copy()

print("\nSamples Originales: ",samples)

def init_ps(samples):
    '''
    Función que inicializa los samples, y parámetros.
    '''

    # Determina cuantas características se estaran evaluando, y crea una matriz con un '1' antes de cada característica
    for i in range(len(samples)):
        if isinstance(samples[i], list):
            samples[i]=  [1]+samples[i]
        else:
            samples[i]=  [1,samples[i]]

    params = np.zeros_like(samples[0])

    # Cuenta el número de características que hay en samples, y inicia una parametro aleatorio para cada una   
    for i in range(len(samples[0])):
        params[i] = np.random.uniform(0,10)
    return params, samples

params, samples = init_ps(samples)

print("Parametros de la funcion: ",params)
print("Samples con la funcion: ",samples)


def GDD(params, samples, y, alpha = 0.01):
    '''
    Función de Gradient Descent.
    '''
    # Convertimos a np.array
    samples = np.array(samples)
    params = np.array(params)
    y = np.array(y)
    aux = params.copy()

    # Obtenemos una matriz con la sumatoria (h(xi) - y)*xi. Cada fila corresponde a un parámetro diferente.
    error = (params * samples)
    error = error.sum(axis = 1)
    error = error - y
    acum = error * samples.T 
    acum = acum.sum(axis = 1)
    
    # Actualización del parámetro.
    aux = params - alpha*(1/len(samples))*acum
    print("Parametros viejos: ",params)
    print("Parametros nuevos: ",aux)
    return aux


def errors(params, samples, y):
    '''
    Función que calcula los errores.
    '''
    # Convertimos a np.array
    samples = np.array(samples)
    params = np.array(params)
    y = np.array(y)
    
    # Caculamos el error 1/2m * sum(pred - y)**2
    error = (params * samples)
    error = error.sum(axis = 1)   
    error = error - y
    error = error ** 2
    error = error.sum()/2*len(samples)
    
    print("\nError: ", error)
    __errors__.append(error)
    return error

epochs = 0

# Realizamos la actualización de parámetros por 15 epochs.
while epochs < 15:
    print("\nEpoch: ", (epochs+1))
    errors(params, samples, y)
    params = GDD(params, samples, y)
    epochs += 1

# Calculamos las predicciones con los parámetros actualizados.
h = params * samples
h = h.sum(axis = 1)

# Graficamos el error y tambien nuesto modelo de regresión contra los datos reales.
plt.figure()
plt.plot(__errors__, color = 'b')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error')

plt.figure()
plt.scatter(samples_g, y, marker='o', linestyle='-', color='b')
plt.scatter(samples_g, h, marker='o', linestyle='-', color='r')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Model vs Real Data')
plt.show()