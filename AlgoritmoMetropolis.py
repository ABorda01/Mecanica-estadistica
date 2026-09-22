"""
-----------------------------------------------------
-----------------------------------------------------
-----------------------------------------------------
Componente computacional tarea 3 mecanica estadistica
Alejandro Borda Kuhlmann - 202020727
-----------------------------------------------------
-----------------------------------------------------
-----------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Definimos las constantes universales y trabajamos en unidades de
J = K_B = 1 tal que Tc = 1/ ln(1+sqrt(2)) por el rango de 2.2.
Estas son constantes universales del programa, particularmente: 
N, J y steps (tiempo discreto del algoritmo)
"""

N = 30 # Red cuadrada de N x N
J = 1
k_B = 1
steps = 50000

"""
Definimos las funciones de ayuda:
    
    Hamiltonian: Toma una configuracion de espines (matriz de -1's y 1's) y calcula la energia
        iterando sobre todos los vecinos mas cercanos. Se toma % N para considerar las condiciones
        de fronteras periodicas.

    LocalEnergy: Energia local del Hamiltoniano en los vecinos mas cercanos a un sitio dado,
        se utiliza para calcular el cambio local en el ajuste del algoritmo y no iterar inecesariamente
        sobre todos los sitios cuando el cambio es de solo un sitio.
"""

def Hamiltonian(S):

    sm = 0
    for k in range(0, N): #Indice Fila
        for j in range(0, N): #Indice Columna
            sm += S[k, j] * S[k, (j+1) % N]
            sm += S[k, j] * S[(k+1) % N, j]
        
    return -J * sm

def localEnergy(S, i, j):
    # Suma de las interacciones del espin (i,j) con sus 4 vecinos
    neighbors = S[(i+1) % N, j] + S[(i-1) % N, j] + S[i, (j+1) % N] + S[i, (j-1) % N]
    return -J * S[i, j] * neighbors

"""
    Ising2D(T): Toma una temperatura fija e implementa el algoritmo de metropoli durante "steps".
    La configuracion inicial es fija, para reducir el ruido encontrado en la ultima parte de la
    implementacion. Se calcula el cambio de energia local para reducir la complejidad del algoritmo
    y este cambio se usa para implementar el criterio de Metropolis por medio de una distribucion
    uniforme. En cada paso se guarda el valor de energia y magnetizacion en valor absoluto.

    Durante todas las iteraciones de Ising2D(T) se usa la misma configuracion inicial de espines
    "S_0", con el objetivo de reducir el ruido entre iteraciones e investigar unicamente los efectos
    provenientes de las diferentes temperaturas y no del cambio en condiciones iniciales.
"""

# Configuracion inicial de espines aleatoria 2D, N x N
S_0 = np.random.choice([-1, 1], size=(N, N))

def ising2D(T):
    beta = 1/T
    S = S_0.copy()
    
    # Iteracion del algoritmo de Metropolis
    es = []
    ms = []
    energy = Hamiltonian(S)
    for k in range(steps):

        es.append(energy)

        magnetization = np.sum(S) / N**2
        ms.append(np.abs(magnetization))

        i, j = np.random.randint(0, N), np.random.randint(0, N)

        EChange = -2 * localEnergy(S, i, j) # Calculamos el cambio local unicamente por eficiencia

        pr = min(1, np.exp(-beta * EChange))

        if np.random.rand() < pr: # Implementamos el cambio con probabilidad pr por medio de la distribucion unif(0,1)
            S[i, j] *= -1
            energy += EChange

    return es, ms



# Punto 1-2: Ejemplo de ising2D con T=0.1

T= 0.8
es_t, ms_t = ising2D(T)

# Graficas:

x = np.arange(len(es_t))
es_arr = np.array(es_t)
es_norm = es_arr / N**2
plt.plot(x, es_norm)

plt.title(f'Energia normalizada por sitio vs tiempo (discreto) a T = {T:.2f}')
plt.xlabel('Paso (k)')
plt.ylabel('Energia normalizada')
plt.axhline(np.mean(es_norm[-2000:]), color='orange')
plt.grid(alpha=0.3)

print(f"3: Valor de magnetizacion y energia promedio obtenidos en las configuraciones estables de: {np.mean(ms_t[-2000:])} y {np.mean(es_norm[-2000:])} respectivamente")

plt.show()


# Puntos 3-5

"""
Dado que T_c en las unidades trabajadas esta por el rango de 2-3, iteramos sobre un rango 
considerable de temperaturas por encima y por debajo de la temperatura critica.
"""

Ts = np.linspace(0.5, 20, num=100)

M = []
E = []
for t in Ts:

    es_t, ms_t = ising2D(t)
    es_arr = np.array(es_t)
    es_norm = es_arr / N**2
    M.append(np.mean(ms_t[-1000:]))
    E.append(np.mean(es_norm[-1000:]))

plt.plot(Ts, M)

Tc = 2  / (np.log(1 + np.sqrt(2))) 
plt.axvline(Tc, linestyle='--', color='orange', label=f'$T_c$ = {Tc:.3f}')
plt.legend()

plt.title(f'Promedio de la magnetizacion del sistema estable vs temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Valor absoluto de Magnetizacion promedio')
plt.grid(alpha=0.3)

plt.show()