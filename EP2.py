### EP2 - MAP 2212
### Fernando Chieffi Moraes - NUSP: 11811072

import random
import math
import numpy as np
from scipy.stats import truncexpon
import time


def crude():

    n = 50
    s = 0
    res = []
    for i in range (n):

        x = random.uniform(0,1)

        func = math.exp(-0.326723067 * x) * math.cos(0.36002998837 * x)

        s += func
        res.append(func)


    var_amostra_piloto = np.var(res, ddof=1)

    #calcula o n final
    n_final = math.ceil((((1.96**2) * var_amostra_piloto) /
                         (0.0005**2)*(s/n)**2)) - n

    for i in range (n_final):

        x = random.uniform(0, 1)
        func = math.exp(-0.326723067 * x) * math.cos(0.36002998837 * x)
        s += func
        res.append(func)

    estimativa = s / (n_final + n)
    var_amostra_final = np.var(res, ddof=1)
    dp = var_amostra_final**(1/2)
    return n_final, estimativa, dp

### Método 2 - Monte Carlo Hit or Miss
def hitmiss():

    n = 50 #n para amostra piloto
    s = 0 #inicia variável indicadora
    res = [] #guarda os resultados da função

    for i in range(n): #calcula amostra piloto

        x = random.uniform(0,1) #gera pontos da uniforme no intervalo [0,1]
        y  = random.uniform(0,1) #gera pontos da uniforme no intervalo [0,1]

        # função que queremos calcular a integral no intervalo [0,1]
        func = math.exp(-0.326723067 * x) * math.cos(0.36002998837 * y)

        if y <= func: #verifica se o ponto caiu abaixo da curva
            s += 1
        res(func)

    proporcao_piloto = s/n
    # calcula variância da amostra piloto
    var_amostra_piloto = proporcao_piloto * (1 - proporcao_piloto)

    #calcula o n final
    n_final = math.ceil(((1.96**2) * var_amostra_piloto) /
                        (0.0005**2)*(proporcao_piloto**2)) - n

    for i in range(n_final):  #calcula amostra final

        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        func = math.exp(-0.326723067 * x) * math.cos(0.36002998837 * x)

        if y <= func:
            s += 1
        res(func)

    estimativa = s / (n_final + n) #calcula a estimativa da integral
    var_amostra_final = estimativa * (1 - estimativa)
    dp = var_amostra_final**(1/2) #calcula o desvio padrão

    return n_final, estimativa, dp

### Método 3 Monte Carlo Importance Sampling
def imp_sampling ():

    n = 50 #n para amostra piloto
    s = 0 #guarda a s dos resultados da função
    res = [] #guarda os resultados da função

    for i in range (n): #calcula amostra piloto

        #gera pontos de uma exponencial truncada no intervalo [0,1] com lambda = 0.RG
        x = truncexpon.rvs(0.326723067, loc=0, scale = 1/0.326723067)

        #resultado da divisão da função que queremos calcular pela PDF da exponencial truncada
        func = (math.cos(0.36002998837 * x) * (1 - math.exp(-0.326723067))) / 0.326723067
        s += func
        res(func)

    #calcula variância da amostra piloto
    var_amostra_piloto = np.var(res, ddof=1)

    #calcula o n final
    n_final = math.ceil((((1.96 ** 2) * var_amostra_piloto) /
                         (0.0005 ** 2)*(s/n)**2)) - n

    for i in range(n_final): #calcula amostra final

        x = truncexpon.rvs(0.326723067, loc=0, scale = 1/0.326723067)
        func = (math.cos(0.36002998837 * x) * (1 - math.exp(-0.326723067))) / 0.326723067
        res(func)
        s += func

    estimativa = s / (n_final + n) #calcula estimativa da integral
    var_amostra_final = np.var(res, ddof=1)
    dp = var_amostra_final ** (1 / 2) #calcula desvio padrão

    return n_final, estimativa, dp

### Método 4 Monte Carlo Control Variate
def control_variate():

    n = 50 #n da amostra piloto
    sg = 0 #guarda a s dos valores avaliados na função g(x) (a integral que queremos calcular)
    sf = 0 #guarda a s dos valores avaliados na função f(x) = -x/4 +1 (a variável de controle)
    listag = [] #guarda os resultados da função g
    listaf = [] #guarda os resultados da função f

    #valor da integral de f(x) = -x/4 +1 avaliada no intervalo [0,1]
    integral_f = 0.875

    for i in range (n): #calcula amostra piloto

        x = random.uniform(0,1) #gera pontos de uma uniforme no interval0 [0,1]

        #avalia a função que queremos calcular
        g = np.exp(-0.326723067 * x) * np.cos(0.36002998837 * x)
        sg += g
        listag.append(g)

        #avalia a função de controle
        f = (-x) / 4 + 1
        sf += f
        listaf.append(f)

    #calcula a constante c
    c = -np.cov(listaf, listag)[0, 1] / np.var(listaf)

    #calcula estimativa piloto
    estimativa_piloto = np.mean(listag) + c * (np.mean(listaf) - integral_f)

    #calcula a variância da amostra piloto
    variancia_piloto = (np.var(listaf) + np.var(listag) -
                        2 * (np.corrcoef(listaf, listag)[0, 1]) *
                        (math.sqrt(np.var(listaf))) * (math.sqrt(np.var(listag)))) / n

    #calcula o n final
    n_final = math.ceil((((1.96 ** 2) * variancia_piloto) /
                         (0.0005 ** 2)*(estimativa_piloto**2)))

    for i in range(n_final): #calcula amostra final

        x = random.uniform(0,1)

        g = np.exp(-0.326723067 * x) * np.cos(0.36002998837 * x)
        sg += g
        listag.append(g)

        f = -x/4 + 1
        sf += f
        listaf.append(f)

    c = -np.cov(listaf, listag)[0, 1] / np.var(listaf)

    #calcula a estimativa da integral
    estimativa = np.mean(listag) + c * (np.mean(listaf) - integral_f)

    var_amostra_final = (np.var(listag) + np.var(listaf) -
                        2 * (np.corrcoef(listag, listaf)[0, 1]) *
                        (math.sqrt(np.var(listag))) * (math.sqrt(np.var(listaf)))) / n_final

    dp = var_amostra_final**(1/2) #calcula o desvio padrão

    return n_final, estimativa, dp


def main (): #imprime os resultados e conclusão

    np.random.seed(27) #fixa seed da função da biblioteca do Scipy
    random.seed(27) #fixa seed das funções da bibliotecam Random

    print("MÉTODO 1 - MONTE CARLO CRUDE\n")
    t0 = time.time()  # calcula o tempo de execução
    metodo1 = crude()
    t1 = time.time()
    print("Tamanho final da amostra = ", metodo1[0])
    print("Valor da integral = ", metodo1[1])
    print("Desvio Padrão = ", metodo1[2])
    print("Tempo de execução =", t1 - t0)

    print("\nMÉTODO 2 - MONTE CARLO HIT OR MISS\n")
    t0 = time.time()
    metodo2 = hitmiss()
    t1 = time.time()
    print("Tamanho final da amostra = ", metodo2[0])
    print("Valor da integral = ", metodo2[1])
    print("Desvio Padrão = ", metodo2[2])
    print("Tempo de execução =", t1 - t0)

    print("\nMÉTODO 3 - MONTE CARLO IMPORTANCE SAMPLING\n")
    t0 = time.time()
    metodo3 = imp_sampling()
    t1 = time.time()
    print("Tamanho final da amostra = ", metodo3[0])
    print("Valor da integral = ", metodo3[1])
    print("Desvio Padrão = ", metodo3[2])
    print("Tempo de execução =", t1 - t0)

    print("\nMÉTODO 4 - MONTE CARLO CONTROL VARIATE\n")
    t0 = time.time()
    metodo4 = control_variate()
    t1 = time.time()
    print("Tamanho final da amostra = ", metodo4[0])
    print("Valor da integral = ", metodo4[1])
    print("Desvio Padrão = ", metodo4[2])
    print("Tempo de execução =", t1 - t0)

    print("\nComparação entre os métodos pela razão dos Desvios Padrões")

    # Crude X outros métodos
    print("\nCrude X Hit or Miss")
    print("DP_Crude / DP_Hit_Miss =", metodo1[2] / metodo2[2])

    print("\nCrude X Importance Sampling")
    print("DP_Crude / DP_Importance_Sampling =", metodo1[2] / metodo3[2])

    print("\nCrude X Control Variate")
    print("DP_Crude / DP_Control_Variate =", metodo1[2] / metodo4[2])

    print("\nHit or Miss X Importance Sampling")
    print("DP_Hit_Miss / DP_Importance_Samplig =", metodo2[2] / metodo3[2])

    print("\nHit or Miss X Control Variate")
    print("DP_Hit_Miss / DP_Control_Variate =", metodo2[2] / metodo4[2])

    print("\nImportance Sampling X Control Variate")
    print("DP_Importance_Sampling / DP_Control_Variate =", metodo3[2] / metodo4[2])

    print("\nConclusão\n")
    print("Os melhores métodos por ordem de eficiência são:\n")
    print("1) Control Variate")
    print("2) Importance Sampling")
    print("3) Crude")
    print("4) Hit or Miss")


main()