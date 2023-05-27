import math

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def calcula_n(x,y):
    tn=np.array([])
    for i in range(100):
        tn=np.append(tn,sum(calcula_func(x,y,100)[0:1]/2))
    var=np.var(tn,ddof=1)
    return(math.ceil((1.96 ** 2) * var / (0.0005 ** 2)))
def calcula_func(x,y,n):
    v=x+y

    theta=np.random.dirichlet(v,n)

    prod=np.array([])
    for i in range(len(theta)):
        prod= np.append(prod, np.prod(np.power(theta[i], v - 1)))

    c=1/((np.prod(gamma(v))) / (gamma(sum(v))))

    return(np.sort(prod*c))

def estima_W(f, v):
    # determina quantos pontos estão abaixo de um determinado v
    n = np.searchsorted(f, v=v)

    # tamanho do vetor que guarda os f_thetas ordenados
    N = len(f)

    # retorna a estimativa
    return n / N



if __name__ == "__main__":
    while True:  # loop principal do programa que executa até o usuário sair

        x = np.array([])
        y = np.array([])

        # solicita entra do vetor x
        for i in range(3):
            x = np.append(x, int(input(f"Digite o valor x{i} do vetor x: ")))

        # solicita entra do vetor y
        for i in range(3):
            y = np.append(y, int(input(f"Digite o valor y{i} do vetor y: ")))

        ntheta= calcula_n(x, y)

        print("Número de thetas que serão gerados: ", ntheta)

        # calcula a f_theta e ordena os resultados
        fs_ordenados = calcula_func(x, y, ntheta)

        # função que retorna a estimativa da função W(v)
        U = lambda v: estima_W(fs_ordenados, v)
        a=np.array([0])
        n=np.array([0])
        for i in range(0,2*(int(fs_ordenados[-1]+3))):
            a=np.append(a,U(i/2))
            n=np.append(n,i/2)
        plt.plot(n,a)
        plt.show()
        # loop secundário que executa até o usuário decidir trocar os vetores x e y
        while True:

            # solicita um valor de v entre 0 e o sup(f_theta)
            v = float(input(f"\nEntre com o valor de v de 0 a {fs_ordenados[-1]}: "))

            # retorna a estimativa na tela
            print(f"U({v})= {U(v)}")

            print("\nQuer calcular uma nova estimativa (s/n)")

            sn_v = input().lower()

            if sn_v == 'n': break

            print()

        print('Quer mudar os vetores utilizados? (s/n)')

        sn = input().lower()

        if sn == 'n': break

        print()
