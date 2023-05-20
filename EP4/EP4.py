import numpy as np
from scipy.special import gamma


def calcula_n_final(x, y, seed=42):  # calcula número de thetas que serão gerados

    n_ini = 1000  # número inicial de thetas para amostra piloto

    v1_s = []  # lista que vai guardar a média entre os primeiros dois f(theta)

    for i in range(100):  # realiza 100 repetições do experimento

        fs = calcula_f(x, y, n_ini, i)  # calcula a f_theta

        # tira a média entre os dois primeiros f_theta
        # que já foram ordenados na função calcula_f
        v1_s.append((fs[0] + fs[1]) / 2)

    vr = np.var(v1_s, ddof=1)  # calcula a variância da média dos dois primeiros f_thetas

    # calcula o n_final
    n_final = int((1.96 ** 2) * vr / (0.0005 ** 2))

    return n_final


def calcula_f(x, y, n, seed=42):
    np.random.seed(seed=seed)

    a = x + y  # soma os vetores de entrada x e y

    thetas = np.random.dirichlet(a, n)  # gera thetas de uma dirichelt

    produtorio = np.array([])

    # calcula o produtório da f_theta
    for i in range(len(thetas)):
        produtorio = np.append(produtorio, np.prod(np.power(thetas[i], a - 1)))

    # calcula a constante de normalização
    c = 1 / ((np.prod(gamma(a))) / (gamma(sum(a))))

    # multiplica a constante pela resultado do produtório
    f_thetas = c * produtorio

    # retorna os f_thetas ordenados
    return np.sort(f_thetas)


def estima_W(f_ord, v):
    # determina quantos pontos estão abaixo de um determinado v
    n = np.searchsorted(f_ord, v=v)

    # tamanho do vetor que guarda os f_thetas ordenados
    N = len(f_ord)

    # retorna a estimativa
    return n / N


if __name__ == "__main__":
    while True:  # loop principal do programa que executa até o usuário sair

        x = np.array([])
        y = np.array([])

        # solicita entra do vetor x
        for i in range(3):
            x = np.append(x, int(input(f"Entre com x{i}: ")))

        # solicita entra do vetor y
        for i in range(3):
            y = np.append(y, int(input(f"Entre com y{i}: ")))


        n_de_thetas = calcula_n_final(x, y)

        print("Número de thetas que serão gerados: ", n_de_thetas)

        # calcula a f_theta e ordena os resultados
        fs_ordenados = calcula_f(x, y, n_de_thetas)

        # função que retorna a estimativa da função W(v)
        U = lambda v: estima_W(fs_ordenados, v)

        # loop secundário que executa até o usuário decidir trocar os vetores x e y
        while True:

            # solicita um valor de v entre 0 e o sup(f_theta)
            v = float(input(f"\nEntre com o valor de v de 0 a {fs_ordenados[-1]}: "))

            # retorna a estimativa na tela
            print(f"U({v})= {U(v)}")

            print("\nGostaria de calcular uma nova estimativa da U(v)? (s/n)")

            sn_v = input().lower()

            if sn_v == 'n': break

            print()

        print('Gostaria de inserir novos vetores de entrada para gerar uma nova U(v)? (s/n)')

        sn = input().lower()

        if sn == 'n': break

        print()
