### MAP2212 - EP04
### Afrânio Maia da Silva   NUSP 6369110
### Felipe Soares Appolari  NUSP 12556607
### Fernando Chieffi Moraes NUSP 11811072
import numba
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet
from scipy.special import gamma
import math
import time
from numba import jit, njit, vectorize

#  Workstorage
ws = {
    "vec_x": [4, 6, 4],
    "vec_y": [1, 2, 3],
    "alpha": None,
    "theta_size": 1000,
    "simulation_size": 100,
    "sample_size": None,
    "first_vlevel": None,
    "last_vlevel": None,
    "f_theta": None,
    "seed": None,
    "u_size":None,
}


def theta_dirichlet(alpha, n=None):
    """Gera no mínimo 3 amostras de theta da distribuição Dirichlet.
    """
    nsize = n if n > 3 else 3
    return np.random.dirichlet(alpha, nsize)

@njit(nogil=True)
def dir_pdf(x, a):
    if ((x < 0).any()): return 0.0
    t=np.empty_like(x)
    for i in range(len(x)):
        t[i]=np.power(x[i],(a[i]-1))
    return np.prod(t)

@njit(nogil=True)
def cria_cov(alfa):
    M = np.array([[0.0, 0.0],[0.0,0.0]])

    for i in range(2):
        for j in range(2):

            if i == j:
                M[i, j] = alfa[i] * (sum(alfa) - alfa[i]) / \
                          ((sum(alfa) ** 2) * (sum(alfa) + 1))  # cálculo das variâncias
            else:
                M[i, j] = -alfa[i] * alfa[j] / \
                      ((sum(alfa) ** 2) * (sum(alfa) + 1))  # cálculo das covariâncias

    return M

@njit(nogil=True)
def met_ac(pontos,p,b,alfa):
    for i in range(1,len(pontos)):
        ponto_atual = np.array \
            ([pontos[i-1][0] + p[i,0], pontos[i-1][1] + p[i,1], 1.0 - (pontos[i-1][0] + p[i,0] + pontos[i-1][1] + p[i,1])])
        # algoritmo de aceitação de Metropolis
        ac = min(1, dir_pdf(ponto_atual, alfa) /
                 dir_pdf(pontos[i-1], alfa))
        if ac >= b[i]:
            pontos[i] = ponto_atual
        else:
            pontos[i] = pontos[i-1]

    return pontos


def gera_dir(alfa=[5, 5, 5], n=100, burnin=1000):
    pontos = np.array([[1/3,1/3,1/3] for _ in range(burnin+n)])  # ponto inicial da cadeia de Markov
    k = 0  # contador de pontos aceitos
    # multiplica matriz de covariância por constante ótima
    M=cria_cov(alfa)* (2.38 ** 2) / 2
    p = np.random.multivariate_normal([0, 0], M,size=n+burnin)  # gera da Normal Multivariada
    b= np.random.uniform(0,1,size=n+burnin)
    dir=met_ac(pontos,p,b,alfa)
    return dir[burnin:]


def generate_sample(alpha, n=None):
    """Calcula a densidade de probabilidade Dirichlet dos thetas ordenadas.
    """
    nsize = n or 1

    c = 1 / ((np.prod(gamma(alpha))) / (gamma(sum(alpha))))

    return np.sort([dir_pdf(theta,alpha)*c for theta in gera_dir(alpha, nsize)])

def sample_simulation(alpha, a=None, n=0):
    """Gera uma quantidade de amostras de densidade de probabilidade da distribuição Dirichlet.
    """
    amount = a or 1

    return [generate_sample(alpha, n) for _ in range(amount)]


def get_sample_size(data_simulation):
    """Calcula um tamanho estatístico ideal para uma quantidade de amostra com intervalo de
    confiança 95% e um erro 0.05%.
    """
    ERROR = 0.0005
    CI = 1.96
    sample_data = [np.mean(single_sample) for single_sample in data_simulation]
    variance = np.var(sample_data,ddof=1)
    #  Atualiza a área de Workstorage
    ws['sample_size'] = math.floor((CI ** 2 * variance) / ERROR ** 2)

def f_theta_density():
    """Retorna os valores finais ordenados da densidade de probabilidade Dirichlet
    dos thetas com o tamanho para a precisão esperada.
    """
    theta = sample_simulation(ws["alpha"], n=ws["sample_size"])[0]
    #  Atualiza a área de Workstorage
    ws["f_theta"] = theta
    ws["first_vlevel"] = theta[0]
    ws["last_vlevel"] = theta[-1]
    ws["u_size"]=len(np.unique(theta))


def u_estimate(v_level):
    """Retorna o valor aproximado da integral no nível pela proporção de pontos simulados.
    """

    return np.searchsorted(ws["f_theta"], v_level) / ws["sample_size"]

def weight_bin(k):
    """Calcula o peso de um corte.
    """
    return u_estimate(np.unique(ws["f_theta"][k])) - u_estimate(np.unique(ws["f_theta"][k-1]))


def simulation():
    """Realiza a simulação com os parâmetros do usuário.
    """
    #  Atualiza a área de Workstorage
    ws["alpha"] = np.array(ws["vec_x"]) + np.array(ws["vec_y"])

    data_simulation = sample_simulation(ws["alpha"],
                                        a=ws["simulation_size"],
                                        n=ws["theta_size"])

    get_sample_size(data_simulation)
    f_theta_density()

def display_prompt_vectors():
    """ Apresenta ao usuário a interação com os valores dos Vetores X e Y
    """
    print("\nValores definidos dos vetores X e Y")
    print("---"*15)
    print(f"{'  Vetor de observação X:':<34}{ws['vec_x']}")
    print(f"{'  Vetor de informação à priori Y:':<34}{ws['vec_y']}\n")

    prompt = ""
    while prompt.upper() not in ("S", "N"):
        prompt = input("[?] Gostaria de alterar estes valores? (S/N) \n    >>> ")

    if prompt.upper() == "S":
        #  Recebe valores o Vetor X
        while True:
            try:
                MSG = "\n[+] Informe os 3 valores do Vetor X separados por vírgulas \n    >>> "
                input_vecx = [int(i) for i in input(MSG).split(",")]
                if len(input_vecx) != 3: raise
                if min(input_vecx) < 0: raise
                #  Atualiza a área de Workstorage
                ws["vec_x"] = input_vecx
                break
            except:
                print("[!] Valores incorretos. Tente novamente.")

        #  Recebe valores o Vetor Y
        while True:
            try:
                MSG = "\n[+] Informe os 3 valores do Vetor Y separados por vírgulas \n    >>> "
                input_vecy = [int(i) for i in input(MSG).split(",")]
                if len(input_vecy) != 3: raise
                if min(input_vecy) < 0: raise
                #  Atualiza a área de Workstorage
                ws["vec_y"] = input_vecy
                break
            except:
                 print("[!] Valores incorretos. Tente novamente.")


def display_prompt_sizes():
    """ Apresenta ao usuário a interação dos valores dos tamanhos amostrais iniciais
    """
    print("\nValores definidos para amostras de Dirichlet")
    print("---"*18)
    print(f"{'  Quantidade de amostras iniciais: ':<50}{ws['simulation_size']}")
    print(f"{'  Quantidade de parâmetros THETA em cada amostra: ':<50}{ws['theta_size']}\n")

    prompt = ""
    while prompt.upper() not in ("S", "N"):
        prompt = input("[?] Gostaria de alterar estes valores? (S/N) \n    >>> ")

    if prompt.upper() == "S":
        #  Recebe valores da quantidade de amostras iniciais
        while True:
            try:
                MSG = "\n[+] Informe a quantidade de amostras iniciais desejada \n    >>> "
                input_simulation_size = int(input(MSG))
                if input_simulation_size < 0: raise
                #  Atualiza a área de Workstorage
                ws["simulation_size"] = input_simulation_size
                break
            except:
                print("[!] Valor incorreto. Tente novamente.")

        #  Recebe valores da quantidade de parâmetros THETA em cada amostra
        while True:
            try:
                MSG = "\n[+] Informe a quantidade de THETAs desejado em cada amostra (mínimo 3) \n    >>> "
                input_theta_size = int(input(MSG))
                if input_theta_size < 3: raise
                #  Atualiza a área de Workstorage
                ws["theta_size"] = input_theta_size
                break
            except:
                print("[!] Valor incorreto. Tente novamente.")


def display_prompt_confirm():
    """ Apresenta ao usuário a interação dos parâmetros do processamento inicial.
    """
    seed_value = "Vazio" if ws["seed"] is None else ws["seed"]
    print("\nConfirmação dos valores para execução")
    print("---"*16)
    print(f"{'  Valor do seed:':<39}{seed_value}")
    print(f"{'  Vetor de observação X:':<39}{ws['vec_x']}")
    print(f"{'  Vetor de informação à priori Y:':<39}{ws['vec_y']}")
    print(f"{'  Quantidade de amostras iniciais: ':<39}{ws['simulation_size']}")
    print(f"{'  Quantidade de THETA em cada amostra: ':<39}{ws['theta_size']}\n")


def display_prompt_u():
    """ Apresenta ao usuário a interação para o cálculo da estimativa da integral.
    """
    #  Recebe o nível de corte para a estimativa da integral
    while True:
        try:
            MSG = ("\n[+] Informe o nível de corte para U(v).\n    Mínimo THETA = "
                   f"{ws['first_vlevel']:.10f}\n    Máximo THETA = {ws['last_vlevel']:.10f} \n    >>> ")
            input_v = float(input(MSG))

            u = u_estimate(input_v)
            print(f"    U({input_v}) = {u:.10f}   (~{u*100:.2f}%)")

            prompt = ""
            while prompt.upper() not in ("S", "N"):
                prompt = input("\n[?] Gostaria de realizar outro corte em U(v)? (S/N) \n    >>> ")
            if prompt.upper() == "N": break

        except:
            print("[!] Valor incorreto. Tente novamente.")


def display_prompt_bin():
    """ Apresenta ao usuário a interação para o cálculo das seleções dos bins
    """
    #  Recebe o nível de corte para a estimativa da integral
    while True:
        try:
            MSG = f"\n[+] Informe um bin entre 1 e {ws['u_size']-1} para consultar o peso\n    >>> "
            input_bin = int(input(MSG))
            if input_bin < 1 or input_bin > ws['u_size']-1: raise
            bins=np.unique(ws["f_theta"])
            wbin = weight_bin(input_bin)[0]
            len_bin = ws["sample_size"]
            top_bin = bins[input_bin]
            bottom_bin = bins[input_bin-1]

            print((f"{f'    ==> O bin {input_bin} possui extremidades ':<60}"
                   f" = ({bins[input_bin-1]:.6f}, {bins[input_bin]:.6f})"))
            print((f"{f'    ==> U({top_bin:.6f}) - U({bottom_bin:.6f})   ':<60} = {wbin:.15f}"))
            print((f"{f'    ==> W({top_bin:.6f}) - W({bottom_bin:.6f}) ~ 1/k (k = {len_bin})   ':<60} = {1/len_bin:.15f}"))
            print((f"{f'    ==> Erro relativo da U() em relação à W()':<60} = {abs(wbin - (1/len_bin))/(1/len_bin)*100:.5f}%"))

            prompt = ""
            while prompt.upper() not in ("S", "N"):
                prompt = input("\n[?] Gostaria de selecionar outro bin? (S/N) \n    >>> ")
            if prompt.upper() == "N": break

        except:
            print("[!] Valor incorreto. Tente novamente.")


def display_prompt_seed():
    """ Apresenta ao usuário a interação de condicionar um estado aleatório na geração dos dados
    """
    seed_value = "vazio" if ws["seed"] is None else ws["seed"]

    #  Recebe um valor para o seed
    while True:
        try:
            MSG = f"\n[+] Informe o valor do Seed ou pressione [ENTER] para manter o valor {seed_value}.\n    >>> "
            input_seed = input(MSG)
            if len(input_seed) != 0:
                ws["seed"] = int(input_seed)

            break

        except:
            print("[!] Valor incorreto. Tente novamente.")


def save_graph():
    '''Salva o gráfico de U(v) no diretório corrente.
    '''

    ## Dados
    x = np.linspace(-4, ws["f_theta"][-1]+4, num=2000)
    y = u_estimate(x)

    ## Grid
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title("Função de densidade U(v)")
    plt.ylabel('Probabilidade acumulada')
    plt.xlabel('Nível v')

    ## Anotações dos valores de V
    v_pmax = list(y).index(0)
    ax.annotate('v = 0.0', fontsize=10,
                xy=(0, 0), xycoords='data',
                xytext=(90, 10), textcoords='offset points',
                arrowprops=dict(facecolor='darkblue', shrink=0.05),
                horizontalalignment='right', verticalalignment='bottom')
    ax.annotate(f'v = {x[v_pmax]:.3f}', fontsize=10,
                xy=(x[v_pmax], 1), xycoords='data',
                xytext=(0, -50), textcoords='offset points',
                arrowprops=dict(facecolor='darkblue', shrink=0.05),
                horizontalalignment='left', verticalalignment='bottom')

    ## Box com os parâmetros
    txt_seed = f"\n{f'Valor do seed:':<20} {ws['seed']}" if ws['seed'] is not None else ""
    textstr = ''.join((
        "Valores do Processamento:\n",
        "-"*30+"\n",
        f"{f'Vetor X:':<20} {ws['vec_x']}\n",
        f"{f'Vetor Y:':<20} {ws['vec_y']}\n",
        f"{f'Qnt. amostras:':<20} {ws['simulation_size']}\n",
        f"{f'Qnt. θ por amostra:':<20} {ws['theta_size']}\n",
        f"{f'Qnt. simulação:':<20} {ws['sample_size']}")) + txt_seed
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='top', bbox=props)

    ## Salva arquivo PNG
    plt.plot(x, y)
    filename = f'u_graph_{time.strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(filename)
    plt.cla()
    plt.clf()

    return filename


def main():
    """Orquestra o processamento e a interação com o usuário.
    """
    print("EP04 - ESTIMAR A MASSA DE UM NÍVEL DE CORTE PARA UMA DISTRIBUIÇÃO DIRICHLET")

    while True:
        display_prompt_seed()
        display_prompt_vectors()
        display_prompt_sizes()
        display_prompt_confirm()
        prompt = ""
        while prompt.upper() not in ("S", "N"):
            prompt = input("[?] Gostaria de alterar estes valores? (S/N) \n    >>> ")

        if prompt.upper() == "N":
            np.random.seed(ws["seed"])

            start_time = time.perf_counter()
            simulation()
            end_time = time.perf_counter()
            print(f"\n ==> Processamento da simulação finalizado em {(end_time - start_time):.3f} segundos")

            if ws["sample_size"] <= 0:
                print((f" ==> Não foi possível encontrar um número de amostras "
                       "para a simulação. Ajuste os parâmetros."))
                continue

            prompt = ""
            while prompt.upper() not in ("S", "N"):
                prompt = input("\n[?] Gostaria de gerar o gráfico de U(v)? (S/N) \n    >>> ")

            if prompt.upper() == "S":
                filename = save_graph()
                print(f"\n ==> Arquivo {filename} salvo no diretório raiz.")


            display_prompt_u()
            display_prompt_bin()

            prompt = ""
            while prompt.upper() not in ("S", "N"):
                prompt = input("\n[?] Gostaria de finalizar o programa? (S/N) \n    >>> ")
            if prompt.upper() == "S": break

if __name__ == "__main__":
    main()