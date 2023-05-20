#EP1 Fernando Chieffi Moraes - 11811072
import random
import math
def estima_pi(): #dado o n necessario para estimar pi com a precisao de 0.9995 esta funcao cria n pontos dentro do quadrado de de lado 2 e verifica quantos desses pontos estao dentro do circulo de raio r
    k=0
    n=int(((3.48*(1-(2**(1/2)-1)))/0.0005)**2)
    print(n)
    for i in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x**2+y**2<=1:
            k+=1
    return 4*k/n

for i in range(10):
    random.seed(10)
    print(abs(estima_pi()-math.pi)<0.0005*math.pi)