   
'''
Curso: Engenharia Elétrica
Disciplina: Sistemas Inteligentes
Professor: Carlos Roberto Jr
Título: Algoritmo do perceptron para duas entradas
Observações:
Aprender uma lógica OR para o algoritmo é mais fácil do que aprender uma lógica AND.
Nos testes com logica AND são muitas iterações e não obtem um MSE satisfatório. O que pode ser feito para solucionar?
  
'''


from random import choice
from numpy import array, dot, random

training_data = [
    (array([0,0]), 0),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 1),
]

test_data = [
    (array([0,0]), 0),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 1),
]
# Define variável peso com valores aleatorios
w = random.rand(2)/2 -0.5

# Define a taxa de aprendizagem
a = 0.2
# Define o critério de parada
STOP = 0.01
# Cria variável para contar iterações
it = 0
# Cria a variável do erro quadrático médio
MSE = float(0.0)
# Cria a função degrau unitário
unit_step = lambda x: 0 if x <= 0.4 else 1
# Fase de Treinamento
print('------Fase de Treinamento')
# Realiza o laço enquanto MSE é alto ou se é a primeira iteração
while MSE > STOP or it == 0:
    it = it + 1
    print('---------- Iteration: ',it)
    for i in range(4):
        # Seleciona uma amostra aleatória da base de dados
        x, d = choice(training_data)
        # Realiza a operação de soma do perceptron
        sum = dot(w, x)
        # Aplica a somatória na função de ativação
        y = unit_step(sum)
        # Calcula o erro
        e = d - y
        # Imprime os dados
        print("x: {} w: {} sum: {} -> y: {} e: {}".format(x, w, sum, y, e))
        # Realiza o ajuste dos pesos do perceptron
        w += a * e * x
        # Imprime o peso corrigido
        print('w : {}'.format(w))
        # Apresenta os resultados 
        
    # Apos treinar com toda base de dados calcula o MSE
    print('--- Calcula o erro quadrático médio')
    MSE = 0
    for i in range(4):
        # Seleciona uma amostra sequencial da base de dados
        x, d = training_data[i]
        # Realiza a operação de soma do perceptron
        sum = dot(w, x)
        # Aplica a somatória na função de ativação
        y = unit_step(sum)
        # Calcula o erro do perceptron
        e = d - y
        # Calcula o erro quadrático médio
        MSE = MSE + e*e
        # Apresenta os resultados 
        print("{}: {} -> {} \t [e = {}, MSE = {}]".format(x, sum, y, e, MSE))
    # O erro quadrático médio é multiplicado por 1/2
    MSE = MSE*0.5
    print("MSE = {}".format(MSE))

# Fase de Testes
print('--------- Fase de Testes')
# Para cada amostra da base de treinamento
for x, _ in test_data:
    # Realiza a operação de somatória
    sum = dot(x, w)
    # Calcula a saída do perceptron
    y = unit_step(sum)
    # Imprime os resultados
    print("{}: {} -> {}".format(x, sum, y))
