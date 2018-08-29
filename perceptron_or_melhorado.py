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
    (array([0,1]), 0),
    (array([1,0]), 0),
    (array([1,1]), 1),
]

test_data = [
    (array([0,0]), 0),
    (array([0,1]), 0),
    (array([1,0]), 0),
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
MSE = 0
# Cria a função degrau unitário
unit_step = lambda x: 0 if x <= 0 else 1
# Fase de Treinamento
print('------Fase de Treinamento')
# Realiza o laço enquanto MSE é alto ou se é a primeira iteração
while MSE > STOP or it == 0:
    it = it + 1
    print('---------- Iteration: ',it)
    for i in range(12):
        # Seleciona uma amostra aleatória da base de dados
        x, d = x, d = choice(training_data)
        # Realiza a operação de soma do perceptron
        sum = dot(w, x)
        # Aplica a somatória na função de ativação
        y = unit_step(sum)
        # Calcula o erro
        e = d - y
        # Realiza o ajuste dos pesos do perceptron
        w += a * e * x
        # Imprime os dados
        print('x:')
        print(x)
        print('w:')
        print(w)
        print('y:')
        print(y)
        print('e:')
        print(e)
    # Apos treinar com toda base de dados calcula o MSE    
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
    # O erro quadrático médio é multiplicado por 1/2
    MSE = (1/2)*MSE
    # Imprime o MSE
    print('--- Calcula o erro quadrático médio')
    print('MSE = %f' %MSE)
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
