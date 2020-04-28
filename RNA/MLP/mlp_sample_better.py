#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# 
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# Implementação do algoritmo de Redes Neurais de Múltiplas Camadas
#  - Incluido rotina de cálculo de erro quadrático médio
#  - Foi utilizada função de ativação tipo sigmóide (e sua derivada)
#  - Note que você pode configurar o tamanho da rede, podendo incluir mais camadas
#  - Sugestão: para ocultar o "print" você pode substitui-los por #print
# -----------------------------------------------------------------------------
import numpy as np
import sys

def sigmoid(x):
    ''' Função sigmóide '''
    return 1./(1+np.exp(-x))

def dsigmoid(x):
    ''' Derivada da função sigmóide '''
    return (x)*(1 - (x))

class MLP:
    ''' Classe do algoritmo MLP '''

    def __init__(self, *args):
        ''' Inicialização de dados da classe MLP '''
        print('------ Inicializa a classe MLP')
        self.shape = args
        n = len(args)

        print('--- Cria as camadas')
        self.layers = []
        # Cria a camada de entrada + bias
        self.layers.append(np.ones(self.shape[0]+1))
        # Cria camadas ocultas e de saída
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        print('self.layers: ', self.layers)

        print('--- Cria os pesos')
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size, self.layers[i+1].size)))
        print(' self.weights: ', self.weights)
        # dw armazena a última variação dos pesos, para uso no momentum
        self.dw = [0,]*len(self.weights)
        print('self.dw: ', self.dw)

    def reset(self):
        ''' Reinicia os valores dos pesos '''
        print('--- Reinicia os pesos')
        # Insere valores aleatórios entre -0.5 e +0.5
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.5
        print('self.weights: \n', self.weights)
              
    def propagate_forward(self, data):
        ''' Propaga os dados da camada de entrada para a camada de saída '''
        print('------ Propaga dados para frente')
        # Insere dados na camada de entrada
        self.layers[0][0:-1] = data
        print('--- data:', data)
        # Propaga os dados da camada de entrada para as camadas ocultas
        for i in range(1,len(self.shape)):
            print('--- camada: ',i)
            soma = np.dot(self.layers[i-1],self.weights[i-1])
            self.layers[i][...] = sigmoid(soma)
            print('soma: ', soma)
            print(self.layers[i])
        # Retorna o valor da saída da rede neural
        # Sendo que [-1] corresponde ao último dado do array
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Propara o erro para trás conforme a taxa de aprendizagem '''
        deltas = []
        print('------ Propaga erro para trás')
        # Calcula erro da camada de saída
        print('--- Calcula o erro camada de saída')
        error = target - self.layers[-1]
        print(' error: ', error)
        delta = error*dsigmoid(self.layers[-1])
        print(' delta: ', delta)
        deltas.append(delta)
 
        # Calcula erro das camadas ocultas
        print('--- Calcula o erro das camadas ocultas')
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            print(' delta: ', delta)
        print(' deltas: ', deltas)

        # Atualiza os valores dos pesos
        print(' --- Correção dos pesos')
        for i in range(len(self.weights)):
            print('--- camada :', i)
            print('self.weights: ', self.weights[i])
            layer = np.atleast_2d(self.layers[i])
            print('layer: ', layer)
            delta = np.atleast_2d(deltas[i])
            print('delta: ', delta)
            dw = np.dot(layer.T,delta)
            print('dw: ', dw)
            # O peso recebe a correção do ciclo atual e a variação do peso anterior
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw
            print('self.weights: ', self.weights[i])
            print('self.dw: ', self.dw[i])
            
    def calc_error(self, samples):
        # Calcula o erro quadrático médio
        error = 0
        errors = 0
        for i in range(samples.size):
            y = self.propagate_forward(samples['input'][i])
            d = samples['output'][i]
            error = d - y
            errors += error*error
        return errors/2


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network,samples, epochs=100000, lrate=.1, momentum=0.25):
        MSE = 1
        it = 0
        stop = 0.01
        # Fase de Treinamento
        print('--------- Fase de Treinamento')
        # O treinamento é realizado enquanto o MSE > 0.01 e o número de iterações não ultrapassou epochs
        while (MSE > stop and it < epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
            # Avalia o erro quadrático médio a cada 10 iterações
            if(it%10 == 0 and it > 0):
                MSE = network.calc_error(samples)
                print('--------- Iteração: ', it, 'MSE :', MSE)
                print('------ Parada de Treinamento!')
                sys.exit()
            it += 1
            print('------ Parada de Treinamento!')
            sys.exit()

        # Fase de Testes
        print('------ Fase de Testes')
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i] )
            print(i, samples['input'][i], '%.2f' % o[0])
            print('(expected %.2f)' % samples['output'][i])
            
    # Define a configuração da rede MLP
    # Se MLP(2,2,1) possui 2 neurônios de entrada, 2 na camada oculta e 1 na camada de saída
    network = MLP(2,2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("------ Aprende a função logica OR")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)
    '''
    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 0
    samples[2] = (0,1), 0
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)
    '''
    '''
    # Example 4 : Learning sin(x)
    # -------------------------------------------------------------------------
    print("Learning the sin function")
    network = MLP(1,15, 1)
    samples = np.zeros(500, dtype=[('input',  float, 1), ('output', float, 1)])
    samples['input'] = np.linspace(0,1,500)
    samples['output'] = np.sin(samples['input']*np.pi)
    MSE = 1
    stop = 0.01
    epochs = 1000000
    it = 0
    while (MSE > stop and it < epochs):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n], lrate=.1, momentum = 0.9*(epochs/(epochs + it)))
        if(it%100 == 0 and it > 0):
            MSE = network.calc_error(samples)
            print('--------- Iteração: ', it, 'MSE :', MSE)
        it += 1
            
    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['input'],samples['output']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()
    '''
