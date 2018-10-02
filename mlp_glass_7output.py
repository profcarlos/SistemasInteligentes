#-----------------------------------------------------------------------
# Copyrights 2018 Carlos Roberto
# 
# Objetivo: Comandos básicos para leitura de arquivos de dados para
#           inserção de dados na Rede Neural de Múltiplas Camadas.
#           Exemplo feito a partir da base de dados glass
#           http://ftp.ics.uci.edu/pub/machine-learning-databases/glass/
#-----------------------------------------------------------------------
import numpy
import os
import sys
from random import random
import math

def sigmoid(x):
    ''' Função sigmóide '''
    return 1./(1+numpy.exp(-x))

def dsigmoid(x):
    ''' Derivada da função sigmóide '''
    return (x)*(1 - (x))

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    t = numpy.tanh(x)
    return (1 - t**2)

class MLP:
    ''' Classe do algoritmo MLP '''

    def __init__(self, *args):
        ''' Inicialização de dados da classe MLP '''
        #print('------ Inicializa a classe MLP')
        self.shape = args
        n = len(args)

        #print('--- Cria as camadas')
        self.layers = []
        # Cria a camada de entrada + bias
        self.layers.append(numpy.ones(self.shape[0]+1))
        # Cria camadas ocultas e de saída
        for i in range(1,n):
            self.layers.append(numpy.ones(self.shape[i]))
        #print('self.layers: ', self.layers)

        #print('--- Cria os pesos')
        self.weights = []
        for i in range(n-1):
            self.weights.append(numpy.zeros((self.layers[i].size, self.layers[i+1].size)))
        #print(' self.weights: ', self.weights)
        # dw armazena a última variação dos pesos, para uso no momentum
        self.dw = [0,]*len(self.weights)
        #print('self.dw: ', self.dw)

    def reset(self):
        ''' Reinicia os valores dos pesos '''
        #print('--- Reinicia os pesos')
        # Insere valores aleatórios entre -0.5 e +0.5
        for i in range(len(self.weights)):
            Z = numpy.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.5
        #print('self.weights: \n', self.weights)
              
    def propagate_forward(self, data):
        ''' Propaga os dados da camada de entrada para a camada de saída '''
        #print('------ Propaga dados para frente')
        # Insere dados na camada de entrada
        self.layers[0][0:-1] = data
        #print('--- data:', data)
        # Propaga os dados da camada de entrada para as camadas ocultas
        for i in range(1,len(self.shape)):
            #print('--- camada: ',i)
            soma = numpy.dot(self.layers[i-1],self.weights[i-1])
            #print('weight:', self.weights[i-1])
            #print('layer:', self.layers[i-1])
            #print('soma: ', soma)
            self.layers[i][...] = sigmoid(soma)
            #self.layers[i][...] = tanh(soma)
            #print(self.layers[i])
        # Retorna o valor da saída da rede neural
        # Sendo que [-1] corresponde ao último dado do array
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Propara o erro para trás conforme a taxa de aprendizagem '''
        deltas = []
        #print('------ Propaga erro para trás')
        # Calcula erro da camada de saída
        #print('--- Calcula o erro camada de saída')
        error = target - self.layers[-1]
        #print(' error: ', error)
        #delta = error*dtanh(self.layers[-1])
        delta = error*dsigmoid(self.layers[-1])
        #print(' delta: ', delta)
        deltas.append(delta)
 
        # Calcula erro das camadas ocultas
        #print('--- Calcula o erro das camadas ocultas')
        for i in range(len(self.shape)-2,0,-1):
            delta = numpy.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            #delta = numpy.dot(deltas[0],self.weights[i].T)*dtanh(self.layers[i])
            deltas.insert(0,delta)
            #print(' delta: ', delta)
        #print(' deltas: ', deltas)

        # Atualiza os valores dos pesos
        #print(' --- Correção dos pesos')
        for i in range(len(self.weights)):
            #print('--- camada :', i)
            #print('self.weights: ', self.weights[i])
            layer = numpy.atleast_2d(self.layers[i])
            #print('layer: ', layer)
            delta = numpy.atleast_2d(deltas[i])
            #print('delta: ', delta)
            dw = numpy.dot(layer.T,delta)
            #print('dw: ', dw)
            # O peso recebe a correção do ciclo atual e a variação do peso anterior
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw
            #print('self.weights: ', self.weights[i])
            #print('self.dw: ', self.dw[i])
            
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


# remapeia os atributos de entrada para o limiar escolhido, no caso 0 a 1
def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    if oMin == oMax:
        print('Warning: Zero input range')
        return None

    if nMin == nMax:
        print('Warning: Zero output range')
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

def learn(network,samples, epochs=1000000, lrate=.4, momentum=0.2):
    MSE = 1
    it = 0
    stop = 0.001
    ITER = []
    # Fase de Treinamento
    #print('--------- Fase de Treinamento')
    # O treinamento é realizado enquanto o MSE > 0.01 e o número de iterações não ultrapassou epochs
    while (MSE > stop and it < epochs):
        n = numpy.random.randint(samples.size)
        network.propagate_forward( samples['input'][n] )
        network.propagate_backward( samples['output'][n], lrate, momentum )
        #print(samples[n])
        # Avalia o erro quadrático médio a cada 10 iterações
        if(it%1000 == 0 and it > 0):
            MSE = numpy.mean(network.calc_error(samples))
            print('--------- Iteração: ', it, 'MSE :', MSE)
            #print('------ Parada de Treinamento!')
            #sys.exit()
            ITER.append([it, MSE])
        it += 1
        #print('------ Parada de Treinamento!')
        #sys.exit()
    numpy.savetxt('c:\\carlos\\iter_data.csv', ITER, delimiter = ',', fmt = ['%d', '%.2f'])
    # Fase de Testes
    #print('------ Fase de Testes')
    acertos = 0
    erros = numpy.zeros(len(samples['output'][0]))
    for i in range(samples.size):
        out = network.propagate_forward(samples['input'][i] )
        if(numpy.argmax(samples['output'][i]) == numpy.argmax(out) ):
            acertos = acertos + 1
        else:
            erros[numpy.argmax(samples['output'][i])] += 1
    print('----------------------------------------')
    print('acertos: ', acertos, ' de ', len(samples))
    print('erros:', erros)


def main():
    # Insira o caminho do arquivo, veja que se usa \\ ao invés de apenas \
    #dir = 'C:\\CARLOS\\IFG\\Disciplinas\\Sistemas Inteligentes\\RNA MLP\\'
    dir = 'E:\\IFG\\\Disciplinas\\\Sistemas Inteligentes\\\RNA MLP\\'
    dir = 'E:\\IFG\\Disciplinas\\Sistemas Inteligentes\\RNA MLP'
    dir = 'C:\\CARLOS\\IFG\\Disciplinas\\Sistemas Inteligentes\\RNA MLP\\'
    # Insira o nome do arquivo
    file = 'glass.data'
    # Use o join para criar o caminho completo
    caminho = os.path.join(dir, file)
    print('Nome do arquivo : %s\n'% caminho)
    DATA = numpy.loadtxt(caminho, delimiter = ',')
    print('Dados do arquivo:')
    n_samp = len(DATA)
    n_atr = len(DATA[0])
    # Cria array para mapeamento de atributos
    DMAP = DATA
    # Extrai as entradas (9 atributos)
    DINP = numpy.zeros((n_samp, n_atr-2))
    # Extrai a saída (1 padrão)
    DOUT = numpy.zeros((n_samp,7))
    
    #-------------------------------------------------------------------------
    # Informações sobre os limites dos atributos em glass.data
    #    
    #Summary Statistics:
    #Attribute:   Min     Max      Mean     SD      Correlation with class
    #   2. RI:       1.5112  1.5339   1.5184  0.0030  -0.1642
    #   3. Na:      10.73   17.38    13.4079  0.8166   0.5030
    #   4. Mg:       0       4.49     2.6845  1.4424  -0.7447
    #   5. Al:       0.29    3.5      1.4449  0.4993   0.5988
    #   6. Si:      69.81   75.41    72.6509  0.7745   0.1515
    #   7. K:        0       6.21     0.4971  0.6522  -0.0100
    #   8. Ca:       5.43   16.19     8.9570  1.4232   0.0007
    #   9. Ba:       0       3.15     0.1750  0.4972   0.5751
    #   10. Fe:       0       0.51     0.0570  0.0974  -0.1879
    #-------------------------------------------------------------------------
    
    # Array com limites dos atributos 
    RANGE = numpy.array([[0,0],[1.5112, 1.5339],[10.73, 17.38],[0, 4.49],[0.29, 3.5],[69.81, 75.41], [0, 6.21],[5.43, 16.19],[0, 3.15], [0, 0.51]])
    # Converte atributos para os novos limites, você pode avaliar 0 a 1, -0.5 a 0.5, -1 a 1...
    for i in range(n_samp):
        print(DATA[i])
        for j in range(1, n_atr-1):
            #print(DATA[i][j])
            DMAP[i][j] = remap(DATA[i][j], RANGE[j][0], RANGE[j][1], 0, 1)
            #print(DMAP[i][j])
            DINP[i][j-1] = DMAP[i][j]
        #print('out:', DMAP[i][10])
        #sys.exit()
        DOUT[i][int(DMAP[i][10]-1)] = 1
        print(DMAP[i])
       	print(DINP[i])
        print(DOUT[i])        
        print('------')
    # Cria array de entrada e saída
    DINP_TR = []
    DINP_TS = []
    DOUT_TR = []
    DOUT_TS = []
    for i in range(n_samp):
        if(random() < 0.7):
            DINP_TR.append(DINP[i])
            DOUT_TR.append(DOUT[i])
        else:
            DINP_TS.append(DINP[i])
            DOUT_TS.append(DOUT[i])            
    print('samples training:', len(DINP_TR))
    print('samples test: ', len(DINP_TS))
    n_tr = len(DINP_TR)
            
    # Define a configuração da rede MLP
    # Configuração topológica da RNA. Se (9, 12, 1) possui 9 neurônios de entrada, 12 ocultos e 1 de saída
    network = MLP(9, 12,  7)
    samples = numpy.zeros(n_tr, dtype=[('input',  float, 9), ('output', float, 7)])
    for i in range(n_tr):
        samples['input'][i]  = DINP_TR[i]
        samples['output'][i] = DOUT_TR[i]
        print('i: ',samples['input'][i], 'o:', samples['output'][i])

    #sys.exit()
    learn(network, samples)
        
if __name__ == "__main__":
    main()
