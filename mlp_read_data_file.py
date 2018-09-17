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

def main():
    # Insira o caminho do arquivo, veja que se usa \\ ao invés de apenas \
    dir = 'C:\\CARLOS\\IFG\\Disciplinas\\Sistemas Inteligentes\\RNA MLP\\'
    # Insira o nome do arquivo
    file = 'glass.data'
    # Use o join para criar o caminho completo
    caminho = os.path.join(dir, file)
    print('Nome do arquivo : %s\n'% caminho)
    # Use o load.txt da biblioteca numpy para ler e salvar os dados do arquivo
    # Conheça as possibilidades do comando em
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html
    # O delimiter representa o símbolo utilizado para delimitar os atributos
    # Veja que alguns arquivos .data usam espaço
    # Arquivos .data com dados ausentes (?) pode ser usado o numpy.genfromtxt pois o
    # numpy.loadtxt pode dar erro devido caractere inválido, veja em:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
    DATA = numpy.loadtxt(caminho, delimiter = ',')
    print('Dados do arquivo:')
    n_samp = len(DATA)
    n_atr = len(DATA[0])
    # Cria array para mapeamento de atributos
    DMAP = DATA
    # Extrai as entradas (9 atributos)
    DINP = numpy.zeros((n_samp, n_atr-2))
    # Extrai a saída (1 padrão)
    DOUT = numpy.zeros((n_samp,1))
    
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
        DOUT[i] = DMAP[i][10]
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
            
        
    
    # Seleciona dados de treinamento e dados de teste
    
   
        
if __name__ == "__main__":
    main()
