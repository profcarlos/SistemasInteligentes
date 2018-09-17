#-----------------------------------------------------------------------
# Copyrights 2018 Carlos Roberto
# 
# Objetivo: Comandos básicos para leitura de arquivos de dados para
#           inserção de dados na Rede Neural de Múltiplas Camadas.
#           Exemplo feito a partir da base de dados glass
#           http://ftp.ics.uci.edu/pub/machine-learning-databases/horse-colic/
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
    file = 'horse-colic.data'
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
    DATA = numpy.genfromtxt(caminho, delimiter = ' ', missing_values = ('?'), filling_values = ('-999'))
    print('Dados do arquivo:')
    n_samp = len(DATA)
    n_atr = len(DATA[0])
    print('n_samp = ', n_samp, 'n_atr = ', n_atr)
    for i in range(n_samp):
        print(DATA[i])
        sys.exit()

   
        
if __name__ == "__main__":
    main()
