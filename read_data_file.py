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

# Insira o caminho do arquivo, veja que se usa \\ ao invés de apenas \
dir = 'E:\\IFG\\Disciplinas\\Sistemas Inteligentes\\RNA\\'
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
for i in range(len(DATA[0])):
    print(DATA)

