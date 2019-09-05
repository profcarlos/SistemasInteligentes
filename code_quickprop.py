'''
Curso: Engenharia Elétrica
Disciplina: Sistemas Inteligentes
Professor: Carlos Roberto Jr
Título: Algoritmo da rede MLP para solução do problema XOR
Observações:
Maiores informações disponível em:
https://aimatters.wordpress.com/2016/01/11/solving-xor-with-a-neural-network-in-python/
  
'''

import numpy as np
import math
import time
import sys

def sigmoid(x):
	# Calc sigmoid function
	return 1.0 / (1.0 + np.exp(-x))

def xor_nn(XOR, Wij, Wjk, Wij_ant=0, Wjk_ant=0, init_w=0, learn=0, alpha=0.01):

	#XOR This is the representation of the training set.
	#Wij & Wjk Current values for the parameters in the network.
	#init_w=0 This tells the function to initialise the weights in the network
	#learn=0 This tells the network to learn from the examples (see below)
	#alpha=0.01 This is the learning rate (default value is 0.01)

	#The first step in our function is to check if we need to initialize the weights
	if init_w == 1:
		Wij = 2*np.random.random([2,3]) - 1
		Wjk = 2*np.random.random([1,3]) - 1
	# Now we initialize the cost variable
	m = 0
	#Now we initialize “m” that record the number of training examples 
	MSE = 0.0
	for x in XOR:
		# Its the FEEDFORWARD PROCESS
		Xi = np.vstack(([1], np.transpose(x[0:2][np.newaxis])))
		Sj = np.dot(Wij, Xi)
		Yj = np.vstack(([1], sigmoid(Sj)))
		Sk = np.dot(Wjk, Yj)
		Yk = sigmoid(Sk)
		print("\n---------- FEEDFORWARD PROCESS")
		print("Xi: {} out: {} \nWij: {} Sj: {}".format(np.transpose(Xi), [x[2]], Wij, np.transpose(Sj)))
		print("Yj: {} Wjk: {}\nSk: {} Yk: {}".format(np.transpose(Yj), Wjk, Sk, Yk))
	   	# We’ll use “m” to record the number of training examples that we present to the network in the epoch
		m = m + 1;
		# Its the BACKWARD PROCESS, not used when its necessary only calc network response
		if learn == 1:
			# Its error of sample: expected - calculed
			error = x[2] - Yk
			# Calc the cost of iteration using Mean Square Error
			MSE = MSE + error*error;
			# Calc delta_k of the last layer
			# Its derivated error about input
			delta_k = Yk*(1 - Yk)*error
			# Calc delta_j of the hidden layer
			# Used [1:] for ignore bias weight
			delta_j = (np.dot(np.transpose(Wjk), delta_k) * (Yj * (1 - Yj)))[1:]
                        Wjk_ant = Wjk
                        Wij_ant = Wij
			Wjk = Wjk + (alpha * np.dot(delta_k, np.transpose(Yj)))
			Wij = Wij + (alpha * np.dot(delta_j, np.transpose(Xi)))
			print("\n---------- BACKWARD PROCESS")
			print("error: {} delta_k: {} delta_j: {} Wij: {} Wjk: {}".format(error, np.transpose(delta_k), np.transpose(delta_j),  Wij, Wjk))
		else:
			# If we’re not learning from this example, then we simply display the cost of this particular example
			print("Hypothesis for {} is {}".format(np.transpose(Xi), Yk));
	# Lastly, we’ll set the return values as our new updated weights
	if learn == 1:
		# Calc the cost of iteration using Mean Square Error
		MSE = MSE/2
		print("MSE: {}".format(MSE))
		return (Wij, Wjk, MSE)
	else:
		return (Wij, Wjk)

# Its XOR table data
XOR = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
# Its learning rate of neural network
learning_rate = 0.1
# Start variables of neural network
[Wij, Wjk, MSE] = xor_nn(XOR, 0, 0, 0, 0, 1, 1)
Wij_ant = Wij
Wjk_ant = Wjk
MSE_sample = []
t_start = time.clock()
# Se python 3 use t_start = time.perf_counter()
for i in range(1, 100000):
	if(i == 1):
		print('\n-------------------- Iteration : {}' .format(1))
	[Wij, Wjk, MSE] = xor_nn(XOR, Wij, Wjk, Wij_ant, Wjk_ant, 0, 1, learning_rate)

	# Print results at the moment
	if (i%10 == 0):
		print('\n-------------------- Iteration : {}' .format(i))
		[Wij, Wjk] = xor_nn(XOR, Wij, Wjk);
		# Save epoch MSE 
		MSE_sample.append([i, MSE])
		np.savetxt('d:\\iter_data.csv', MSE_sample, delimiter = ',', fmt = ['%d', '%.4f'])		
	# Exit case MSE <= threshold
	if (MSE <= 0.001):
		print('\n-------------------- Stoping process in Iteration: {} MSE = {}' .format(i, MSE))
		[Wij, Wjk] = xor_nn(XOR, Wij, Wjk);
		t_end = time.clock()
		# Se python 3 use t_end = time.perf_counter()
		print('--- Elapsed time ', t_end - t_start, ' s')
		sys.exit()
		# Se python 3 Instale a biblioteca. No command use: pip install sys
print('\n-------------------- Finish in Iteration: {}'.format(i))
[Wij, Wjk] = xor_nn(XOR, Wij, Wjk);

t_end = time.clock()
# Se python 3 use t_end = time.perf_counter()
print('--- Elapsed time ', t_end - t_start, ' s')
