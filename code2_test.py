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



def xor_nn(XOR, Wij, Wjk, init_w=0, learn=0, alpha=0.01):

    #XOR This is the representation of the training set.
    #Wij & Wjk Current values for the parameters in the network.
    #init_w=0 This tells the function to initialise the weights in the network
    #learn=0 This tells the network to learn from the examples (see below)
    #alpha=0.01 This is the learning rate (default value is 0.01)

    #The first step in our function is to check if we need to initialize the weights
	if init_w == 1:
		Wij = 2*np.random.random([2,3]) - 1
		Wjk = 2*np.random.random([1,3]) - 1
    # We need to record the number of training examples
	sum_delta_j = np.zeros(Wij.shape)
	sum_delta_k = np.zeros(Wjk.shape)
    # Now we initialize the cost variable
	m = 0
    #Now we initialize “m” that record the number of training examples 
	J = 0.0

	for x in XOR:
        
        # Its the FEEDFORWARD PROCESS
		Xi = np.vstack(([1], np.transpose(x[0:2][np.newaxis])))
		Sj = np.dot(Wij, Xi)
		Yj = np.vstack(([1], sigmoid(Sj)))
		Sk = np.dot(Wjk, Yj)
		Yk = sigmoid(Sk)
		print("\n---------- FEEDFORWARD PROCESS")
		print("Xi: {} out: {} \nWij: {} Sj: {}".format(np.transpose(Xi), [x[2]], np.transpose(Wij), np.transpose(Sj)))
		print("Yj: {} Wjk: {}\nSk: {} Yk: {}".format(np.transpose(Yj), Wjk, Sk, Yk))
        # Calc the cost of iteration
		J = J + (x[2] * math.log(Yk[0])) + ((1 - x[2]) * math.log(1 - Yk[0]));
        #We’ll use “m” to record the number of training examples that we present to the network in the epoch
		m = m + 1;
        #Its the BACKWARD PROCESS, not used when its necessary only calc network response
		if learn == 1:
            # Its error of sample: expected - calculed
			error = x[2] - Yk
			delta_k = Yk*(1 - Yk)*error
            # Its derivated erro about input
            # Used [1:] for ignore bias weight
			delta_j = (np.dot(np.transpose(Wjk), delta_k) * (Yj * (1 - Yj)))[1:]
			sum_delta_k += np.dot(delta_k, np.transpose(Yj))
			sum_delta_j += np.dot(delta_j, np.transpose(Xi))
			print("\n---------- BACKWARD PROCESS")
			print("error: {} delta_k: {}\tdelta_j: {}\t\nsum_delta_k: {}\t sum_delta_j:{}\t".format(error, np.transpose(delta_k), np.transpose(delta_j), sum_delta_k, sum_delta_j))
		else:
			#If we’re not learning from this example, then we simply display the cost of this particular example
			print("Hypothesis for {} is {}".format(np.transpose(Xi), Yk));
    # Now we calculate the average cost across all the examples
	J = J / -m
    # Lastly, we’ll set the return values as our new updated weights
	if learn == 1:
		Wij = Wij + (alpha * (sum_delta_j / m))
		Wjk = Wjk + (alpha * (sum_delta_k / m))
		print("Wij: {} Wjk: {} J: {}".format(Wij, Wjk, J))
		return (Wij, Wjk, J)
	else:
		print("J: {}".format(J))
		return (Wij, Wjk)

# Its XOR table data
XOR = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
# Its learning rate of neural network
learning_rate = 0.01
# Start variables of neural network
[Wij, Wjk, J] = xor_nn(XOR, 0, 0, 1, 1, learning_rate)
J_cost = []
t_start = time.clock()
for i in range(1, 10000):
    [Wij, Wjk, J] = xor_nn(XOR, Wij, Wjk, 0, 1, learning_rate);
    J_cost.append(J)
    # Print results at the moment
    if (i%10 == 0):
        print('\n-------------------- Iteration : {}' .format(i))
        [Wij, Wjk] = xor_nn(XOR, Wij, Wjk);
        
    # Exit because J cost is very small
    if (J <= 0.1):
        print('\n-------------------- Stoping process : J = {}' .format(J))
        [Wij, Wjk] = xor_nn(XOR, Wij, Wjk);
        t_end = time.clock()
        print("J_cost: {}".format(J_cost))
        print('--- Elapsed time ', t_end - t_start, ' s')
        sys.exit()
print('\n-------------------- Finish Iteration ')
[Wij, Wjk] = xor_nn(XOR, Wij, Wjk);

t_end = time.clock()
print("J_cost: {}".format(J_cost))
print('--- Elapsed time ', t_end - t_start, ' s')
