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



def xor_nn(XOR, THETA1, THETA2, init_w=0, learn=0, alpha=0.01):

    #XOR This is the representation of the training set.
    #THETA1 & THETA2 Current values for the parameters in the network.
    #init_w=0 This tells the function to initialise the weights in the network
    #learn=0 This tells the network to learn from the examples (see below)
    #alpha=0.01 This is the learning rate (default value is 0.01)

    #The first step in our function is to check if we need to initialize the weights
	if init_w == 1:
		THETA1 = 2*np.random.random([2,3]) - 1
		THETA2 = 2*np.random.random([1,3]) - 1
    # We need to record the number of training examples
	T1_DELTA = np.zeros(THETA1.shape)
	T2_DELTA = np.zeros(THETA2.shape)
    # Now we initialize the cost variable
	m = 0
    #Now we initialize “m” that record the number of training examples 
	J = 0.0

	for x in XOR:
        
        # Its the FEEDFORWARD PROCESS
		A1 = np.vstack(([1], np.transpose(x[0:2][np.newaxis])))
		Z2 = np.dot(THETA1, A1)
		A2 = np.vstack(([1], sigmoid(Z2)))
		Z3 = np.dot(THETA2, A2)
		h = sigmoid(Z3)
		print("\n---------- FEEDFORWARD PROCESS")
		print("A1: {} out: {} \nTHETA1: {} Z2: {}".format(np.transpose(A1), [x[2]], np.transpose(THETA1), np.transpose(Z2)))
		print("A2: {} THETA2: {}\nZ3: {} h: {}".format(np.transpose(A2), THETA2, Z3, h))
        # Calc the cost of iteration
		J = J + (x[2] * math.log(h[0])) + ((1 - x[2]) * math.log(1 - h[0]));
        #We’ll use “m” to record the number of training examples that we present to the network in the epoch
		m = m + 1;
        #Its the BACKWARD PROCESS, not used when its necessary only calc network response
		if learn == 1:
            # Its error of sample: expected - calculed
			error = x[2] - h
			delta3 = h*(1 - h)*error
            # Its derivated erro about input
            # Used [1:] for ignore bias weight
			delta2 = (np.dot(np.transpose(THETA2), delta3) * (A2 * (1 - A2)))[1:]
			T2_DELTA = T2_DELTA + np.dot(delta3, np.transpose(A2))
			T1_DELTA = T1_DELTA + np.dot(delta2, np.transpose(A1))
			print("\n---------- BACKWARD PROCESS")
			print("error: {} delta3: {}\tdelta2: {}\t\nT2_DELTA: {}\t T1_DELTA:{}\t".format(error, np.transpose(delta3), np.transpose(delta2), T2_DELTA, T1_DELTA))
		else:
			#If we’re not learning from this example, then we simply display the cost of this particular example
			print("Hypothesis for {} is {}".format(np.transpose(A1), h));
    # Now we calculate the average cost across all the examples
	J = J / -m
    # Lastly, we’ll set the return values as our new updated weights
	if learn == 1:
		THETA1 = THETA1 - (alpha * (T1_DELTA / m))
		THETA2 = THETA2 - (alpha * (T2_DELTA / m))
		print("THETA1: {} THETA2: {} J: {}".format(THETA1, THETA2, J))
		return (THETA1, THETA2, J)
	else:
		print("J: {}".format(J))
		return (THETA1, THETA2)

# Its XOR table data
XOR = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
# Its learning rate of neural network
learning_rate = 0.01
# Start variables of neural network
[THETA1, THETA2, J] = xor_nn(XOR, 0, 0, 1, 1, learning_rate)

t_start = time.clock()
for i in range(1, 100000):
    [THETA1, THETA2, J] = xor_nn(XOR, THETA1, THETA2, 0, 1, learning_rate);
    # Print results at the moment
    if (i%10 == 0):
        print('\n-------------------- Iteration : {}' .format(i))
        [THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2);
    # Exit because J cost is very small
    if (J <= 0.1):
        print('\n-------------------- Stoping process : J = {}' .format(J))
        [THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2);
        t_end = time.clock()
        print('--- Elapsed time ', t_end - t_start, ' s')
        sys.exit()
print('\n-------------------- Finish Iteration ')
[THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2);

t_end = time.clock()
print('--- Elapsed time ', t_end - t_start, ' s')
