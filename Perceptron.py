import math
import pickle,numpy as np
class Perceptron():
	def  __init__(self,weight_matrix,target_no,activation_function='identity',bias=1,alpha=.1,threshold=0):
		self.activation_function=activation_function
		self.threshold=threshold
		self.alpha=alpha
		self.bias=np.full((target_no,1),bias)
#		self.weight_matrix=np.zeros(shape=(len(input_matrix.flat),target_no))
		self.weight_matrix=weight_matrix
	'''
	updates the input matrix of the neural net
	'''	
	def set_input(self,input_matrix):
		self.input_matrix=input_matrix		
	'''
	This functions takes the inner product of weight matrix and given input,finds its sum, adds bias 
	and then applies activation function and return the output as numpy array
	'''
	def output(self):
		activ=self.activation()
		return activ(self.bias+np.sum(np.inner(self.input_matrix,self.weight_matrix)))
	'''
	This functions takes alpha, target(t) and updates the weights
	'''
	def update_weights(self,t):
#		print("Updating Weights: ")
		alphta_x=np.full(self.weight_matrix.shape,self.alpha*t)
		alphta_x_t=(alphta_x.T*self.input_matrix).T
		self.weight_matrix=self.weight_matrix+alphta_x_t
#		print("Weights Updated/nUpdating Bias")
		bias=self.alpha*t
#		print("Bias Updated")
	def iterations(self,input_array,target):
		for i in range(len(input_array)):
			self.set_input(input_array[i])
			print("input "+str(input_array[i])+" output: "+str(self.output()))
			if self.output()!=target[i]:
				self.update_weights(target[i])
			print("Weights Now: "+str(self.weight_matrix))
	def activation(self):

		nf=True
		if self.activation_function=='identity':
			def activ(a):
				return a
		elif self.activation_function=='binary step':
			def activ(a):
				if a>=self.threshold:
					return 1
				else:
					return 0
		elif self.activation_function=='binary sigmoid':
			def activ(a):
				return 1 / (1 + math.exp(-a))
		elif self.activation_function=='ramp':
			def activ(a):
				if x>1:
					return 1
				elif x>=0 and x<=1:
					return x
				else:
					return 0
		elif self.activation_function=='bipolar step':
			def activ(a):
				if a==0:
					return 0
				elif a>self.threshold:
					return 1
				else:
					return -1
		elif self.activation_function=='bipolar sigmoid':
			def activ(a):
				return ((2 / (1 + math.exp(-a)))-1)
		else:
			nf=False
			print("Activation function not recognized. Choose between. \n1) identity\n2) binary step\n3) ramp")
			print("\n4) bipolar step \n 5) bipolar sigmoid")
		if nf:
			return np.vectorize(activ)

nn=Perceptron(weight_matrix=np.array([0,0]),target_no=2,bias=0,alpha=1,activation_function='bipolar step')
nn.iterations(np.array([[1,1],[1,-1],[-1,1],[-1,-1]]),np.array([1,-1,-1,-1]))