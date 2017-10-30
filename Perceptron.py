import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
class Perceptron():
	def  __init__(self,layer_info,target,input_matrix,activation_function='identity',alpha=.1,iterations=100,threshold=0):
		self.activation_function=activation_function
		self.threshold=threshold
		self.alpha=alpha
		self.weight_matrix=[]
		self.bias=[]
		self.a0=[]
		self.activ_fun=self.activation()
		self.deriva=self.deriv()
		self.layer_info=layer_info
		self.delta=[]
		self.target=target
		self.input_matrix=input_matrix
		self.error=[]
		self.iterations=iterations
		if self.layer_info['hidden']==[0]:
			val=[self.layer_info['inputs']]+[self.layer_info['outputs']]
			val1=[self.layer_info['outputs']]

		else:
			val=[self.layer_info['inputs']]+self.layer_info['hidden']+[self.layer_info['outputs']]
			val1=self.layer_info['hidden']+[self.layer_info['outputs']]

		for i in val1:
			self.delta.append(np.zeros(shape=(1,i)))		

		for i in val:
			self.a0.append(np.zeros(shape=(1,i)))
		
		self.layer_info['back']=len(self.a0)-1
		
		for i in range(0,len(val)-1):
			self.weight_matrix.append(np.random.rand(val[i],val[i+1]))
#			self.weight_matrix.append(np.zeros(shape=(val[i],val[i+1])))
		for i in val1:
			self.bias.append(np.zeros(shape=(1,i)))
	def deriv(self):
		def activ(a):
			return a*(1-a)
		return np.vectorize(activ)
	def backpropogate(self):
		for i in range(len(self.a0)-1,0,-1):
			if i==(len(self.a0)-1):
				Err=self.target-self.a0[i]
				del1=self.deriva(self.a0[i])
				self.delta[i-1]=del1.dot(Err)
			else:
				del1=self.deriva(self.a0[i])

				Entropy=del1*self.delta[i]
				self.delta[i-1]=np.multiply(self.weight_matrix[i].T,Entropy)
	def update_weights(self):
		for i in range(0,len(self.bias)):
			self.bias[i]=self.bias[i]+self.delta[i]*self.alpha
			self.weight_matrix[i]=self.weight_matrix[i]+self.alpha*self.delta[i]*self.a0[i].T
	#to calculates the error
	def error_LMS(self):
		return 0.5*np.sum(np.power((self.a0[-1]-self.target),2))
	#to display the present weights and bias
	def display_weights(self):
		print("Weights\n")
		for i in self.weight_matrix:
			print(i,'\n')
		print("Bias\n")
		for i in self.bias:
			print(i,'\n')
		print("Inputs\n")
		for i in self.a0:
			print(i,'\n')
	#given he input matrix, this calculates the output of the network
	def feedforward(self):
		self.a0[0]=np.array(self.input_matrix).reshape(1,len(self.input_matrix))
		for i in range(0,len(self.weight_matrix)):
			z1=self.a0[i].dot(self.weight_matrix[i])+self.bias[i]
			self.a0[i+1]=self.activ_fun(z1)
	#to update a particular bias denoted by 'i' in the bias_matrix
	def update_particular_bias(self,i,input_matrix):
		self.bias[i]=np.array(input_matrix).reshape(1,len(input_matrix))
	#to update a particular weight denoted by 'i' in the weight_matrix
	def update_particular_weight(self,i,input_matrix):
		self.weight_matrix[i]=np.array(input_matrix)
	#returns the vectorised activation function.
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
	def iterate(self):
		for i in range(self.iterations):
			self.feedforward()
			self.error.append(self.error_LMS())
			self.backpropogate()
			self.update_weights()
	def plot_error(self):
		plt.plot(self.error)
		plt.show()


if __name__=="__main__":

	layer_info={
		'inputs':2,
		'outputs':1,
		'hidden':[2]
	}
	er=[]
	nn=Perceptron(layer_info=layer_info,alpha=.9,activation_function='binary sigmoid',input_matrix=[0,1],target=[1])
	#nn.update_particular_weight(0,[[0.6,-0.1],[-0.3,0.4]])
	#nn.update_particular_weight(1,[[0.4],[0.1]])
	#nn.update_particular_bias(0,[.3,.5])
	#nn.update_particular_bias(1,[-0.2])
	#nn.display_weights()
	nn.iterate()
	nn.plot_error()
#	nn.display_weights()
