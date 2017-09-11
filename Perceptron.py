import math
import pickle

import numpy as np


class Perceptron():
	def  __init__(self,layer_info,activation_function='identity',alpha=.1,threshold=0):
		self.activation_function=activation_function
		self.threshold=threshold
		self.alpha=alpha
		self.weight_matrix=[]
		self.bias=[]
		self.a0=[]
		self.activ_fun=self.activation()
		self.target=None

		if layer_info['hidden']==[0]:
			val=[layer_info['inputs']]+[layer_info['outputs']]
			val1=[layer_info['outputs']]

		else:
			val=[layer_info['inputs']]+layer_info['hidden']+[layer_info['outputs']]
			val1=layer_info['hidden']+[layer_info['outputs']]

		#self.delta=np.zeros(shape=(1,sum((val1)))

		for i in val:
			self.a0.append(np.zeros(shape=(1,i)))

		for i in range(0,len(val)-1):
			self.weight_matrix.append(np.zeros(shape=(val[i],val[i+1])))

		for i in val1:
			self.bias.append(np.zeros(shape=(1,i)))
	def error_LMS(self,t):
		return 0.5*np.sum(np.power((self.a0[-1]-t),2))

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
	
	def update_weights(self,input_matrix):
		self.a0[0]=np.array(input_matrix)
		for i in range(0,len(self.weight_matrix)):
			z1=np.array(self.a0[i].dot(self.weight_matrix[i]))+self.bias[i]
			#print(self.a0[i],self.weight_matrix[i],self.bias[i])
			self.a0[i+1]=self.activ_fun(z1)
	def update_particular_bias(self,i,input_matrix):
		input_matrix=np.array(input_matrix)
		self.bias[i]=input_matrix
		#self.bias[i]=input_matrix.reshape(1,len(input_matrix))

	def update_particular_weight(self,i,input_matrix):
		self.weight_matrix[i]=np.array(input_matrix)
		#self.weight_matrix[i]=input_matrix.reshape(len(input_matrix),len(input_matrix[0]))
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

	def cal_delta(self,t):
		delta=[]
		delta.append(t-self.a0[len(self.a0)-1])
		#print(delta)
		for i in range(len(self.weight_matrix)-1,0,-1):
			delta.insert(0,delta[0]*self.weight_matrix[i])
		#print(delta)
		for i in range(0,len(self.weight_matrix)):
			weight_matrix_new=self.weight_matrix+self.alpha*t*delta[i]
if __name__=="__main__":
	layer_info={
		'inputs':2,
		'outputs':2,
		'hidden':[2]
	}
	nn=Perceptron(layer_info=layer_info,alpha=.25,activation_function='binary sigmoid')
	
	nn.update_particular_weight(0,[[0.15,0.20],[0.25,0.3]])
	nn.update_particular_weight(1,[[0.40,0.45],[.5,.55]])
	nn.update_particular_bias(0,[0.35,0.35])
	nn.update_particular_bias(1,[0.60,0.60])
	nn.target=np.array([.01,.99])

	nn.update_weights(input_matrix=np.array([0.05,.1]))
	err=nn.error_LMS(np.array([.01,.99]))
	print(err)
	#print(nn.a0[len(nn.a0)-1])
	#nn.display_weights()
	#nn.cal_delta(1)