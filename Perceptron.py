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
		self.layer_info=layer_info
		if self.layer_info['hidden']==[0]:
			val=[self.layer_info['inputs']]+[self.layer_info['outputs']]
			val1=[self.layer_info['outputs']]

		else:
			val=[self.layer_info['inputs']]+self.layer_info['hidden']+[self.layer_info['outputs']]
			val1=self.layer_info['hidden']+[self.layer_info['outputs']]
		
		#self.delta=np.zeros(shape=(1,sum((val1)))

		for i in val:
			self.a0.append(np.zeros(shape=(1,i)))
		
		self.layer_info['back']=len(self.a0)-1
		
		for i in range(0,len(val)-1):
			self.weight_matrix.append(np.zeros(shape=(val[i],val[i+1])))

		for i in val1:
			self.bias.append(np.zeros(shape=(1,i)))
	#to calculates the error
	def error_LMS(self,t):
		return 0.5*np.sum(np.power((self.a0[-1]-t),2))
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
	def feedforward(self,input_matrix):
		self.a0[0]=np.array(input_matrix).reshape(1,len(input_matrix))
		for i in range(0,len(self.weight_matrix)):
			z1=np.array(self.a0[i].dot(self.weight_matrix[i])+self.bias[i])
			#print(self.a0[i],self.weight_matrix[i],self.bias[i])
			self.a0[i+1]=self.activ_fun(z1)
			self.a0[i+1]=self.a0[i+1].reshape(1,self.a0[i+1].shape[1])
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

	def cal_delta(self,t):
		temp_weight_matrix=self.weight_matrix
		temp_bias=self.bias
	#for the output layer
		del_output=np.array(self.a0[len(self.a0)-1]*(1-self.a0[len(self.a0)-1])*(t-self.a0[len(self.a0)-1]))
	#for 1st hidden layer from back
		temp1=del_output*self.weight_matrix[len(self.weight_matrix)-1]*self.alpha		
		temp2=self.bias[len(self.bias)-1]*del_output
		temp_weight_matrix[len(temp_weight_matrix)-1]=self.weight_matrix[len(self.weight_matrix)-1]+temp1
		temp_bias[len(self.bias)-1]=self.bias[len(self.bias)-1]+temp2
		
		for i in range(self.layer_info['back']-1,0,-1):
			del_output=self.a0[i]*(1-self.a0[i])*np.sum(self.weight_matrix[i]*del_output,axis=1)
			temp1=self.a0[i-1].reshape(len(self.weight_matrix[i-1]),1)*del_output*self.alpha		
			temp2=self.bias[i-1]*del_output
			#print(temp_weight_matrix[i])
			temp_weight_matrix[i-1]=self.weight_matrix[i-1]+temp1
			temp_bias[i-1]=self.bias[i-1]+temp2

		self.weight_matrix=temp_weight_matrix
		self.bias=temp_bias

if __name__=="__main__":

	layer_info={
		'inputs':2,
		'outputs':2,
		'hidden':[3]
	}
	nn=Perceptron(layer_info=layer_info,alpha=.1,activation_function='binary sigmoid')
	nn.update_particular_weight(0,[[0.1,0,0.3],[-0.2,0.2,-0.4]])
	nn.update_particular_weight(1,[[-0.4,0.2],[0.1,-0.1],[0.6,-0.2]])
	nn.update_particular_bias(0,[.1,.2,.5])
	nn.update_particular_bias(1,[-.1,.6])
	nn.target=np.array([1,0])
'''
	for i in range(10000):
		nn.feedforward(input_matrix=[.6,.1])
		#nn.display_weights()
		if np.array_equal(nn.target,nn.a0[len(nn.a0)-1])==0:
			nn.cal_delta(nn.target)
		print(nn.error_LMS(nn.target))
'''
