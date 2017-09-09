import math
import pickle,numpy as np
class Perceptron():
	def  __init__(self,input_matrix,weight_matrix,target_no,activation_function='identity',bias=1,alpha=.1,threshold=0):
		self.activation_function=activation_function
		self.threshold=threshold
		self.alpha=alpha
		self.bias=np.full((target_no,1),bias)
		self.input_matrix=input_matrix
#		self.weight_matrix=np.zeros(shape=(len(input_matrix.flat),target_no))
		self.weight_matrix=weight_matrix
	'''
	This functions takes the inner product of weight matrix and given input,finds its sum, adds bias 
	and then applies activation function and return the output as numpy array
	'''
	def output(self,i):
		activ=self.activation()
		return activ(self.bias[i]+np.sum(np.inner(self.input_matrix,self.weight_matrix)))

	def update_weights(self,):
		pass
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
				if a>=self.threshold:
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

nn=Perceptron(input_matrix=np.array([0.8,0.6,0.4]),weight_matrix=np.array([0.1,0.3,-0.2]),target_no=1,bias=0.35,activation_function='bipolar sigmoid')
print(nn.output(0))