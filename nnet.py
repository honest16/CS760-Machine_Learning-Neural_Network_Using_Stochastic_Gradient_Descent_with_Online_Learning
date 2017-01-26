# Importing libraries
import sys
import math
import random
import collections
import scipy
import numpy as np

class dataContents(object):
	def __init__(self):
		self.attributes = None
		self.instances = None

def getContents(filename):
	f = open(str(filename),'r')
	data, meta = arff.loadarff(f)
	f.close()
	contents = dataContents()
	contents.instances = data
	contents.attributes = []
	
	
	# Names of the attributes
	for i in range(len(data.dtype.names)):
		contents.attributes.append([data.dtype.names[i]])
		
	
	metaSpl = str(meta).split('\n')
	attrInd = 0
	for i in range(len(metaSpl)):
		if 'type' in metaSpl[i] and not 'range' in metaSpl[i]:
			for j in range(len(metaSpl[i])):
				k = len(metaSpl[i])-1-j
	
				if metaSpl[i][k] == ' ':
					 contents.attributes[attrInd].append(metaSpl[i][k+1:])
					 contents.attributes[attrInd].append(None)
					 attrInd += 1
					 break
		elif  'type' in metaSpl[i] and 'range' in metaSpl[i]:
			furthSpl = metaSpl[i].split(',')
			
			for j in range(len(furthSpl[0])):
				k = len(furthSpl[0])-1-j
				if furthSpl[0][k] == ' ':
					 contents.attributes[attrInd].append(furthSpl[0][k+1:])
					 break
					 
			rangeVals = []
	
			rangeStIn = metaSpl[i].index('(')
			rangeEnIn = metaSpl[i].index(')')
			vals = metaSpl[i][rangeStIn+1:rangeEnIn].split(',')
			rangeVals = [vals[i].strip()[1:-1] for i in range(len(vals))]
			contents.attributes[attrInd].append(rangeVals)
			attrInd += 1
	
	
	return contents
	
	
class nnet(object):
	def __init__(self):
		self.l = None
		self.h = None
		self.e = None
		self.train_set = None
		self.test_set = None
		self.train_instsNorm = None
		self.test_instsNorm = None
		self.encodedInput = None
		self.featEncodedInputMap = None
		self.epochError = None
		self.epochCorrect = None
		self.epochIncorrect = None
		
		self.epochOutput = []
		
	
		self.encodedTrainIns = None
		self.encodedTestIns = None
		
	
		self.weights = None
		self.wtRange = [-0.01,0.01]
		
	
		self.obtainCLIArgs()
		self.normalizeFeat()
		self.determineInputEncoding()
		self.encodeInstances()
		# Method to initialize the weights
		self.initializeWeights()
		
		for epNo in range(0,self.e):
		
			self.epochError = 0
			self.epochCorrect = 0
			self.epochIncorrect = 0
			
			perm = np.random.permutation(len(self.train_set.instances))
		
			for insNo in range(0,len(perm)):
				insEncd = self.encodedTrainIns[perm[insNo]]
				
				output = self.getOutput(insEncd)
				
				if self.train_set.attributes[-1][2].index(self.train_set.instances[perm[insNo]][-1]) == 0: 
					labelNum = 0
				elif self.train_set.attributes[-1][2].index(self.train_set.instances[perm[insNo]][-1]) == 1:
					labelNum = 1
				
				
				if self.h == 0:
					if output > 0.5:
						if labelNum  == 1:
							self.epochCorrect += 1
						elif  labelNum  == 0:
							self.epochIncorrect += 1
					else:
						if labelNum == 1:
							self.epochIncorrect += 1
						elif labelNum == 0:
							self.epochCorrect += 1
				elif self.h > 0:
					if output[1] > 0.5:
						if labelNum  == 1:
							self.epochCorrect += 1
						elif  labelNum  == 0:
							self.epochIncorrect += 1
					else:
						if labelNum == 1:
							self.epochIncorrect += 1
						elif labelNum == 0:
							self.epochCorrect += 1
				
				
				if self.h == 0:
					error = self.computeError(output,labelNum)
					
				
					self.epochError += error
					grad = self.computeGrad(error,output,None,labelNum, insEncd)
						
					self.updateWeights(grad)
					
				elif self.h > 0:
					error = self.computeError(output[1],labelNum)
					
					
					self.epochError += error
					grad = self.computeGrad(error,output[1],output[0],labelNum, insEncd)
					
					
					self.updateWeights(grad)
			self.epochOutput.append(self.epochError)		
			print str(epNo + 1)+ '\t' + '%0.06f'%(self.epochError) + '\t' + str(self.epochCorrect)+ '\t' + str(self.epochIncorrect)
			
		self.classifyTestSet()	


	def classifyTestSet(self):
		
		numCorrectTest = 0
		numInCorrectTest = 0
		for insNo in range(0,len(self.encodedTestIns)):
			insEncd = self.encodedTestIns[insNo]
			
			output = self.getOutput(insEncd)
			if self.h == 0:
				if output > 0.5:
				
					print '%0.06f'%(output)+ '\t'+ str(self.test_set.attributes[-1][2][1]) + '\t' + str(self.test_set.instances[insNo][-1])
					if str(self.test_set.attributes[-1][2][1]) == str(self.test_set.instances[insNo][-1]):
						numCorrectTest += 1
					else:	
						numInCorrectTest += 1
				else:
					print '%0.06f'%(output)+ '\t'+ str(self.test_set.attributes[-1][2][0]) + '\t' + str(self.test_set.instances[insNo][-1])
					if str(self.test_set.attributes[-1][2][0]) == str(self.test_set.instances[insNo][-1]):
						numCorrectTest += 1
					else:	
						numInCorrectTest += 1
			elif self.h > 0:
				if output[1] > 0.5:
					print '%0.06f'%(output[1])+ '\t'+ str(self.test_set.attributes[-1][2][1]) + '\t' + str(self.test_set.instances[insNo][-1])
					if str(self.test_set.attributes[-1][2][1]) == str(self.test_set.instances[insNo][-1]):
						numCorrectTest += 1
					else:	
						numInCorrectTest += 1
				else:
					print '%0.06f'%(output[1])+ '\t'+ str(self.test_set.attributes[-1][2][0]) + '\t' + str(self.test_set.instances[insNo][-1])
					if str(self.test_set.attributes[-1][2][0]) == str(self.test_set.instances[insNo][-1]):
						numCorrectTest += 1
					else:	
						numInCorrectTest += 1
			
			
		print 'Number of correctly classified test instances: ' + str(numCorrectTest) 
		print 'Number of incorrectly classified test instances: '+ str(numInCorrectTest) 
			
			

	
	def updateWeights(self, grad):

		if self.h == 0:

			self.weights = self.weights - self.l*(grad)

		elif self.h > 0:

			
			self.weights[0] = self.weights[0] - self.l*(grad[0])
			self.weights[1] = self.weights[1] - self.l*(grad[1])
	
			
	
	def computeGrad(self, error, o, hidOut ,y,x):
		
		if self.h == 0:
		
			grad = []
			grad.append((o-y))
		
			for i in range(0,len(self.weights)-1):
		
				grad.append((o-y)*x[i])
				
			grad = np.array(grad)	
			return grad
			
		elif self.h > 0:
		
			
			gradInputHidden = []
			gradHiddenOutput = []
			
			gradHiddenOutput.append((o-y))
			for i in range(1,len(self.weights[1])):
				gradHiddenOutput.append((o-y)*hidOut[i])
				
			for h in range(1,self.h+1):
				gradThisH = []
		
				gradThisH.append((o-y)*self.weights[1][h]*hidOut[h]*(1-hidOut[h])*1)
				for xi in range(len(x)):
					gradThisH.append((o-y)*self.weights[1][h]*x[xi]*hidOut[h]*(1-hidOut[h]))
				gradInputHidden.append(gradThisH)
			
			grad = [np.array(gradInputHidden), np.array(gradHiddenOutput)]
		
			
			return grad
			
			
	
	def computeError(self,o,y):
		res = -y*math.log(o)-(1-y)*math.log(1-o)
		return res
	
	
	def getOutput(self,insEncd):
		if self.h ==0:
			x = [1]
			x = x + insEncd
			
			x = np.array(x)
			
			w = np.array(self.weights)
			
			net = x.dot(w)
			
			out = self.sigmoid(net)
			return out
		elif self.h >0:
			hidOut = [1]
			x = [1]
			x = x + insEncd
			x = np.array(x)
			
			
			
			for i in range(self.h):
				w = np.array(self.weights[0][i])
				
				net = x.dot(w)
				out = self.sigmoid(net)
				hidOut.append(out)
			
			
			hidOut = np.array(hidOut)

						
			w = np.array(self.weights[1])
			
						
			net = hidOut.dot(w)
			out = self.sigmoid(net)
			
			
			out1 = [hidOut, out]
			return out1 
			
			
	
	def sigmoid(self,net):
		out = 1.0/(1+math.exp(-net))
		return out
		
	def encodeInstances(self):	
		self.encodedTrainIns = []
		for i in range(len(self.train_instsNorm)):
			feat = list(self.train_instsNorm[i])[0:-1]
			einp = self.encodedInput[:]
			
			for j in range(len(feat)):
				if ((not self.train_set.attributes[j][2]) and (self.train_set.attributes[j][1] == 'numeric' or self.train_set.attributes[j][1] == 'real')):
					einp[self.featEncodedInputMap[j]] = feat[j]
				elif ( (self.train_set.attributes[j][2]) and (self.train_set.attributes[j][1] == 'nominal')):
					offset = self.train_set.attributes[j][2].index(feat[j])
			
					einp[self.featEncodedInputMap[j]+offset] = 1
					
			self.encodedTrainIns.append(einp)			
			
		self.encodedTestIns = []
		for i in range(len(self.test_instsNorm)):
			feat = list(self.test_instsNorm[i])[0:-1]
			einp = self.encodedInput[:]
			for j in range(len(feat)):
				if ((not self.test_set.attributes[j][2]) and (self.test_set.attributes[j][1] == 'numeric' or self.test_set.attributes[j][1] == 'real')):
					einp[self.featEncodedInputMap[j]] = feat[j]
				elif ( (self.test_set.attributes[j][2]) and (self.test_set.attributes[j][1] == 'nominal')):
					offset = self.test_set.attributes[j][2].index(feat[j])
					einp[self.featEncodedInputMap[j]+offset] = 1
					
			self.encodedTestIns.append(einp)
		
		
		
		
	def determineInputEncoding(self):
		self.encodedInput = []
		encodedInputInd = 0
		self.featEncodedInputMap = {}
		
		for i in range(len(self.train_set.attributes)-1):
			
		
			if ((not self.train_set.attributes[i][2]) and (self.train_set.attributes[i][1] == 'numeric' or self.train_set.attributes[i][1] == 'real')):
		
				self.encodedInput.append(0)
				self.featEncodedInputMap[i] = encodedInputInd
				encodedInputInd += 1
				
			elif ( (self.train_set.attributes[i][2]) and (self.train_set.attributes[i][1] == 'nominal')):
		
				for j in range(len(self.train_set.attributes[i][2])):
					self.encodedInput.append(0)
				self.featEncodedInputMap[i] = encodedInputInd
				encodedInputInd += len(self.train_set.attributes[i][2])
		
		
	
	def initializeWeights(self):
		if self.h == 0:
		 
			self.weights = []
					
			self.weights.append(np.random.uniform(self.wtRange[0],self.wtRange[1]))
			for i in range(len(self.encodedInput)):
				self.weights.append(np.random.uniform(self.wtRange[0],self.wtRange[1]))
			
		elif self.h > 0:
		
			wtsInpHidden = []
		
			for i in range(self.h):
				wtsForThisH = []
				for i in range(len(self.encodedInput)+1):
					wtsForThisH.append(np.random.uniform(self.wtRange[0],self.wtRange[1]))
		
				wtsInpHidden.append(wtsForThisH)
				
				
			
			wtsHiddenOut = []
		
			wtsHiddenOut.append(np.random.uniform(self.wtRange[0],self.wtRange[1]))
		
			for i in range(self.h):
				wtsHiddenOut.append(np.random.uniform(self.wtRange[0],self.wtRange[1]))
				
			self.weights = [wtsInpHidden, wtsHiddenOut]
			
			
			
	# Method to obtain data from command line arguments
	def obtainCLIArgs(self):
		
		self.l = sys.argv[1]
		self.h = sys.argv[2]
		self.e = sys.argv[3]
		
		trainFileName = sys.argv[4]
		
		testFileName = sys.argv[5]
		
		self.train_set = getContents(trainFileName)  
		self.test_set = getContents(testFileName)
		
		
	
	def normalizeFeat(self):
		meanVec = []
		stdDevVec = []
		numericInd = []
		
			
		for i in range(len(self.train_set.attributes)-1):
			if self.train_set.attributes[i][1] == 'numeric' or self.train_set.attributes[i][1] == 'real':
				meanVec.append(0)
				stdDevVec.append(0)
				numericInd.append(i)
			else:
				meanVec.append(None)
				stdDevVec.append(None)
				
		
		
		if len(numericInd) == 0:
			self.train_instsNorm = self.train_set.instances
			self.test_instsNorm = self.test_set.instances
			return
		else:
			# Computing mean
			for trainIns in range(len(self.train_set.instances)):
				for i in range(len(numericInd)):
					meanVec[numericInd[i]] += self.train_set.instances[trainIns][numericInd[i]]
					
			for i in range(len(numericInd)):
				meanVec[numericInd[i]] = meanVec[numericInd[i]]/len(self.train_set.instances)
			
			# Computing std dev
			for trainIns in range(len(self.train_set.instances)):
				for i in range(len(numericInd)):
					stdDevVec[numericInd[i]] += (self.train_set.instances[trainIns][numericInd[i]]-meanVec[numericInd[i]])**2
					
			for i in range(len(numericInd)):
				stdDevVec[numericInd[i]] = (stdDevVec[numericInd[i]]/len(self.train_set.instances))**0.5
				
			# Standardizing training set numeric features
			self.train_instsNorm = []
			for trainIns in range(len(self.train_set.instances)):
				trInlist = list(self.train_set.instances[trainIns])
				for i in range(len(numericInd)):
					trInlist[numericInd[i]] = (trInlist[numericInd[i]]-meanVec[numericInd[i]])/stdDevVec[numericInd[i]]
				self.train_instsNorm.append(tuple(trInlist))
			
		
			
			# Standardizing test set numeric features
			self.test_instsNorm = []
			for testIns in range(len(self.test_set.instances)):
				tstInlist = list(self.test_set.instances[testIns])
				for i in range(len(numericInd)):
					tstInlist[numericInd[i]] = (tstInlist[numericInd[i]]-meanVec[numericInd[i]])/stdDevVec[numericInd[i]]
				self.test_instsNorm.append(tuple(tstInlist))
				
		
	
	
nnet()

