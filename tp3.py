import math
import numpy as np
from pylab import rand
from sklearn.model_selection import cross_val_score
import os
from tp5utils import chargementVecteursImages

def noyauGaussien(x1, x2, sigma):
	npx1 = np.array(x1)
	npx2 = np.array(x2)
	return np.exp(-(pow(np.linalg.norm(npx1-npx2),2))/pow(sigma,2))

def noyauPolynomial(x1,x2,k):
	xa = np.array(x1)
	xb = np.array(x2)
	return pow((1+np.vdot(xa,xb)),k)

def predir():
	pass

def learnKernelPerceptron(data, target, kernel, h):
	alpha = np.zeros(len(data))
	iter = 0
	N = 10
	somme = 0
	flag = 1
	while(flag != 0 and iter <= N):
		flag = 0
		for i in range(len(data)):
			xi = data[i]
			if(kernel == 1):
				for j in range(len(data)):
					xj = data[j]
					somme += alpha[j]*target[j]*noyauGaussien(xj,xi,h)
			else:
				for j in range(len(data)):
					xj = data[j]
					somme += alpha[j]*target[j]*noyauPolynomial(xi,xj,h)
			y_predit = np.sign(somme)
			if (y_predit != target[i]) :
				flag = 1
				alpha[i] += 1
			somme = 0
		iter += 1
	return alpha

def predictKernelPerceptron(kp, x, data, target, kernel, h):
	somme = 0
	if(kernel == 1):
		for j in range(len(data)):
			xj = data[j]
			somme += kp[j]*target[j]*noyauGaussien(xj,x,h)
	else:
		for j in range(len(data)):
			xj = data[j]
			somme += kp[j]*target[j]*noyauPolynomial(xj,x,h)
	return np.sign(somme)


def genererDonnees(n):
	x1b = (rand(n)*2-1)/2-0.5
	x2b = (rand(n)*2-1)/2+0.5
	x1r = (rand(n)*2-1)/2+0.5
	x2r = (rand(n)*2-1)/2-0.5
	donnees = []
	for i in range(len(x1b)):
		donnees.append(((x1b[i],x2b[i],1),-1.0))
		donnees.append(((x1r[i],x2r[i],1),1.0))
	return donnees

data = genererDonnees(50)
x = []
y = []
for xl,yl in data:
	x += [xl]
	y += [yl]

ecarttype = np.std(x)
#p = learnKernelPerceptron(x,y,0,3)
#p = learnKernelPerceptron(x,y,1,ecarttype)


datatest = genererDonnees(50)
xtest = []
ytest = []
for xl,yl in datatest:
	xtest += [xl]
	ytest += [yl]
erreur = 0
#for i in range(50):
	#ok = predictKernelPerceptron(p,xtest[i],x,y,1,ecarttype)
	#if(ok != ytest[i]):
		#erreur += 1

print(erreur)

n, data, target, taille = chargementVecteursImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1, 400)
print(data)
print(target)