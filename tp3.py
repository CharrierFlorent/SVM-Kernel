import numpy as np
from pylab import rand
from tp5utils import chargementVecteursImages
from tp5utils import chargementHistogrammesImages
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
def noyauGaussien(x1, x2, sigma):
	npx1 = np.array(x1)
	npx2 = np.array(x2)
	return np.exp(-(pow(np.linalg.norm(npx1-npx2),2))/float(pow(float(sigma),2)))

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

n, x, Y, taille = chargementHistogrammesImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1)
y = np.array(Y)
x2 = np.array(x)
dataset_size = len(x2)
X = x2.reshape(dataset_size, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
ecarttype = np.std(x)
taux = []
r = []
for i in np.arange(1,20,1):
	r += [i]
k = 0
for i in r:
	k+=1
	p = learnKernelPerceptron(X_train,y_train,0,i)


	erreur = 0
	for i in range(len(y_val)):
		ok = predictKernelPerceptron(p,X_val[i],X_val,y_val,1,i)
		if(ok != y_val[i]):
			erreur += 1

	taux += [1-erreur/len(y_val)]
k = 0
for i in r:
	print("d " + str(i) + " score : " + str(taux[k]))
	k += 1
err = []
print(np.argmax(np.array(taux)))
erreur = 0
j = 0
for k in r:
	if k == np.argmax(np.array(taux)):
		p = learnKernelPerceptron(X_train, y_train, 0, k)
		for i in range(len(y_test)):
			ok = predictKernelPerceptron(p, X_test[j], X_train, y_train, 1, i)
			if (ok != y_val[j]):
				erreur += 1

		print("erreur test : ", 1 - erreur / len(y_test))
	j+=1

p2 = learnKernelPerceptron(X_train,y_train,1,ecarttype)



erreur = 0
for i in range(len(y_test)):
	ok = predictKernelPerceptron(p2,X_test[i],X_train,y_train,1,ecarttype)
	if(ok != y_test[i]):
		erreur += 1

print(1-erreur/len(y_test))

