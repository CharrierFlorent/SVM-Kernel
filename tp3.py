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

n, x, Y, taille = chargementHistogrammesImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1)
y = np.array(Y)
x2 = np.array(x)
dataset_size = len(x2)
X = x2.reshape(dataset_size, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ecarttype = np.std(x)
taux = []
r = []
for i in np.arange(0.1,2.0,0.1):
	r += [i]

for i in r:
	print(i)
	p = learnKernelPerceptron(X_train,y_train,1,i)


	erreur = 0
	for i in range(len(y_test)):
		ok = predictKernelPerceptron(p,X_test[i],X_train,y_train,1,i)
		if(ok != y_test[i]):
			erreur += 1

	taux += [1-erreur/len(y_test)]

plt.plot(r, taux)
plt.show()
plt.figure()


p2 = learnKernelPerceptron(X_train,y_train,1,ecarttype)



erreur = 0
for i in range(len(y_test)):
	ok = predictKernelPerceptron(p2,X_test[i],X_train,y_train,1,ecarttype)
	if(ok != y_test[i]):
		erreur += 1

print(1-erreur/len(y_test))



X = []
y = []
X_train = []
X_test = []
y_train = []
y_test = []
j = 0
"""
for i in (range(10,60,5)):
	n, x, Y, taille = chargementVecteursImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1, i)
	y += [np.array(Y)]
	x2 = np.array(x)
	dataset_size = len(x2)
	X += [x2.reshape(dataset_size, -1)]
	X_train1, X_test1, y_train1, y_test1 = train_test_split(X[j], y[j], test_size=0.33, random_state=42)
	X_train += [X_train1]
	X_test += [X_test1]
	y_train += [y_train1]
	y_test += [y_test1]

	j += 1

	print(i)

n, x, Y, taille = chargementVecteursImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1, 225)
y += [np.array(Y)]
x2 = np.array(x)
dataset_size = len(x2)
X += [x2.reshape(dataset_size, -1)]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.33, random_state=42)
X_train += [X_train1]
X_test += [X_test1]
y_train += [y_train1]
y_test += [y_test1]
"""

n, x, Y, taille = chargementHistogrammesImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1)
y = np.array(Y)
x2 = np.array(x)
dataset_size = len(x2)
X = x2.reshape(dataset_size, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
result = []

"""
print("Perceptron fit")
for i in range(0,10):
	clf = Perceptron(tol=1e-3, random_state=0, max_iter = 40, class_weight='balanced')
	result += [cross_val_score(clf, X_train[i], y_train[i], cv=10).mean()]
	clf.fit(X_train[i], y_train[i])
	print("perceptron score : ", clf.score(X_test[i], y_test[i]))
"""


result_tree = []
data = []
data_train = []
print("Tree fit")
clf = Perceptron(random_state=0)
clf.fit(X_train, y_train)
print("Taux de bonne classification : %f" %clf.score(X_test, y_test))
print("crolss val default : ", cross_val_score(clf, X_train, y_train, cv=10).mean())
print("default")
r = []
for i in range(10,110,10):
	r += [i]
for i in range(1000,4000,1000):
	r += [i]
for i in r:
	clf = Perceptron(random_state=0, max_iter=i)
	clf.fit(X_train, y_train)
	data += [clf.score(X_test, y_test)]
	print("Taux de bonne classification : %f " % clf.score(X_test, y_test), i)
	result_tree += [cross_val_score(clf, X_train, y_train, cv=10).mean()]
	print("cross val ", i, cross_val_score(clf, X_train, y_train, cv=10).mean())

plt.plot(r,data)
plt.show()
plt.figure()
plt.plot(r, result_tree)
plt.show()
plt.figure()

print("kppv fit")
#clf = KNeighborsClassifier(n_neighbors=3)
#result += [cross_val_score(clf, X, y, cv=10)]

print("svm")
#clf = LinearSVC(random_state=0, tol=1e-5)
#result += [cross_val_score(clf, X, y, cv=10)]

"""
plt.plot(result[0])
plt.show()
plt.figure()

plt.plot(result[1])
plt.show()
plt.figure()

plt.plot(result[2])
plt.show()
plt.figure()

plt.plot(result[3])
plt.show()
plt.figure()

print("Perceptron")
for i in range(0,10):
	print(result[i])
"""
print("Tree")
#for i in range(0,10):
#	print(result_tree[i])
"""
print("kppv")
print(result[2].mean())

print("svm")
print(result[3].mean())
"""