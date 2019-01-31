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
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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
#n, x, Y, taille = chargementVecteursImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1, 50)
n, x, Y, taille = chargementHistogrammesImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1)
y = np.array(Y)
x2 = np.array(x)
dataset_size = len(x2)
X = x2.reshape(dataset_size, -1)

X_histo = np.load('./tp3-M1info2019/BinTest/tHisto.npy')
clf = KNeighborsClassifier(n_neighbors=28)
clf.fit(X,y)
y_pred = clf.predict(X_histo)
print(y_pred)

"""
result = []
score = []

n, x, Y, taille = chargementVecteursImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1, 100)
#n, x, Y, taille = chargementHistogrammesImages("./tp3-M1info2019/Data/Mer", "./tp3-M1info2019/Data/Ailleurs", 1, -1)
y = np.array(Y)
x2 = np.array(x)
dataset_size = len(x2)
X = x2.reshape(dataset_size, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf = Perceptron(max_iter=25)
clf.fit(X_train,y_train)


clf_list_AD = []
clf_list_KNN = []
clf_list_SVM = []
clf_list_perceptron = []


clf_cv_AD = []
clf_cv_KNN = []
clf_cv_SVM = []
clf_cv_perceptron = []

clf_val_AD = []
clf_val_KNN = []
clf_val_SVM = []
clf_val_perceptron = []

clf_train_AD = []
clf_train_KNN = []
clf_train_SVM = []
clf_train_perceptron = []
knn = 0
if knn == 1:
	for i in range(1,30):
		clf_list_KNN += [KNeighborsClassifier(n_neighbors=i)]
		clf_cv_KNN += [cross_val_score(clf_list_KNN[i-1],X_train, y_train, cv=10).mean()]
		clf_list_KNN[i-1].fit(X_train, y_train)
		clf_train_KNN += [clf_list_KNN[i - 1].score(X_train, y_train)]

	plt.plot(range(1,30), clf_train_KNN, color = 'red')
	plt.plot(range(1,30), clf_cv_KNN, color = 'blue')
	plt.show()
	plt.figure()
	print("sur train : " + str(np.argmax(clf_train_KNN)) + " valeur : " + str(clf_train_KNN[np.argmax(clf_train_KNN)]))
	print("sur cv : " + str(np.argmax(clf_cv_KNN)) + " valeur : " + str(clf_cv_KNN[np.argmax(clf_cv_KNN)]))
	print("performance : " + str(clf_list_KNN[np.argmax(clf_cv_KNN)].score(X_test,y_test)))

AD =0
if AD ==1:
	t = 0
	for i in ('gini', 'entropy'):
		for j in range(1,15):
			for k in range(2,15):
				clf_list_AD += [RandomForestClassifier(criterion=i,max_depth=j,min_samples_split=k)]
				if(t == 78):
					print(i,j,k)
				clf_cv_AD += [cross_val_score(clf_list_AD[t],X_train, y_train, cv=10).mean()]
				clf_list_AD[t].fit(X_train, y_train)
				clf_train_AD += [clf_list_AD[t].score(X_train, y_train)]
				t += 1

	plt.plot(range(len(clf_cv_AD)), clf_train_AD, color = 'red')
	plt.plot(range(len(clf_cv_AD)), clf_cv_AD, color = 'blue')
	plt.show()
	plt.figure()
	print("sur train : " + str(np.argmax(clf_train_AD)) + " valeur : " + str(clf_train_AD[np.argmax(clf_train_AD)]))
	print("sur cv : " + str(np.argmax(clf_cv_AD)) + " valeur : " + str(clf_cv_AD[np.argmax(clf_cv_AD)]))
	t = 0
	for i in ('gini', 'entropy'):
		for j in range(1,15):
			for k in range(2,15):
				if t == np.argmax(clf_cv_AD):
					print("performance : " + str(clf_list_AD[np.argmax(clf_cv_AD)].score(X_test,y_test)))
				t += 1

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 100,200,300,500, 10000]}]
tree_parameters = {'criterion' : ['entropy', 'gini'],'max_depth': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'max_features': [1, 2, 3, 4], 'min_samples_split': [2, 3, 4, 5, 6, 7], 'min_samples_leaf': [2, 3, 4, 5, 6, 7]}
neighbor_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25,26,27,28,29, 30,31,32,33,34, 35]}
perceptron_parameters = {'max_iter': [5,10,15,20,30,50,80,100,200,500,1000,10000]}
scores = ['precision', 'recall']
for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()
	clf = GridSearchCV(Perceptron(), perceptron_parameters, cv=10,
					   scoring='%s_macro' % score)
	#clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
	#				   scoring='%s_macro' % score)
	#clf = GridSearchCV(KNeighborsClassifier(), neighbor_parameters, cv=10,
	#				   scoring='%s_macro' % score)
	#clf = GridSearchCV(DecisionTreeClassifier(random_state=5), tree_parameters, cv=10,
	#				   scoring='%s_macro' % score)
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))
	print()

	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))
	print()
"""