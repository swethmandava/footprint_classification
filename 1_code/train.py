import cv2
from scipy.fftpack import dct
import numpy
import glob
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Extracts zig zag manner of image compression that seems to be most efficient
#Not used, dead code. Feel free to remove, store for future use.
# def extract_zig_zag(dct):
# 	m1 = dct.shape[0]
# 	n1 = dct.shape[1]
# 	arr = numpy.zeros([1, m1*n1])
# 	count = 0
# 	for s in range(m1):
# 		if(s%2 == 0):
# 			for m in range(s, -1, -1):
# 				arr[0][count] = dct[m][s-m]
# 				count = count+1
# 		else:
# 			for m in range(s):
# 				arr[0][count] = dct[m][s-m]
# 				count = count + 1
# 	return dct.flatten()


#Extracts Discrete Cosine Transform of blurred image
def extract_dct(filename):
	im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	im = numpy.float32(im)/255.0
	im = cv2.GaussianBlur(im, (5,5), 0)
	c = dct(numpy.transpose(im))
	dcta = dct(numpy.transpose(c))
	dcta = dcta[:40, :40]
	dcta = cv2.normalize(dcta, 0, 255, cv2.NORM_MINMAX)
	# dcta = extract_zig_zag(dcta)
	return dcta

#Computes features and saves the Best estimated SVM model with rbf kernel
if __name__ == '__main__':
	X_train = numpy.zeros([10000, 1600])
	Y_train = numpy.zeros([10000, 1])
	files = glob.glob("1/train/right/*.png")
	right_files = len(files)
	for i, file in enumerate(files):
		print file
		dct_feature = extract_dct(file)
		X_train[i] = dct_feature
		Y_train[i] = 1
	files = glob.glob("1/train/left/*.png")
	for i, file in enumerate(files):
		print file
		dct_feature = extract_dct(file)
		X_train[i+right_files] = dct_feature
		Y_train[i+right_files] = 2

	numpy.save('X_train_1', X_train)
	Y_train = Y_train.reshape(10000,)

	print "Completed Data Extraction"

	pca = PCA(n_components=30).fit(X_train)
	XTrain_PCA = pca.transform(X_train)
	numpy.save('X_train_pca_1', XTrain_PCA)
	numpy.save('Y_train_1', Y_train)
	filename = 'pca_1.sav'
	joblib.dump(pca, filename)

	print "Computed PCA components"
	
	param_grid = {'C': [10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
			  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
	clf = clf.fit(XTrain_PCA, Y_train)
	filename = 'finalized_model_1.sav'

	print("Best estimator found by grid search:")
	print(clf.best_estimator_)
	
	joblib.dump(clf, filename)
	print "Fit a Model"