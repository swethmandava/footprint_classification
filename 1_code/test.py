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
import train

#Load the saved model and classify the test images
if __name__ == '__main__':
	pca = joblib.load('pca_1.sav')
	clf = joblib.load("finalized_model_1.sav")
	X_test = numpy.zeros([10000,1600])
	files = glob.glob("1/test/*.png")
	files.sort(key=lambda x:'{0:0>1000}'.format(x).lower())
	for i, file in enumerate(files):
		print file
		dct_feature = extract_dct(file)
		X_test[i] = dct_feature

	numpy.save('X_test_1', X_test)
	XTest_PCA = pca.transform(X_test)
	numpy.save('X_test_pca_1', XTest_PCA)
	y_pred = clf.predict(XTest_PCA)
	numpy.savetxt("1.csv", y_pred, delimiter=",")

