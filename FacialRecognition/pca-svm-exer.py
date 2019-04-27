import os
from gzip import GzipFile

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import urllib.request as urllib
import tarfile

url = "https://downloads.sourceforge.net/project/scikit-learn/data/lfw_preprocessed.tar.gz"
archive_name = "lfw_preprocessed.tar.gz"
folder_name = "lfw_preprocessed"


def show_explained_variance_ratio(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

def load_data():
    if not os.path.exists(folder_name):
        if not os.path.exists(archive_name):
            print ("Downloading data, please Wait (58.8MB)...")
            print (url)
            opener = urllib.urlopen(url)
            open(archive_name, 'wb').write(opener.read())


        print ("Decompressiong the archive: {} ".format(archive_name))
        tarfile.open(archive_name, "r:gz").extractall()
        
    ################################################################################

    faces_filename = os.path.join(folder_name, "faces.npy.gz")
    filenames_filename = os.path.join(folder_name, "face_filenames.txt")

    faces = np.load(GzipFile(faces_filename))
    face_filenames = [l.strip() for l in open(filenames_filename).readlines()]

    return faces, face_filenames

def train_data(y_train,X_test_pca,y_test,selected_target):
    ################################################################################
    # Train a SVM classification model

    print ("Fitting the classifier to the training set")
    param_grid = {
    'gamma': [0.001],
    'kernel': ["linear"]
    }
    clf = GridSearchCV(SVC(), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print ("Best estimator found by grid search:")
    print (clf.best_estimator_)

    y_pred = clf.predict(X_test_pca)
    print (classification_report(y_test, y_pred, labels=selected_target))
    # print (confusion_matrix(y_test, y_pred, labels=selected_target))
    print("Accuracy: ", accuracy_score(y_test,y_pred))


    return y_pred

def scale_data(train_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data)

if __name__ == "__main__":
    
    number_of_components = [100,200,300,400,500,600,700,800,900,1000]

    for i in number_of_components:
        faces,face_filenames = load_data()
        faces -= faces.mean(axis=1)[:, np.newaxis]
        categories = np.array([f.rsplit('_', 1)[0] for f in face_filenames])
        category_names = np.unique(categories)
        target = np.searchsorted(category_names, categories)
        selected_target = np.argsort(np.bincount(target))[-5:]
        mask = np.array([item in selected_target for item in target])

        X = faces[mask]
        y = target[mask]

        n_samples, n_features = X.shape

        show_explained_variance_ratio(X)

        # Scale the data
        X = scale_data(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        print("Extracting the top {} eigenfaces".format(i))
        pca = PCA(n_components=i).fit(X_train)
        eigenfaces = pca.components_.T.reshape((i, 64, 64))
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        y_pred = train_data(y_train,X_test_pca,y_test,selected_target)