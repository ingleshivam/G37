from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

def train_svm_classifier(X_train, Y_train):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    return classifier

def evaluate_model(classifier, X_train, Y_train, X_test, Y_test):
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = classifier.predict(X_test)
    # X_test_prediction
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    cm_svm = confusion_matrix(Y_test, X_test_prediction)
    return training_data_accuracy, test_data_accuracy,cm_svm,X_test_prediction

