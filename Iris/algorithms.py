from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class algorithm:
    def __init__(self, dataset):
        self.dataset = dataset
        array = self.dataset.values
        self.X = array[:, 0:4]
        self.Y = array[:, 4]
        self.models =[]
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)

    def testModels(self):
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))
        # self.predictions = None
        self.results = []
        self.names = []
        for name, model in self.models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring='accuracy')
            self.results.append(cv_results)
            self.names.append(name)

            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
            self.bestModel()
            self.evaluate()
    def bestModel(self):
        max = 0
        idx = 0
        for i in range(len(self.results)):
            if (max < self.results[i].mean()):
                max = self.results[i].mean()
                idx = i
        print("the best model is: ", self.names[idx])
        model = SVC(gamma='auto')
        model.fit(self.X_train, self.Y_train)
        self.predictions = model.predict(self.X_validation)

    def evaluate(self):
        # print("Accuracy Score: ", accuracy_score(self.Y_validation, self.predictions))
        print("Confusion Matrix: ", confusion_matrix(self.Y_validation, self.predictions))
        print("Classification Report: ", classification_report(self.Y_validation, self.predictions))
