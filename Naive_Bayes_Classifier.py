import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
import numpy as np

def print_formated_number(number):
    print("{:.1f}%".format(number*100))

if __name__ == "__main__":
    #Enter the path to get data
    file_path = ""
    column_names = ['party', 'handicapped-infants', 'water-project-cost-sharing', 
                'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 
                'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 
                'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 
                'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    
    data = pd.read_csv(file_path,header=None,na_values='?', names=column_names)

    data = data.replace({'y': 1, 'n': 0})
    data.fillna(2,inplace=True)
    data['party'] = data['party'].map({'democrat': 0, 'republican': 1})
    
    X = data.drop('party', axis=1)
    y = data['party']

    kfold = KFold(n_splits=10)
    accuracies = []

    for train_index, test_index in kfold.split(X,y):
        #Split into test and training sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #this uses Laplace smoothing and log probabilities
        classifier = CategoricalNB()

        #train the classifier
        classifier.fit(X_train, y_train)

        #make predictions
        predictions = classifier.predict(X_test)

        #Calculate accuracy
        accuracies.append(accuracy_score(y_test, predictions))

    average_accuracy = np.mean(accuracies)

    print("Accuracies:")
    for accuracy in accuracies:
        print_formated_number(accuracy)

    print("Average accuracy: ",end="")
    print_formated_number(average_accuracy)


