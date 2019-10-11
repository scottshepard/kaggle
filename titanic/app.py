import pandas as pd
from datetime import date

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def reports(modelname, ytrain, ytrain_pred, ytest, ytest_pred):
    '''
    Print the classification report and confusion matrix for both train and test
    '''
    print("{0} Train Accuracy Score: {1}".format(modelname, (ytrain == ytrain_pred).mean()))
    print("\n{0} Train Confusion Matrix".format(modelname))
    print(metrics.confusion_matrix(ytrain, ytrain_pred))
    print("\n{0} Train Classification Report".format(modelname))
    print(metrics.classification_report(ytrain, ytrain_pred))
    print('-' * 53 + '\n')
    print("{0} Test Accuracy Score: {1}".format(modelname, (ytest == ytest_pred).mean()))
    print("\n{0} Test Confusion Matrix".format(modelname))
    print(metrics.confusion_matrix(ytest, ytest_pred))
    print("\n{0} Test Classification Report".format(modelname))
    print(metrics.classification_report(ytest, ytest_pred))


df = pd.read_csv('data/train.csv')

df['Age'] = df.Age.fillna(-1)
df['Embarked'] = df.Embarked.fillna('NA')

df = df.dropna()
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])
df = pd.get_dummies(df)

y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)

reports('Default Random Forest', y_train, y_train_pred, y_val, y_val_pred)

# Test predictions
test = pd.read_csv('data/test.csv')
test['Age'] = test.Age.fillna(-1)
test['Embarked'] = test.Embarked.fillna('NA')
test['Fare'] = test.Fare.fillna(X_train.Fare.median())

ids = test['PassengerId']
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test = pd.get_dummies(test)
test['Embarked_NA'] = 0

y_test_pred = clf.predict(test)

out = pd.DataFrame({'PassengerId': ids, 'Survived': y_test_pred})
out.to_csv('Titanic_RandomForest_' + str(date.today()) + '.csv', index=False)
