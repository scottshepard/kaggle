from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from datetime import date


df_raw = pd.read_csv('data/train.csv')
df = df_raw.copy()

df['Age'] = df.Age.fillna(-1)
df['Embarked'] = df.Embarked.fillna('NA')

df = df.dropna()
y = df['Survived']
df = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
X = pd.get_dummies(df)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

scaler = StandardScaler()
X_scale = scaler.fit_transform(X_train)

model =  Sequential()
model.add(Dense(3, input_dim=X_scale.shape[1], activation='relu'))
model.add(Dense(3, input_dim=X_scale.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.utils import plot_model
plot_model(model, to_file='model1.png',show_shapes=True,show_layer_names=True)

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=30, verbose=0)
prob1 = model.predict(X_val)
print(model.metrics_names)
print(model.evaluate(X_val, y_val,verbose=0))

threshold = 0.6
pred1 = (prob1 > threshold).flatten()

print(metrics.confusion_matrix(y_val, pred1))

# Test predictions
test = pd.read_csv('data/test.csv')
test['Age'] = test.Age.fillna(-1)
test['Embarked'] = test.Embarked.fillna('NA')
test['Fare'] = test.Fare.fillna(X_train.Fare.median())

ids = test['PassengerId']
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test = pd.get_dummies(test)
test['Embarked_NA'] = 0

X_test = scaler.transform(test)

y_test_prob = model.predict(X_test)
y_test_pred = pd.Series((y_test_prob.flatten() > threshold))

y_test_pred.value_counts()

y_test_pred = y_test_pred.astype('int')

out = pd.DataFrame({'PassengerId': ids, 'Survived': y_test_pred})
out.to_csv('Titanic_DenseNN_' + str(date.today()) + '.csv', index=False)
# Kaggle score of 0.72248
print("Done!")
