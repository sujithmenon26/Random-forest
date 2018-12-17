#random forest using iris datasets

# Loading the library with the iris datasets
from sklearn.datasets import load_iris

#loading the scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

#loading the matplotlib library
import matplotlib.pyplot as plt

#loading numpy
import numpy as np


#loading pandas
import pandas as pd

#setting random seed
np.random.seed(0)

#creating an object iris 
iris = load_iris()

#creating a dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#viewing the top 5 rows
df.head()


#adding a new column 
df["species"]=pd.Categorical.from_codes(iris.target,iris.target_names)

df.head()

#creating the test and train data
df["is_train"]= np.random.uniform(0,1, len(df))<= .75

df.head()


#creating dataframe with test rows and training rows
train, test = df [df["is_train"]==True],df[df["is_train"]==False]

#shows number of observations 
print("number of observations in the training data:", len(train))
print("number of observations in the test data:", len(test))

#create a list of the features column's names
features = df.columns[:4]

#To view 
features


#converting each species name into digits
y=pd.factorize(train["species"])[0]

#to view
y

#Creating a random forest classifier
clf=RandomForestClassifier(n_jobs=2, random_state=0)

#training the classifier
clf.fit(train[features], y)



test[features]

#applying the trained classifier to the test
clf.predict(test[features])


#view the predicted probabilities
clf.predict_proba(test[features])[0:20]


#mapping names for the plants with predicted class
preds = iris.target_names[clf.predict(test[features])]

#To view
preds[0:20]


#view the actual species
test["species"].head()



#creating matrix
pd.crosstab(test["species"], preds, rownames=["Actual Species"], colnames=["Predicted Species"])

#ploting onto a graph
plt.plot(preds)
plt.show()





