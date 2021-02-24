

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
import tkinter
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


wine_df_training = pd.read_excel('Wine_Training.xlsx')  #create DataFrame
wine_df_testing = pd.read_excel('Wine_Testing.xlsx')    #create DataFrame


print(wine_df_training.head())
print(wine_df_testing.head())   #data set sample
X_target = pd.DataFrame(wine_df_training['quality']) #target init


#Data description
print(wine_df_training.describe())
print(wine_df_testing.describe())


print(wine_df_training.info())
print(wine_df_testing.info())


distribution = wine_df_testing.hist()

#Creation of correlation matriz
correlation = wine_df_training.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

#NaN values check
print(wine_df_training.isna().sum())
print(wine_df_testing.isna().sum())




#outliers detection
Q1 = wine_df_training.quantile(0.25)
Q3 = wine_df_training.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)
print((wine_df_training < (Q1 - 1.5 * IQR)) |(wine_df_training > (Q3 + 1.5 * IQR)))


#drop target column "quality" and take only the features
features = wine_df_training[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

#X_train as features of ml algorithm
X_train = pd.DataFrame(features) 

#shapes of data set
print(X_train.shape)
print(wine_df_testing.shape)


#GUI creation


#Windo creator
window = tkinter.Tk()
window.title("Wine quality prediction")
window.geometry("300x200")
window.configure(background='black')

#Labels creator
label = tkinter.Label(window, text="Choose prediction algorithm")
label.pack()

#Option creator
OptionList = ['SVM', 'Logistic Reggression', 'K-means', 'Random Forest'] 
variable = tkinter.StringVar(window)
variable.set(OptionList[0])


def on_selection(value):
	global choosen_algo
	choosen_algo = value  
	window.destroy() 

opt = tkinter.OptionMenu(window, variable, *OptionList, command = on_selection)
opt.config(width=10, font=('Helvetica', 10))
opt.pack()
window.mainloop()



Algo = choosen_algo


if Algo == 'SVM':


    preSVM = svm.SVC(kernel = 'rbf', gamma = 0.01 , C = 10 ) #SVM model
    preSVM.fit(X_train, X_target)
    predictionSVM = preSVM.predict(wine_df_testing) 
    print(len(predictionSVM))
    print(predictionSVM)   
    predictionSVM = pd.DataFrame(predictionSVM)
    distribution = predictionSVM.hist()
    plt.ylabel('Count')
    plt.xlabel('Quality')
    
    
elif Algo == 'Logistic Reggression':
    
    
    preLR = LogisticRegression().fit(X_train, X_target)
    predictionLR = preLR.predict(wine_df_testing) 
    print(len(predictionLR))
    print(predictionLR)
    predictionLR = pd.DataFrame(predictionLR) 
    distribution = predictionLR.hist()
    plt.ylabel('Count')
    plt.xlabel('Quality')
    
elif Algo == 'K-means':
    
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train, X_target)
    preKM = kmeans.predict(wine_df_testing) 
    print(len(preKM))
    print(preKM)
    preKM = pd.DataFrame(preKM)
    distribution = preKM.hist()    
    plt.ylabel('Count')
    plt.xlabel('Quality')
    
    
    
elif Algo == 'Random Forest':
    
    
    preRF = RandomForestClassifier(random_state=1).fit(X_train, X_target)
    predictionRF = preRF.predict(wine_df_testing) 
    print(len(predictionRF))
    print(predictionRF)
    predictionRF = pd.DataFrame(predictionRF)
    distribution = predictionRF.hist()
    plt.ylabel('Count')
    plt.xlabel('Quality')
    
