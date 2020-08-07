# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:00:54 2020

@author: Jatin
"""
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlxtend
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('asteroid.csv')
ast=df.head(100)

print(df.shape)
print(df.describe())

#df = df.sample(frac=0.1, random_state = 42)
df.drop(['name','equinox','pdes','id','prefix','spkid','full_name'],axis=1,inplace=True)


print(df.pha.value_counts()) #how many belong to each class of target variable

threat=df[df.pha=='Y']
non_threat=df[df.pha=='N']
outlier_percentage=(df.pha.value_counts()[1]/df.pha.value_counts()[0])*100
print('Potential threat asteroids are: %.3f%%'%outlier_percentage)
print('Threat asteroids: ',len(threat))
print('Non-Threat asteroids: ',len(non_threat))





null_cutoff=0.5


def numericalCategoricalSplit(df):
    numerical_features=df.select_dtypes(exclude=['object']).columns
    categorical_features=df.select_dtypes(include=['object']).columns
    numerical_data=df[numerical_features]
    categorical_data=df[categorical_features]
    return(numerical_data,categorical_data)
numerical=numericalCategoricalSplit(df)[0]
categorical=numericalCategoricalSplit(df)[1]


def nullFind(df):
    null_numerical=pd.isnull(df).sum().sort_values(ascending=False)
    #null_numerical=null_numerical[null_numerical>=0]
    null_categorical=pd.isnull(df).sum().sort_values(ascending=False)
   # null_categorical=null_categorical[null_categorical>=0]
    return(null_numerical,null_categorical)
null_numerical=nullFind(numerical)[0]
null_categorical=nullFind(categorical)[1]


null=pd.concat([null_numerical,null_categorical])
null_df=pd.DataFrame({'Null_in_Data':null}).sort_values(by=['Null_in_Data'],ascending=False)
null_df_many=(null_df.loc[(null_df.Null_in_Data>null_cutoff*len(df))])
null_df_few=(null_df.loc[(null_df.Null_in_Data!=0)&(null_df.Null_in_Data<null_cutoff*len(df))])


many_null_col_list=null_df_many.index
few_null_col_list=null_df_few.index

#remove many null columns
df_wo_null=df.drop(many_null_col_list,axis=1)



def removeNullRows(df):
    for col in few_null_col_list:
        df=df[df[col].notnull()]
    return(df)
    
df_wo_null=(removeNullRows(df_wo_null)) 



#dividing the X and the y
X=df_wo_null.drop(['pha'], axis=1)
y=df_wo_null.pha

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)




from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions as plot_dr 

logreg=LogisticRegression()
SVM=SVC()
knn=KNeighborsClassifier()
gnb=GaussianNB()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)



scaler=StandardScaler()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()

features=X_train.columns.tolist()

X_train[categorical.columns.drop('pha')]=X_train[categorical.columns.drop('pha')].apply(le.fit_transform)
X_test[categorical.columns.drop('pha')]=X_test[categorical.columns.drop('pha')].apply(le.fit_transform)
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)



X_train_scaled=scaler.fit_transform(X_train) 
X_test_scaled=scaler.fit_transform(X_test) 




#feature selection
start_time = timeit.default_timer()
mod=etree
# fit the model
mod.fit(X_train_scaled, y_train)
# get importance
importance = mod.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
df_importance=pd.DataFrame({'importance':importance},index=features)
df_importance.plot(kind='barh')
#plt.bar([x for x in range(len(importance))], importance)
elapsed = timeit.default_timer() - start_time
print('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1]) 
n_top_features=20
top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=scaler.fit_transform(X_train_sfs)
X_test_sfs_scaled=scaler.fit_transform(X_test_sfs)


#print(knn.get_params().keys())


#models=[knn,etree,SVM]
#param_distributions=[{'n_neighbors':[5,10]},{'criterion':['gini', 'entropy'],'n_estimators':[100,200]},{'kernel':['rbf','linear'],'C':[0.1,1],'gamma':[0.1,0.01]}]

models=[etree]
param_distributions=[{'criterion':['gini', 'entropy'],'n_estimators':[100,200,300]}]

for model in models:
    rand=RandomizedSearchCV(model,param_distributions=param_distributions[models.index(model)],cv=3,scoring='accuracy', n_jobs=-1, random_state=42,verbose=10)
    rand.fit(X_train_sfs_scaled,y_train)
    print(rand.best_params_,rand.best_score_) 
    

 

    
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
from sklearn.metrics import  classification_report,accuracy_score

def yellowbrick_visualizations(model,classes,X_tr,y_tr,X_te,y_te):
    visualizer=ConfusionMatrix(model,classes=classes)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    
    visualizer = ClassificationReport(model, classes=classes, support=True)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    
    visualizer = ROCAUC(model, classes=classes)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    

classes=['Non-Hazardous','Hazardous']
model=ExtraTreesClassifier(n_estimators=300,criterion='entropy',random_state=42)
model.fit(X_train_sfs_scaled,y_train)


y_pred = model.predict(X_test_sfs_scaled)

yellowbrick_visualizations(model,classes,X_train_sfs_scaled,y_train,X_test_sfs_scaled,y_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print(np.bincount(y_pred))

print(np.bincount(y_train))



from imblearn.over_sampling import SMOTE,RandomOverSampler,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
smt = SMOTE()
nr = NearMiss()
bsmt=BorderlineSMOTE(random_state=42)
ros=RandomOverSampler(random_state=42)
rus=RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = rus.fit_sample(X_train_sfs_scaled, y_train)
print(np.bincount(y_train_bal))


model_bal=model
model_bal.fit(X_train_bal, y_train_bal)

y_pred = model_bal.predict(X_test_sfs_scaled)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

yellowbrick_visualizations(model_bal,classes,X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

#Plot decision region
def plot_classification(model,X_t,y_t):
    clf=model
    pca = PCA(n_components = 2)
    X_t2 = pca.fit_transform(X_t)
    clf.fit(X_t2,np.array(y_t))
    plot_dr(X_t2, np.array(y_t), clf=clf, legend=2)

plot_classification(model_bal,X_test_sfs_scaled,y_test)