import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


with st.container():
    st.title("Explore different classifiers for different inbuilt sci-kit learn dataset")
    st.sidebar.title("Select the following to get the results")

    dataset_name = st.sidebar.selectbox("Select Dataset",("Iris Dataset","Breast Cancer dataset", "Wine dataset"))
    st.write("""### Dataset : {}""".format(dataset_name))


with st.container():
    Classifier_name = st.sidebar.selectbox("Select Classifer",("KNN","SVM","Random Forest"))
    st.write("""### Classifier : {}""".format(Classifier_name))
    


with st.container():
    st.write(""" #### Dataset """)

    def get_dataset(dataset_name):
        if dataset_name == "Iris dataset":
            data = datasets.load_iris()
        elif dataset_name == "Breast Cancer dataset":
            data = datasets.load_breast_cancer()
        elif dataset_name == "Wine dataset":
            data = datasets.load_wine()
        else:
            data = datasets.load_iris()    

        X = data.data
        y = data.target
        features = data.feature_names
        describe_data = data.DESCR
        return X,y,features,describe_data

    X,y,features,describe_data = get_dataset(dataset_name)
    df = pd.DataFrame(X,columns=features)
    st.dataframe(df)

    st.sidebar.write("Parameters:")

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K",1,15)
            params["K"] = K
        elif clf_name == "SVM":
            C = st.sidebar.slider("C",0.01,10.0)    
            params["C"] = C
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth", 2,15)
            n_estimators = st.sidebar.slider("n_estimators",1,100)    
            params["Max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        return params

    params = add_parameter_ui(Classifier_name)

    def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "SVM":
            clf = SVC(C = params["C"])
        else:
            # max_depth = st.sidebar.slider("max_depth",2,15)
            # n_estimators = st.sidebar.slider("n_estimators",1,100)
            clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["Max_depth"], random_state=1234)
        return clf

    clf = get_classifier(Classifier_name,params)

    # Classification 



with st.container():
    st.write(describe_data)

st.sidebar.caption(" '>>>' scroll down to get the accuracy score for the classifier and its parameters")

with st.container():
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    report = confusion_matrix(y_test,y_pred)
    st.write(""" ## Score of Classifier """)
    st.write(""" ##### Scroll the select box in side bar to select parmeters""")
    st.write(""" ##### For given Classifier {} Parameters: """.format(Classifier_name))
    st.write(""" ##### Classifier Accuracy Score = {}""".format(acc)) 
    st.write(""" ##### Confusion Matrix: 
                    matrix values:
                    True Positives: {}
                    True Negitives: {}
                    False Positives: {}
                    False Negatives: {} """.format(report[1,1],report[0,0],report[1,0],report[0,1]))  
    
    

with st.container():
    #plot
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    fig = plt.figure()
    plt.scatter(x1,x2,c = y, alpha = 0.8,cmap = "viridis")
    plt.title("Visulaization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()

    #splt.show()
    st.write(""" ## Visualisation of different classes in the Dataset""")
    st.pyplot(fig)

with st.container():
    st.caption("Credits: sci-kit learn for Datasets information")

