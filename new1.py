import streamlit as st
from PIL import Image
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mahotas
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd




global X1, Y1
global X2, Y2

global scaler1, scaler2
global svm1, svm2
global X_train1, X_test1, y_train1, y_test1
global X_train2, X_test2, y_train2, y_test2

bins = 8

svm_acc = []
knn_acc = []
lr_acc = []
rf_acc = []
precision1=[]
precision2=[]
recall1=[]
recall2=[]
f1_score1=[]
f1_score2=[]

picture = Image.open("vitic.png")
st.image(picture,width=600)
#st.title("Viticulture Solutions for you 🍇🍇")


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def uploadDataset():
    global scaler1, scaler2

    global X1, Y1
    global X2, Y2

    ds1 = np.load(
        'X1feature.npy')
    ds1_label = np.load(
        'dtst1_label.npy')
    ds2 = np.load(
        'X2feature.npy')
    ds2_label = np.load(
        'dtst2_label.npy')

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(len(ds1)):
        Y1.append(ds1_label[i])
        X1.append(ds1[i])
    Y1 = np.asarray(Y1)
    X1 = np.asarray(X1)

    le = LabelEncoder()
    Y1 = le.fit_transform(Y1)

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    X1 = scaler1.fit_transform(X1)

    print(X1.shape)
    print(Y1.shape)

    for i in range(len(ds2)):
        Y2.append(ds2_label[i])
        X2.append(ds2[i])
    Y2 = np.asarray(Y2)
    X2 = np.asarray(X2)

    le = LabelEncoder()
    Y2 = le.fit_transform(Y2)
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    X2 = scaler2.fit_transform(X2)

    print(X2.shape)
    print(Y2.shape)


def runSVM():
    global svm1, svm2
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2

    global X1, Y1
    global X2, Y2

    global svm_acc
    svm_acc.clear()

    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    indices = np.arange(X2.shape[0])
    np.random.shuffle(indices)
    X2 = X2[indices]
    Y2 = Y2[indices]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.10)

    svm1 = SVC(C=10)
    svm1.fit(X_train1, y_train1)
    predict = svm1.predict(X_test1)
    y_pred = accuracy_score(y_test1, predict) * 100
    svm_acc.append(y_pred)

    svm2 = SVC(C=10)
    svm2.fit(X_train2, y_train2)
    predict1 = svm2.predict(X_test2)
    y_pred = accuracy_score(y_test2, predict1) * 100

    svm_acc.append(y_pred)
    report1 = classification_report(y_test1, predict, digits=4, output_dict=True)

    weighted_avg1 = report1['weighted avg']

    weighted_precision1 = weighted_avg1['precision']

    weighted_recall1 = weighted_avg1['recall']
    weighted_f1score1 = weighted_avg1['f1-score']
    precision1.append(weighted_precision1 * 100)
    recall1.append(weighted_recall1 * 100)
    f1_score1.append(weighted_f1score1 * 100)
    #print(f'Weighted Avg for dataset1- Precision: {weighted_precision1 * 100:.4f}, Recall: {weighted_recall1 * 100:.4f}, F1-score: {weighted_f1score1 * 100:.4f}')

    report2 = classification_report(y_test2, predict1, digits=4, output_dict=True)

    # Extract the weighted average values
    weighted_avg2 = report2['weighted avg']
    weighted_precision2 = weighted_avg2['precision']

    weighted_recall2 = weighted_avg2['recall']

    weighted_f1score2 = weighted_avg2['f1-score']
    precision2.append(weighted_precision2 * 100)
    recall2.append(weighted_recall2 * 100)
    f1_score2.append(weighted_f1score2 * 100)
    #print(f'Weighted Avg for dataset2 - Precision: {weighted_precision2 * 100:.4f}, Recall: {weighted_recall2 * 100:.4f}, F1-score: {weighted_f1score2 * 100:.4f}')


# print(svm_acc)


def runKNN():
    global X1, Y1
    global X2, Y2
    global knn_acc
    global knn1, knn2

    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    knn_acc.clear()
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    indices = np.arange(X2.shape[0])
    np.random.shuffle(indices)
    X2 = X2[indices]
    Y2 = Y2[indices]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.10)
    # checking


    # checking
    knn1 = KNeighborsClassifier(n_neighbors=11)
    knn1.fit(X1, Y1)
    predict = knn1.predict(X_test1)
    y_pred = accuracy_score(y_test1, predict) * 100
    knn_acc.append(y_pred)

    knn2 = KNeighborsClassifier(n_neighbors=6)
    knn2.fit(X2, Y2)
    predict1 = knn2.predict(X_test2)
    y_pred = accuracy_score(y_test2, predict1) * 100
    knn_acc.append(y_pred)
    report1 = classification_report(y_test1, predict, digits=4, output_dict=True)

    weighted_avg1 = report1['weighted avg']
    weighted_precision1 = weighted_avg1['precision']
    weighted_recall1 = weighted_avg1['recall']
    weighted_f1score1 = weighted_avg1['f1-score']
    precision1.append(weighted_precision1 * 100)
    recall1.append(weighted_recall1 * 100)
    f1_score1.append(weighted_f1score1 * 100)
    #print(f'Weighted Avg for dataset1- Precision: {weighted_precision1 * 100:.4f}, Recall: {weighted_recall1 * 100:.4f}, F1-score: {weighted_f1score1 * 100:.4f}')

    report2 = classification_report(y_test2, predict1, digits=4, output_dict=True)

    # Extract the weighted average values
    weighted_avg2 = report2['weighted avg']
    weighted_precision2 = weighted_avg2['precision']
    weighted_recall2 = weighted_avg2['recall']
    weighted_f1score2 = weighted_avg2['f1-score']
    precision2.append(weighted_precision2 * 100)
    recall2.append(weighted_recall2 * 100)
    f1_score2.append(weighted_f1score2 * 100)
    #print(f'Weighted Avg for dataset2 - Precision: {weighted_precision2 * 100:.4f}, Recall: {weighted_recall2 * 100:.4f}, F1-score: {weighted_f1score2 * 100:.4f}')


# logistic regression
def runlogistic():
    global X1, Y1
    global X2, Y2
    global lr_acc
    global lr1, lr2

    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    lr_acc.clear()
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    indices = np.arange(X2.shape[0])
    np.random.shuffle(indices)
    X2 = X2[indices]
    Y2 = Y2[indices]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.10)
    # checking

    # Define the logistic regression model


    # checking

    lr1 = LogisticRegression(max_iter=100)
    lr1.fit(X1, Y1)
    predict1 = lr1.predict(X_test1)
    y_pred = accuracy_score(y_test1, predict1) * 100
    lr_acc.append(y_pred)

    lr2 = LogisticRegression(max_iter=100)
    lr2.fit(X2, Y2)
    predict = lr2.predict(X_test2)
    y_pred = accuracy_score(y_test2, predict) * 100
    lr_acc.append(y_pred)
    report1 = classification_report(y_test1, predict1, digits=4, output_dict=True)

    weighted_avg1 = report1['weighted avg']
    weighted_precision1 = weighted_avg1['precision']
    weighted_recall1 = weighted_avg1['recall']
    weighted_f1score1 = weighted_avg1['f1-score']
    precision1.append(weighted_precision1 * 100)
    recall1.append(weighted_recall1 * 100)
    f1_score1.append(weighted_f1score1 * 100)
    #print(f'Weighted Avg for dataset1- Precision: {weighted_precision1 * 100:.4f}, Recall: {weighted_recall1 * 100:.4f}, F1-score: {weighted_f1score1 * 100:.4f}')

    report2 = classification_report(y_test2, predict, digits=4, output_dict=True)

    # Extract the weighted average values
    weighted_avg2 = report2['weighted avg']
    weighted_precision2 = weighted_avg2['precision']
    weighted_recall2 = weighted_avg2['recall']
    weighted_f1score2 = weighted_avg2['f1-score']
    precision2.append(weighted_precision2 * 100)
    recall2.append(weighted_recall2 * 100)
    f1_score2.append(weighted_f1score2 * 100)
    #print(f'Weighted Avg for dataset2 - Precision: {weighted_precision2 * 100:.4f}, Recall: {weighted_recall2 * 100:.4f}, F1-score: {weighted_f1score2 * 100:.4f}')


# randomforest
def runrandomforest():
    global X1, Y1
    global X2, Y2
    global rf_acc
    global rf1, rf2

    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    rf_acc.clear()
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    indices = np.arange(X2.shape[0])
    np.random.shuffle(indices)
    X2 = X2[indices]
    Y2 = Y2[indices]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.10)



    rf1 = RandomForestClassifier(n_estimators=6)
    rf1.fit(X1, Y1)
    predict1 = rf1.predict(X_test1)
    y_pred = accuracy_score(y_test1, predict1) * 100
    rf_acc.append(y_pred)



    rf2 = RandomForestClassifier(n_estimators=5)
    rf2.fit(X2, Y2)
    predict = rf2.predict(X_test2)
    y_pred = accuracy_score(y_test2, predict) * 100
    rf_acc.append(y_pred)
    report1 = classification_report(y_test1, predict1, digits=4, output_dict=True)

    weighted_avg1 = report1['weighted avg']
    weighted_precision1 = weighted_avg1['precision']
    weighted_recall1 = weighted_avg1['recall']
    weighted_f1score1 =weighted_avg1['f1-score']
    precision1.append(weighted_precision1 * 100)
    recall1.append(weighted_recall1 * 100)
    f1_score1.append(weighted_f1score1 * 100)
    #print(f'Weighted Avg for dataset1- Precision: {weighted_precision1 * 100:.4f}, Recall: {weighted_recall1 * 100:.4f}, F1-score: {weighted_f1score1 * 100:.4f}')

    report2 = classification_report(y_test2, predict, digits=4, output_dict=True)

    # Extract the weighted average values
    weighted_avg2 = report2['weighted avg']
    weighted_precision2 = weighted_avg2['precision']
    weighted_recall2 =weighted_avg2['recall']
    weighted_f1score2 = weighted_avg2['f1-score']
    precision2.append(weighted_precision2 * 100)
    recall2.append(weighted_recall2 * 100)
    f1_score2.append(weighted_f1score2 * 100)
    #print(f'Weighted Avg for dataset2 - Precision: {weighted_precision2 * 100:.4f}, Recall: {weighted_recall2 * 100:.4f}, F1-score: {weighted_f1score2 * 100:.4f}')


def main():
    st.sidebar.title("🎓 Computer Vision and Machine Learning for Efficient Viticulture Applications ")
    menu = ["🏡 Home", "📷 Upload Image","📋 Model Analysis"]

    choice = st.sidebar.selectbox("📜 Menu", menu)
    if choice == "🏡 Home":
        st.subheader("Home")
        st.write("Viticulture is the science, study, and production of grapes. It is a branch of horticulture that focuses on cultivating grapevines, managing vineyards, and producing grapes for winemaking, juice production, and consumption as fresh fruit. viticulture involves a wide range of activities, including soil management, irrigation, pruning, trellising, pest and disease control, harvesting, and processing. Viticulturists also study grape varieties and their characteristics")
        picture3 = Image.open("bunch.jpg")
        st.image(picture3, use_column_width=True)

        st.write("Today, viticulturists use a combination of traditional and modern techniques to produce high-quality grapes and wines that are enjoyed around the world.")
        st.write("Machine learning and artificial intelligence are the technologies used to analyze large amounts of data and predict vineyard conditions and grape yields. This information can be used to optimize vineyard management practices and improve grape quality.")

    if choice == "📷 Upload Image":

        uploaded_file = st.file_uploader(label="Upload an image", type=["jpg", "jpeg", "png"])

        # Check if an image has been uploaded
        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded image", use_column_width=True)

            # Add a button to show the uploaded image
            if st.button("🔮🔍 Predict "):
                st.header("Predict")
                img1 = img
                # Path to YOLO model weights
                

                # Load YOLO model
                model = YOLO('last.pt')

                # Detection threshold
                threshold = 0.4

                # Class name dictionary
                class_name_dict = {0: 'Grape'}

                # Perform object detection
                results = model(img)[0]
                img = np.array(img)
                flag = 0

                # Draw bounding boxes and class labels on output image
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result

                    if score > threshold:
                        flag = 1
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        cv2.putText(img, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Save output image
                cv2.imwrite('output.jpg', img)

                if flag == 0:
                    cv2.putText(img, 'Grape not found!! Cant Predict stages', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2)
                    st.image(img, caption="Output Image", use_column_width=True)
                else:
                    box = cv2.imread('output.jpg')
                    st.image(box, caption="Output Image box", use_column_width=True)
                    img1 = np.array(img1)

                    img = cv2.resize(img1, (64, 64))

                    hu = fd_hu_moments(img)
                    haralick = fd_haralick(img)
                    histogram = fd_histogram(img)
                    img = np.hstack([histogram, haralick, hu])
                    img = np.asarray(img)
                    temp1 = scaler1.transform([img])
                    temp2 = scaler2.transform([img])

                    harvest_time = rf1.predict(temp1)
                    growth_rate = rf2.predict(temp2)

                    img = cv2.resize(img1, (480, 640))
                    cv2.putText(img, 'Harvest stage ' + str(harvest_time[0] + 1), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)
                    cv2.putText(img, 'Growth Grade ' + str(growth_rate[0] + 1), (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 0), 2)

                    st.image(img, caption="Classification Result", use_column_width=True)
                    st.subheader("Harvest stage " + str(harvest_time[0] + 1))
                    st.subheader("Growth Grade  " + str(growth_rate[0] + 1))
        else:
            st.warning("Please upload an image")
    if choice=="📋 Model Analysis":

        model_analysis = ["📊 Graphs1", "📊 Graphs2", "📝 Metrics 1 ", "📝 Metrics 2 "]
        choice = st.sidebar.selectbox("📜 Model Analysis", model_analysis)
        if choice == "📊 Graphs1":
            st.write("    Dataset 1 plots ")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors = ['orange', 'green', 'grey', 'brown']

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([svm_acc[0], knn_acc[0], lr_acc[0], rf_acc[0]])

            ax1.bar(x, y,color=colors[0])

            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Accuracy')

            st.pyplot(fig1)

            st.write("   Precision  ")
            fig2, ax2 = plt.subplots(figsize=(8, 6))

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([precision1[0], precision1[1], precision1[2], precision1[3]])

            ax2.bar(x, y,color=colors[1])

            ax2.set_xlabel('Algorithm')
            ax2.set_ylabel('Precision')

            st.pyplot(fig2)

            st.write("    Recall")
            fig3, ax3 = plt.subplots(figsize=(8, 6))

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([recall1[0], recall1[1], recall1[2], recall1[3]])

            ax3.bar(x, y,color=colors[2])

            ax3.set_xlabel('Algorithm')
            ax3.set_ylabel('Recall')

            st.pyplot(fig3)




        if choice == "📊 Graphs2":
            st.write("    Dataset 2 plots ")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors = ['orange', 'green', 'grey', 'brown']

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([svm_acc[1], knn_acc[1], lr_acc[1], rf_acc[1]])

            ax1.bar(x, y,color=colors[0])

            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Accuracy')

            st.pyplot(fig1)

            st.write("   Precision  ")
            fig2, ax2 = plt.subplots(figsize=(8, 6))

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([precision2[0], precision2[1], precision2[2], precision2[3]])

            ax2.bar(x, y,color=colors[1])

            ax2.set_xlabel('Algorithm')
            ax2.set_ylabel('Precision')

            st.pyplot(fig2)

            st.write("    Recall")
            fig3, ax3 = plt.subplots(figsize=(8, 6))

            # Generate the bar graph plot using Matplotlib
            x = np.array(['SVM', 'KNN', 'LR', 'RF'])
            y = np.array([recall2[0], recall2[1], recall2[2], recall2[3]])

            ax3.bar(x, y,color=colors[2])

            ax3.set_xlabel('Algorithm')
            ax3.set_ylabel('Recall')

            st.pyplot(fig3)








        if choice== "📝 Metrics 1 ":

            st.header("    Dataset 1 Metrics ")
            canvas = st.empty()
            # Display the accuracies as text
            st.subheader("SVM")

            st.write(f"Accuracy: {'{:.4f}'.format(svm_acc[0])}")
            st.write(f"Precision: {'{:.4f}'.format(precision1[0])}")

            st.write(f"Recall     : {'{:.4f}'.format(recall1[0])}")

            st.subheader("KNN")
            st.write(f" Accuracy  : {'{:.4f}'.format(knn_acc[0])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision1[1])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall1[1])}")

            st.subheader("Logistic Regression")
            st.write(f" Accuracy  : {'{:.4f}'.format(lr_acc[0])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision1[2])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall1[2])}")

            st.subheader("Random Forest")
            st.write(f" Accuracy  : {'{:.4f}'.format(rf_acc[0])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision1[3])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall1[3])}")



        if choice == "📝 Metrics 2 ":
            st.header("   Dataset 2 Metrics ")
            canvas = st.empty()
            # Display the accuracies as text
            st.subheader("SVM")
            st.write(f" Accuracy  : {'{:.4f}'.format(svm_acc[1])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision2[0])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall2[0])}")

            st.subheader("KNN")
            st.write(f" Accuracy  : {'{:.4f}'.format(knn_acc[1])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision2[1])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall2[1])}")

            st.subheader("Logistic Regression")
            st.write(f"Accuracy   : {'{:.4f}'.format(lr_acc[1])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision2[2])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall2[2])}")

            st.subheader("Random Forest")
            st.write(f"Accuracy   : {'{:.4f}'.format(rf_acc[1])}")
            st.write(f"Precision  : {'{:.4f}'.format(precision2[3])}")
            st.write(f"Recall     : {'{:.4f}'.format(recall2[3])}")

uploadDataset()

runrandomforest()
runSVM()
runKNN()
runlogistic()
main()
