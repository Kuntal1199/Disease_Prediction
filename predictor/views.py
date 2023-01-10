from django.shortcuts import render
# from django.shortcuts import redirect
from django.conf import settings
# from django.http import HttpResponse
# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from .models import DiseaseDescription
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.metrics import accuracy_score
import tensorflow.keras
from PIL import Image, ImageOps
import os
import tempfile

# Create your views here.
def home(request):
    description_queryset = DiseaseDescription.objects.all()
    return render(request, 'home.html',
                  context={"disease_description": description_queryset})


def front(request):
    return render(request, 'front.html')


def terms_condition(request):
    return render(request, 'terms_condition.html')


def disease_form(request, id):
    disease_obj = DiseaseDescription.objects.filter(id=id).first()
    return render(request, "disease_form.html",
                  context={"disease_obj": disease_obj})


def checkdiabetes(request):
    pregnancy = request.POST.get('pregnancies')
    glucose = request.POST.get('glucose')
    blood_pressure = request.POST.get('blood_pressure')
    skin_thickness = request.POST.get('skin_thickness')
    insulin_level = request.POST.get('insulin')
    body_mass_index = request.POST.get('bmi')
    diabetes_pedigree_function = request.POST.get('diabetes_pedigree_function')
    age = request.POST.get('age')

    # Loading required csv file
    data = pd.read_csv(r"D:\Tanushree\datasets\diabetes.csv")
    # Train test split
    X = data.drop("Outcome", axis=1)  # Contains all independent variables
    Y = data['Outcome']
    print(X)
    print(Y)
    # This line of code divides splits the 20% of the data as test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train)

    # Instantiate or calling the model using Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # user data
    user_data = np.array([pregnancy, glucose, blood_pressure, skin_thickness, insulin_level,
                          body_mass_index, diabetes_pedigree_function, age], dtype=np.float).reshape(1, 8)
    # fit the model with data
    model.fit(X_train, Y_train)
    # Predicting the output for our test set
    prediction = model.predict(user_data)
    # accuracy = accuracy_score(Y_train,prediction)
    # print("Accuracy: ", accuracy * 100, "%")
    return render(request, "result.html",
                  {'output': "You've diabetes." if prediction[0] == 1 else "You don't have diabetes."})
    # , 'accuracy': accuracy})


def checkbreastcancer(request):
    mean_radius = request.POST.get('mean_radius')
    mean_texture = request.POST.get('mean_texture')
    mean_perimeter = request.POST.get('mean_perimeter')
    mean_area = request.POST.get('mean_area')
    mean_smoothness = request.POST.get('mean_smoothness')

    # Loading required csv file
    data = pd.read_csv(r"D:\Tanushree\datasets\datasets_56485_108594_Breast_cancer_data.csv")
    # Train test split
    X = data.drop("diagnosis", axis=1)  # Contains all independent variables
    Y = data['diagnosis']
    print(X)
    print(Y)
    # This line of code divides splits the 20% of the data as test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train)

    # Instantiate or calling the model using Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # user data
    user_data = np.array([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness],
                         dtype=np.float).reshape(1, 5)
    # fit the model with data
    model.fit(X_train, Y_train)
    # Predicting the output for our test set
    prediction = model.predict(user_data)
    # accuracy = accuracy_score(Y_train,prediction)
    # print("Accuracy: ", accuracy * 100, "%")
    return render(request, "result.html",
                  {'output': "You've BreastCancer." if prediction[0] == 1 else "You don't have BreastCancer."})
    # , 'accuracy': accuracy})


def checklungcancer(request):
    age = request.POST.get('Age')
    smokes = request.POST.get('Smokes')
    areaQ = request.POST.get('AreaQ')
    alkhol = request.POST.get('Alkhol')

    # Loading required csv file
    data = pd.read_csv(r"D:\Tanushree\datasets\lung_cancer.csv")
    # Train test split
    X = data.drop("Result", axis=1)  # Contains all independent variables
    Y = data['Result']
    print(X)
    print(Y)
    # This line of code divides splits the 20% of the data as test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train)

    # Instantiate or calling the model using Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # user data
    user_data = np.array([age, smokes, areaQ, alkhol],
                         dtype=np.float).reshape(1, 4)
    # fit the model with data
    model.fit(X_train, Y_train)
    # Predicting the output for our test set
    prediction = model.predict(user_data)
    # accuracy = accuracy_score(Y_train,prediction)
    # print("Accuracy: ", accuracy * 100, "%")
    return render(request, "result.html",
                  {'output': "You've Lung Cancer." if prediction[0] == 1 else "You don't have Lung Cancer."})
    # , 'accuracy': accuracy})


def checkkidneydisease(request):
    sg = request.POST.get('sg')
    al = request.POST.get('al')
    sc = request.POST.get('sc')
    hemo = request.POST.get('hemo')
    pcv = request.POST.get('pcv')

    # Loading required csv file
    data = pd.read_csv(r"D:\Tanushree\datasets\kidney_disease1.csv")

    # # Create a list of columns to retain
    # columns_to_retain = ["sg", "al", "sc", "hemo",
    #                      "pcv", "wbcc", "rbcc", "htn", "classification"]
    #
    # # columns_to_retain = df.columns, Drop the columns that are not in columns_to_retain
    # data = data.drop([col for col in data.columns if not col in columns_to_retain], axis=1)
    #
    # # Drop the rows with na or missing values
    # data = data.dropna(axis=0)
    # # Transform non-numeric columns into numerical columns
    # for column in data.columns:
    #     if data[column].dtype == np.number:
    #         continue
    #     data[column] = LabelEncoder().fit_transform(data[column])

    # Train test split
    X = data.drop("classification", axis=1)  # Contains all independent variables
    Y = data['classification']
    print(X)
    print(Y)
    # This line of code divides splits the 20% of the data as test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    print(X_train)

    # Instantiate or calling the model using Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # user data
    user_data = np.array([sg, al, sc, hemo, pcv],
                         dtype=np.float).reshape(1, 5)
    # fit the model with data
    model.fit(X_train, Y_train)
    # Predicting the output for our test set
    prediction = model.predict(user_data)
    # accuracy = accuracy_score(Y_train,prediction)
    # print("Accuracy: ", accuracy * 100, "%")
    return render(request, "result.html",
                  {'output': "You've Kidney Disease." if prediction[0] == 1 else "You don't have Kidney Disease."})
    # , 'accuracy': accuracy})


def brain_tumor(request):
    results = ["You've Brain Tumor.", "You don't have Brain Tumor."]
    uploaded_image = request.FILES.get("Image")
    np.set_printoptions(suppress=True)
    model_path = str(settings.BASE_DIR) + "\models\keras_model.h5"
    model = tensorflow.keras.models.load_model(model_path,compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # # Replace this with the path to your image
    image = Image.open(uploaded_image)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # # display the resized image
    # image.show()
    # # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index_of_result = np.argmax(prediction)
    return render(request, "result.html", {'output': results[index_of_result]})


def heartdisease(request):
    age = request.POST.get('age')
    sex = request.POST.get('sex')
    cp = request.POST.get('cp')
    trestbps = request.POST.get('trestbps')
    chol = request.POST.get('chol')
    fbs = request.POST.get('fbs')
    restecg = request.POST.get('restecg')
    thalach = request.POST.get('thalach')
    exang = request.POST.get('exang')
    oldpeak = request.POST.get('oldpeak')
    slope = request.POST.get('slope')
    ca = request.POST.get('ca')
    thal = request.POST.get('thal')

    # Loading required csv file
    data = pd.read_csv(r"D:\Tanushree\datasets\heart.csv")
    # Train test split
    X = data.drop("target", axis=1)  # Contains all independent variables
    Y = data['target']
    print(X)
    print(Y)
    # This line of code divides splits the 20% of the data as test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train)

    # Instantiate or calling the model using Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # user data
    user_data = np.array([age, sex, cp, trestbps, chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal],
                         dtype=np.float).reshape(1, 13)
    # fit the model with data
    model.fit(X_train, Y_train)
    # Predicting the output for our test set
    prediction = model.predict(user_data)
    # accuracy = accuracy_score(Y_train,prediction)
    # print("Accuracy: ", accuracy * 100, "%")
    return render(request, "result.html",
                  {'output': "You've heart disease." if prediction[0] == 1 else "You don't have heart disease."})
    # , 'accuracy': accuracy})