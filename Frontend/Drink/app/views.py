from django.shortcuts import render, redirect

# Create your views here.
from django.contrib.auth.models import User
from django.contrib import messages
from . models import Register
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from pandas.plotting import lag_plot
from sklearn.linear_model import LinearRegression

#importing the required libraries


Home = 'index.html'
About = 'about.html'
Login = 'login.html'
Registration = 'registration.html'
Userhome = 'userhome.html'
Load = 'load.html'
View = 'view.html'
Preprocessing = 'preprocessing.html'
Model = 'model.html'
Prediction = 'prediction.html'


# # Home page
def index(request):

    return render(request, Home)

# # About page


def about(request):
    return render(request, About)

# # Login Page


def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = Register.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        if d:
            return redirect(userhome)
        else:
            msg = 'Login failed'
            return render(request, Login, {'msg': msg})
    return render(request, Login)

# # registration page user can registration here


def registration(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        if password == conpassword:
            userdata = Register.objects.filter(email=email).exists()
            if userdata:
                msg = 'Account already exists'
                return render(request, Registration, {'msg': msg})
            else:
                userdata = Register(name=Name, email=email,
                                    password=password, age=age, contact=contact)
                userdata.save()
                return render(request, Login)
        else:
            msg = 'Register failed!!'
            return render(request, Registration, {'msg': msg})

    return render(request, Registration)

# # user interface


def userhome(request):

    return render(request, Userhome)

# # Load Data


def load(request):
    if request.method == "POST":
        global df
        file = request.FILES['file']
        df = pd.read_csv(file)
        messages.info(request, "Data Uploaded Successfully")

    return render(request, Load)

# # View Data


def view(request):
    col = df.to_html
    dummy = df.head(100)

    col = dummy.columns
    rows = dummy.values.tolist()
    # return render(request, 'view.html',{'col':col,'rows':rows})
    return render(request, View, {'columns': df.columns.values, 'rows': df.values.tolist()})


# preprocessing data
def preprocessing(request):
    global x_train, x_test, y_train, y_test, x, y ,df
    
    if request.method == "POST":

        size = int(request.POST['split'])
        size = size / 100

        df['Month'] = pd.to_datetime(df.Data).dt.month.astype(int)
        df['Day'] = pd.to_datetime(df.Data).dt.day.astype(int)
        df['Year'] = pd.to_datetime(df.Data).dt.year.astype(int)
        df = df.drop('Data', axis=1)

        df['Hour'] = pd.to_datetime(df.Time).dt.hour.astype(int)
        df['Minute'] = pd.to_datetime(df.Time).dt.minute.astype(int)
        df['Second'] = pd.to_datetime(df.Time).dt.second.astype(int)
        df = df.drop('Time', axis=1)

        df['SunriseHour'] = pd.to_datetime(df.TimeSunRise).dt.hour.astype(int)
        df['SunriseMinute'] = pd.to_datetime(df.TimeSunRise).dt.minute.astype(int)
        df['SunsetHour'] = pd.to_datetime(df.TimeSunSet).dt.hour.astype(int)
        df['SunsetMinute'] = pd.to_datetime(df.TimeSunSet).dt.minute.astype(int)
        df = df.drop(['TimeSunRise','TimeSunSet'], axis=1)

        df=df.drop('Year', axis=1)
        df = df.drop('SunriseHour', axis=1)
        
        # Data Splitting
        y = df['Radiation'].copy()
        x = df.drop('Radiation',axis = 1).copy()

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
       
        messages.info(request, "Data Preprocessed and It Splits Succesfully")
    return render(request, Preprocessing)


# Model Training
def model(request):
    global x_train, x_test, y_train, y_test,module
    if request.method == "POST":
        model = request.POST['algo']
        if model == "0":
            from sklearn.tree import DecisionTreeRegressor
            module = DecisionTreeRegressor()
            dt = module.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt=r2_score(y_test,y_pred)
            acc_dt=acc_dt*100
            msg = 'Accuracy of DecisionTreeRegressor : ' + str(acc_dt)
            return render(request, Model, {'msg': msg})

        elif model == "1":
            from sklearn.ensemble import RandomForestRegressor
            module = RandomForestRegressor()
            module = module.fit(x_train,y_train)
            y_pred = module.predict(x_test)
            acc_rf=r2_score(y_test,y_pred)
            acc_rf=acc_rf*100
            msg = 'Accuracy of RandomForestRegressor : ' + str(acc_rf)
            return render(request, Model, {'msg': msg})

        elif model == "2":
            from sklearn.linear_model import LinearRegression
            module = LinearRegression()
            module = module.fit(x_train,y_train)
            y_pred = module.predict(x_test)
            acc_lr=r2_score(y_test,y_pred)
            acc_lr=acc_lr*100
            msg = 'Accuracy of LinearRegression : ' + str(acc_lr)
            return render(request, Model, {'msg': msg})
        
        elif model == "3":
            from sklearn.ensemble import AdaBoostRegressor
            abd=AdaBoostRegressor()
            abd=abd.fit(x_train,y_train)
            y_pred=abd.predict(x_test)
            acc_abd=r2_score(y_test,y_pred)
            acc_abd=acc_abd*100
            msg = 'accuracy of AdaBoostRegressor : ' + str(acc_abd)
            return render(request, Model, {'msg': msg})
        
        elif model == '4':
            from sklearn.neighbors import KNeighborsRegressor
            knn=KNeighborsRegressor()
            knn=knn.fit(x_train, y_train)
            y_pred=knn.predict(x_test)
            acc_knn=r2_score(y_test, y_pred)
            acc_knn=acc_knn*100 
            print(acc_knn)
            msg = 'accuracy of KNeighborsRegressor : ' + str(acc_knn)
            return render(request, Model, {'msg': msg})


    return render(request, Model)


# Prediction here we can find the result based on user input values.
def prediction(request):

    global x_train,x_test,y_train,y_test,x,y
    

    if request.method == 'POST':
        

        f1=float(request.POST['UNIXTime'])
        f2=float(request.POST['Temperature'])
        f3=float(request.POST['Pressure'])
        f4=float(request.POST['Humidity'])
        f5=float(request.POST['WindDirection(Degrees)'])
        f6=float(request.POST['Speed'])
        f7=float(request.POST['Month'])
        f8=float(request.POST['Day'])
        f9=float(request.POST['Hour'])
        f10=float(request.POST['Minute'])
        f11=float(request.POST['Second'])
        f12=float(request.POST['SunriseMinute'])
        f13=float(request.POST['SunsetHour'])
        f14=float(request.POST['SunsetMinute'])

        PRED = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]]
        					
        
        from sklearn.ensemble import RandomForestRegressor
        RF = RandomForestRegressor()
        RF=RF.fit(x_train,y_train)
        result = RF.predict(PRED)
        print(result)
        msg=f"Radiation = {result}"
        return render(request,Prediction,{'msg':msg})  

    return render(request,Prediction)