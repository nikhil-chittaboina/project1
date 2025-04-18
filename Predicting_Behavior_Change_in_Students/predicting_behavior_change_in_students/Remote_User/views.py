from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,predicting_behavior_change,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Student_Behavior_Change_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            Certification_Course= request.POST.get('Certification_Course')
            Gender= request.POST.get('Gender')
            Department= request.POST.get('Department')
            Height_CM= request.POST.get('Height_CM')
            Weight_KG= request.POST.get('Weight_KG')
            Tenth_Mark= request.POST.get('Tenth_Mark')
            Twelth_Mark= request.POST.get('Twelth_Mark')
            hobbies= request.POST.get('hobbies')
            daily_studing_time= request.POST.get('daily_studing_time')
            prefer_to_study_in= request.POST.get('prefer_to_study_in')
            like_your_degree= request.POST.get('like_your_degree')
            social_medai_video= request.POST.get('social_medai_video')
            Travelling_Time= request.POST.get('Travelling_Time')
            Stress_Level= request.POST.get('Stress_Level')
            Financial_Status= request.POST.get('Financial_Status')
            alcohol_consumption= request.POST.get('alcohol_consumption')
            part_time_job= request.POST.get('part_time_job')


        df = pd.read_csv('Student_Behaviour.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0  # Good
            elif (Label == 1):
                return 1  # Bad

        df['Results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Fid']
        y = df['Label']

        print("FID")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()
        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Deep Neural Network-DNN")
        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        testscore_mlpc = accuracy_score(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Good'
        elif (prediction == 1):
            val = 'Bad'


        print(val)
        print(pred1)

        predicting_behavior_change.objects.create(Fid=Fid,
        Certification_Course=Certification_Course,
        Gender=Gender,
        Department=Department,
        Height_CM=Height_CM,
        Weight_KG=Weight_KG,
        Tenth_Mark=Tenth_Mark,
        Twelth_Mark=Twelth_Mark,
        hobbies=hobbies,
        daily_studing_time=daily_studing_time,
        prefer_to_study_in=prefer_to_study_in,
        like_your_degree=like_your_degree,
        social_medai_video=social_medai_video,
        Travelling_Time=Travelling_Time,
        Stress_Level=Stress_Level,
        Financial_Status=Financial_Status,
        alcohol_consumption=alcohol_consumption,
        part_time_job=part_time_job,
        Prediction=val)

        return render(request, 'RUser/Predict_Student_Behavior_Change_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Student_Behavior_Change_Type.html')



