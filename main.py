import json
from flask import Flask, render_template, request, redirect,session, url_for, jsonify
import bcrypt
from flask_mail import Mail,Message
import random
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient, collection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
import secrets
cluster = MongoClient("mongodb+srv://root:root@cluster0.mngotjl.mongodb.net/?retryWrites=true&w=majority",connect=False)

if cluster:
    print("Connected to Databse")

db=cluster["registrationInfo"]
collection = db["users"]
collection2 = db['patients']
collection3 = db['appointments']
# post = {"_id":0,"name":"shivam"}
# collection.insert_one(post)

app = Flask(__name__,template_folder='templates')
app.secret_key="login"

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, username):
        self.username = username
    def is_active(self):
        return True
    def get_id(self):
        return self.username

@login_manager.user_loader
def load_user(username):
    user_data = db.users.find_one({'username': username})
    # u = mongo.db.Users.find_one({"Name": username})
    if not user_data:
        return None
    return User(username=user_data['username'])

with open('templates/config.json','r') as f:
    params = json.load(f)['param']

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = params['gmail-user']
app.config['MAIL_PASSWORD'] = params['gmail-password']
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
mail1 = Mail(app)

otp =  str(secrets.randbelow(10**6)).zfill(6)
R_id = 'DR' + str(random.randint(10000, 99999))
Patient_Id = 'P' + str(random.randint(10000, 99999))
print(R_id)
@app.route('/')
def index():
	return render_template('login.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = db.users.find_one({'username': username})
        print(user)
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            user_id =user['R_id']
            print(user_id)
            session['s_username'] = username
            session['s_id'] = user_id
            user_obj=User(username=user.get('username'))
            login_user(user_obj)
            return render_template('/home.html',s_username=username,s_id = user_id)
            # return redirect(url_for('home'))
        else:
            return render_template('/login.html',msg="Password or Username is Incorrect.")
    else:
        return render_template('login.html')

@app.route('/loginPatient', methods=['GET', 'POST'])
def loginPatient():
    if request.method == 'POST':
        p_username = request.form['p_username']
        p_password = request.form['p_password']

        user = collection2.find_one({'p_username': p_username})
        user1 = collection2.find_one({'p_username': p_username}, {'_id': False})
        print(user)
        print(user1)
        if user and bcrypt.checkpw(p_password.encode('utf-8'), user['p_password']):
            P_id =user['Patient_ID']
            print(P_id)
            session['p_username'] = p_username
            session['p_id'] = P_id
            session['user'] = user1
            user_obj=User(username=user.get('p_username'))
            login_user(user_obj)

            return render_template('/patientHome.html',p_username=p_username,P_id = P_id,patient=user1)
            # return redirect(url_for('home'))
        else:
            return render_template('/login.html',msg="Password or Username is Incorrect.")
    else:
        return render_template('login.html')

def user_exists(email, mob, username):
    search_values = {"username": username, "email": email, "mob": mob}
    all_documents = collection.find()
    for document in all_documents:
        if any(document.get(key) == value for key, value in search_values.items()):
            print("Found a matching document:")
            print(document)
            return True  # User exists
    print("User not found.")
    return False  # User does not exist
def patient_exists(email, mob, username):
    patient_search_values = {"p_username": username, "p_email": email, "p_mob": mob}
    patient_all_documents = collection2.find()
    for p_document in patient_all_documents:
        if any(p_document.get(key) == value for key, value in patient_search_values.items()):
            print("Found a matching document:")
            print(p_document)
            return True  # User exists
    print("User not found.")
    return False  # User does not exist
@app.route('/register', methods=['GET', 'POST'])
def register():
    display = ""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        mob = request.form['mob']
        gender = request.form['gender']

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        if user_exists(email,mob,username): #username,
            # print(f"User with name '{username}' and mobile number '{mob}' already exists.")
            display ="block"
            return render_template("register.html",block=display)
        else:
            # post = {"username": username, "password": hashed_password,"name":name,"email":email,"mob":mob,"gender":gender}
            # collection.insert_one(post)

            msg= Message('OTP',sender='g37.dypcet@gmail.com',recipients=[email])
            msg.body="Your One Time Password Password is : " + str(otp)
            mail.send(msg)
            session['username'] = username
            session['hashed_password'] = hashed_password
            session['name'] = name
            session['email'] = email
            session['mob'] = mob
            session['gender'] = gender

            # return redirect('index')
            return render_template("verifyDoc.html")
    else:
        return render_template('register.html')

@app.route('/patientRegister', methods=['GET', 'POST'])
def patientRegister():
    display = ""
    if request.method == 'POST':
        p_username = request.form['p_username']
        p_password = request.form['p_password']
        p_name = request.form['p_name']
        p_email = request.form['p_email']
        p_mob = request.form['p_mob']
        p_gender = request.form['p_gender']

        print(p_username,p_password,p_name,p_email,p_mob,p_gender)

        p_hashed_password = bcrypt.hashpw(p_password.encode('utf-8'), bcrypt.gensalt())

        if patient_exists(p_email,p_mob,p_username): #username,
            # print(f"User with name '{username}' and mobile number '{mob}' already exists.")
            display ="block"
            return render_template("register.html",block=display)
        else:
            # post = {"username": username, "password": hashed_password,"name":name,"email":email,"mob":mob,"gender":gender}
            # collection.insert_one(post)

            send_msg= Message('OTP',sender='g37.dypcet@gmail.com',recipients=[p_email])
            send_msg.body="Your One Time Password Password is : "+str(otp)
            mail.send(send_msg)
            session['p_username'] = p_username
            session['p_hashed_password'] = p_hashed_password
            session['p_name'] = p_name
            session['p_email'] = p_email
            session['p_mob'] = p_mob
            session['p_gender'] = p_gender

            # return redirect('index')
            return render_template("verifyPatient.html")
    else:
        return render_template('register.html')

@app.route('/patientHome', methods=['GET', 'POST'])
# @login_required
def patinetHome():
    p_username = session.get('p_username', 'PUsername')
    P_id = session.get('p_id', 'PatinetID')
    user = session.get('user', 'Patient_Info')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        # mob = request.form['mob']
        gender = request.form['gender']
        dob = request.form['dob']
        age = request.form['age']
        fdate = request.form['FDate']
        addrss = request.form['address']
        city = request.form['city']
        zip = request.form['zip']
        state = request.form['state']

        print(name,gender,fdate,addrss,city,zip,state,age)

        filter_criteria = {"p_email": email}

        update_operation = {
            "$set": {
                "p_name": name,
                "p_gender": gender,
                "p_address": addrss,
                "p_dob": dob,
                "p_city" : city,
                "p_zip" : zip,
                "p_state" : state,
                "p_age" :age
            }
        }

        collection2.update_one(filter_criteria,update_operation)

        p_username = session.get('p_username', 'PUsername')
        P_id = session.get('p_id', 'PatinetID')
        user = session.get('user','Patient_Info')
        return render_template('patientHome.html',p_username=p_username, P_id = P_id,patient=user)
    else:
        return render_template('patientHome.html', p_username=p_username, P_id=P_id, patient=user)

@app.route('/appointments',methods=['GET','POST'])
def appointments():
    if request.method =="POST":
        patient_reason = request.form['patient_reason'];
        patient_name = request.form['patient_name'];
        patient_email = request.form['patient_email'];
        patient_phone = request.form['patient_phone'];
        patient_date = request.form['patient_date'];
        formatted_date = request.form['appDate'];
        patient_time = request.form['patient_time'];

        user = session.get('user','Patient_Info')

        post = {"p_reason": patient_reason, "p_name": patient_name,"p_email":patient_email,"p_phone":patient_phone,"p_date":patient_date,"formatted_date":formatted_date,"p_time":patient_time,"p_id":user['Patient_ID'],"status":"Scheduled"}
        collection3.insert_one(post)

        print(patient_reason,patient_name,patient_email,patient_phone,patient_date,patient_time);
        p_username = session.get('p_username', 'PUsername')
        P_id = session.get('p_id', 'PatinetID')

        # Patient id added to databse and schedules added
        return render_template('patientHome.html',p_username=p_username, P_id = P_id,patient=user)
@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    s_username = session.get('s_username', 'User')
    s_id = session.get('s_id','DR')
    if request.method == 'POST':
        uploadedFile = request.files['uploadFile']
        file_path = f'content/{uploadedFile.filename}'
        uploadedFile.save(file_path)
    return render_template('home.html',s_username=s_username,s_id=s_id)

def docEmailExits(email):
    search_values = {"email": email}
    all_documents = collection.find()
    for document in all_documents:
        if any(document.get(key) == value for key, value in search_values.items()):
            print("Sended Email found in the doc collection")
            print(document)
            return True  # User exists
    print("User not found.")
    return False

def patientEmailExits(email):
    search_values = {"p_email": email}
    all_documents = collection2.find()
    for document in all_documents:
        if any(document.get(key) == value for key, value in search_values.items()):
            print("Sended Email found in the patient collection")
            print(document)
            return True  # User exists
    print("User not found.")
    return False
@app.route('/docForgotPass',methods = ['GET','POST'])
def  docForgotPass():
    if request.method == "POST":
        doc_email = request.form['doc_email']
        session['doc_email'] = doc_email
        if(docEmailExits(doc_email)):
            msg= Message('OTP',sender='g37.dypcet@gmail.com',recipients=[doc_email])
            msg.body="Your One Time Password to Forgot Password is : " + str(otp)
            mail.send(msg)
            # return redirect('index')
            return render_template("verifyDocOtp.html")
        else :
            return render_template('docForgotPass.html',msg = "Sorry, Email Not Found")
    return render_template('docForgotPass.html')

@app.route('/patientForgotPass',methods = ['GET','POST'])
def patientForgotPass():
    if request.method == "POST":
        patient_email = request.form['patient_email']
        session['patient_email'] = patient_email
        if(patientEmailExits(patient_email)):
            msg= Message('OTP',sender='g37.dypcet@gmail.com',recipients=[patient_email])
            msg.body="Your One Time Password to Forgot Password is : " + str(otp)
            mail.send(msg)
            # return redirect('index')
            return render_template("verifyPatientOtp.html")
        else :
            return render_template('patientForgotPass.html',msg = "Sorry, Email Not Found")
    return render_template('patientForgotPass.html')


@app.route('/validatePatientInfo',methods=['GET','POST'])
def validatePatientInfo():
    p_username = session.get('p_username',None)
    p_hashed_password = session.get('p_hashed_password', None)
    p_name = session.get('p_name', None)
    p_email = session.get('p_email', None)
    p_mob = session.get('p_mob', None)
    p_gender = session.get('p_gender', None)

    userOtp = request.form['otp']
    if otp==userOtp:
        post = {"p_username": p_username, "p_password": p_hashed_password,"p_name":p_name,"p_email":p_email,"p_mob":p_mob,"p_gender":p_gender,"Patient_ID":Patient_Id}
        collection2.insert_one(post)
        print(p_username, p_hashed_password, p_name, p_email, p_mob, p_gender,Patient_Id)
        return render_template("success.html")
    else:
        return render_template('verifyPatient.html',msg="Invalid OTP")

@app.route('/validateDocInfo',methods=['GET', 'POST'])
def validate():
    username = session.get('username',None)
    hashed_password = session.get('hashed_password', None)
    name = session.get('name', None)
    email = session.get('email', None)
    mob = session.get('mob', None)
    gender = session.get('gender', None)

    userOtp = request.form['otp']
    if otp==userOtp:
        post = {"username": username, "password": hashed_password,"name":name,"email":email,"mob":mob,"gender":gender,"R_id":R_id}
        collection.insert_one(post)
        print(username, hashed_password, name, email, mob, gender,R_id)
        return render_template("success.html")
    else:
        return render_template('verifyDoc.html',msg="Invalid OTP")\

@app.route('/validateDocOtp',methods=['GET', 'POST'])
def validateDocOtp():
    userOtp = request.form['otp']
    if otp==userOtp:
        return render_template("changeDocPass.html")
    else:
        return render_template('verifyDocOtp.html',msg="Invalid OTP")

@app.route('/validatePatientOtp',methods=['GET', 'POST'])
def validatePatientOtp():
    userOtp = request.form['otp']
    if otp==userOtp:
        return render_template("changePatientPass.html")
    else:
        return render_template('verifyPatientOtp.html',msg="Invalid OTP")
@app.route('/changeDocPassword',methods=['GET', 'POST'])
def changeDocPassword():
    f_pass = request.form['f_pass']
    c_pass = request.form['c_pass']
    if f_pass == c_pass:
        doc_email = session.get('doc_email', None)
        hashed_password = bcrypt.hashpw(c_pass.encode('utf-8'), bcrypt.gensalt())
        filter_criteria = {"email": doc_email}

        update_operation = {
            "$set": {
                "password": hashed_password
            }
        }
        collection.update_one(filter_criteria, update_operation)
        return render_template('passChangeSuccessfully.html')
    else:
        return render_template('changeDocPass.html',msg = "Password do not match")

@app.route('/changePatientPassword',methods=['GET', 'POST'])
def changePatientPassword():
    f_pass = request.form['f_pass']
    c_pass = request.form['c_pass']
    if f_pass == c_pass:
        patient_email = session.get('patient_email', None)
        hashed_password = bcrypt.hashpw(c_pass.encode('utf-8'), bcrypt.gensalt())
        filter_criteria = {"p_email": patient_email}

        update_operation = {
            "$set": {
                "p_password": hashed_password
            }
        }
        collection2.update_one(filter_criteria, update_operation)
        return render_template('passChangeSuccessfully.html')
    else:
        return render_template('changePatientPass.html',msg = "Password do not match")

@app.route('/aboutUs')
@login_required
def aboutUs():
    s_username = session.get('s_username','User')
    s_id = session.get('s_id','DR')
    print(s_id)
    return render_template('aboutUs.html',s_username=s_username,s_id=s_id)

@app.route('/predict',methods=['GET', 'POST'])
@login_required
def predict():
    s_username = session.get('s_username', 'User')
    if request.method=="POST":
        glevel = request.form['glevel']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        age = request.form['age']

        from data_utils import load_diabetes_dataset, get_data_stats
        from data_split import split_data
        from model import train_svm_classifier, evaluate_model
        from prediction import predict_diabetes_status

        data_path = 'content/diabetes.csv'

        # Load the dataset
        diabetes_dataset = load_diabetes_dataset(data_path)

        # Get data statistics
        shape, data_desc, outcome_counts, outcome_means,dataset_head,dataset_tail,dataset_column,dataset_info= get_data_stats(diabetes_dataset)
        # print("Data Shape:", shape)
        # print("Data Description:", data_desc)
        summary_stats= data_desc.transpose().round(2)
        table2 =summary_stats.reset_index().values.tolist()
        titles2 =summary_stats.reset_index().columns.tolist()
        # print("Outcome Counts:", outcome_counts)
        # print("Head",dataset_head)
        # print('Values',dataset_head.values)
        # print("Tail",dataset_tail)
        # print('Column Names',dataset_column.values)


        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        outcome_counts.plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        ax[0].set_title('Outcome')
        ax[0].set_ylabel('')
        sns.countplot(x='Outcome', data=diabetes_dataset, ax=ax[1])
        N, P = outcome_counts.value_counts()
        print('Negative (0)', N)
        print('Positive (1)', P)
        plt.title('Count Plot')
        plt.grid()
        outcome_path = 'static/plots/plot1.png'
        plt.savefig(outcome_path)
        # plt.show()

        #Histograms of each feature
        new_size1=(10,10)
        diabetes_dataset.hist(bins=10, figsize=new_size1)
        histogram_path = 'static/plots/histogram.png'
        plt.savefig(histogram_path)
        # plt.show()

        #Scatter Plot Matrix
        from pandas.plotting import scatter_matrix
        scatter_matrix(diabetes_dataset, figsize=(20, 20))
        scatter_path = 'static/plots/scatter.png'
        plt.savefig(scatter_path)
        # plt.show()

        # Corelation analysis
        # get coreleation of each features in dataset
        corrmat = diabetes_dataset.corr()
        top_corr_features = corrmat.index
        new_size2=(10,10)
        plt.figure(figsize=new_size2)
        # plot heat map
        g = sns.heatmap(diabetes_dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
        heatmap_path = 'static/plots/heatmap.png'
        plt.savefig(heatmap_path)
        # plt.show()

        # print("Outcome Means:", outcome_means)

        X = diabetes_dataset.drop(columns='Outcome')
        Y = diabetes_dataset['Outcome']

        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        # Train the SVM classifier
        classifier = train_svm_classifier(X_train, Y_train)

        # Evaluate the model
        training_accuracy, test_accuracy,cm_svm,sv_pred= evaluate_model(classifier, X_train, Y_train, X_test, Y_test)
        print('Accuracy score of the training data:', training_accuracy*100)
        print('Accuracy score of the test data:', test_accuracy*100)
        print(' Precision Score of SVM : ', precision_score(Y_test, sv_pred) * 100)
        print(' Recall Score of SVM : ', recall_score(Y_test, sv_pred) * 100)
        # print(' F1 Score of SVM : ',f1_score(Y_test,sv_pred)*100)

        # print('Classification Report on SVM',classification_report(Y_test,sv_pred,digits=4))
        print('Confusion Matric : ',cm_svm)

        print('Confusion Matrix of SVM ')
        plt.clf()
        plt.imshow(cm_svm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['0', '1']
        plt.title('Confusion Matrix of SVM')
        plt.ylabel('Actual (true) values')
        plt.xlabel('Predicted Values')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm_svm[i][j]))
        confusionMatrix = 'static/plots/confusionMatrix.png'
        plt.savefig(confusionMatrix)
        # plt.show()

        # from sklearn.metrics import classification_report, confusion_matrix
        # sns.heatmap(confusion_matrix(Y_test,sv_pred),annot=True,fmt='d')
        # heatmap_path = 'static/plots/heatmap.png'
        # plt.savefig(heatmap_path)
        # plt.show()

        input_data = (glevel,insulin,bmi,age)
        prediction = predict_diabetes_status(classifier, input_data)

        if prediction == 0:
            result  = 'THE PERSON IS NOT DIABETIC'
            # return('The person is not diabetic')
            return render_template('predict.html',s_username=s_username,result=result,outcome_path=outcome_path,histogram_path=histogram_path,scatter_path=scatter_path,heatmap_path=heatmap_path,confusionMatrix=confusionMatrix,table1=dataset_head.reset_index().values.tolist(),titles1 = dataset_column.values,table2=table2, titles2=titles2,table3=dataset_tail.reset_index().values.tolist(), titles3=dataset_column.values,info=dataset_info)
        else:
            # print('The person is diabetic')
            # return('The person is diabetic')
            result = 'THE PERSON IS DIABETIC'
            return render_template('predict.html',s_username=s_username,result=result,outcome_path=outcome_path,histogram_path=histogram_path,scatter_path=scatter_path,heatmap_path=heatmap_path,confusionMatrix=confusionMatrix,table1=dataset_head.reset_index().values.tolist(),titles1 = dataset_column.values,table2=table2, titles2=titles2,table3=dataset_tail.reset_index().values.tolist(), titles3=dataset_column.values,info=dataset_info)

    else:
        return('Error Occured !')
@app.route('/logout')
@login_required
def logout():
    session.pop('email',None)
    logout_user()
    return redirect(url_for('index'))
    # render_template('login.html')

if __name__ == '__main__':
	app.run(debug=True)