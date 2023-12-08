import json
from flask import Flask, render_template, request, redirect,session, url_for
import bcrypt
from flask_mail import *
from random import randint
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

otp =  str(secrets.randbelow(10**6)).zfill(6)

@app.route('/')
def index():
	return render_template('login.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = db.users.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['s_username'] = username
            user_obj=User(username=user.get('username'))
            login_user(user_obj )
            return render_template('/home.html',s_username=username)
            # return redirect(url_for('home'))
        else:
            return render_template('/login.html',msg="Password or Username is Incorrect.")
    else:
        return render_template('login.html')
    #
    # print(usernmae)
    # print(password)
    #
    # main.py
    # --------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
#     return render_template('login.html')

def user_exists(email): #username
    # Query the collection for a document with the given name and mobile_number
    query = {"email" :email} #"username": username,
    result = db.users.find_one(query)
    # If result is not None, a matching document exists
    return result is not None

@app.route('/register', methods=['GET', 'POST'])
def register():
    display = ""
    if request.method == 'POST':
        # Handle the registration form submission here
        # def user_exists(username):
        #     # Query the collection for a document with the given name and mobile_number
        #     query = {"username": username}
        #     result = collection.find_one(query)
        #     # If result is not None, a matching document exists
        #     return result is not None

        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        mob = request.form['mob']
        gender = request.form['gender']

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # print(username,password,name,email,mob,gender)

        if user_exists(email): #username,
            # print(f"User with name '{username}' and mobile number '{mob}' already exists.")
            display ="block"
            return render_template("register.html",block=display)
        else:
            # post = {"username": username, "password": hashed_password,"name":name,"email":email,"mob":mob,"gender":gender}
            # collection.insert_one(post)
            msg= Message('OTP',sender='g37.dypcet@gmail.com',recipients=[email])
            msg.body=str(otp)
            mail.send(msg)
            session['username'] = username
            session['hashed_password'] = hashed_password
            session['name'] = name
            session['email'] = email
            session['mob'] = mob
            session['gender'] = gender

            # validate(username,hashed_password,name,email,mob,gender)
            return render_template("verify.html")
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    s_username = session.get('s_username', 'User')
    if request.method == 'POST':
        uploadedFile = request.files['uploadFile']
        file_path = f'content/{uploadedFile.filename}'
        uploadedFile.save(file_path)
    return render_template('home.html',s_username=s_username)

@app.route('/email',methods = ['GET','POST'])
def email():
    return render_template('email.html')
@app.route('/validate',methods=['GET', 'POST'])
def validate():
    username = session.get('username',None)
    hashed_password = session.get('hashed_password', None)
    name = session.get('name', None)
    email = session.get('email', None)
    mob = session.get('mob', None)
    gender = session.get('gender', None)

    userOtp = request.form['otp']
    if otp==userOtp:
        post = {"username": username, "password": hashed_password,"name":name,"email":email,"mob":mob,"gender":gender}
        collection.insert_one(post)
        print(username, hashed_password, name, email, mob, gender)
        return render_template("success.html")
    else:
        return redirect('register')

@app.route('/aboutUs')
@login_required
def aboutUs():
    s_username = session.get('s_username','User')
    return render_template('aboutUs.html',s_username=s_username)

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