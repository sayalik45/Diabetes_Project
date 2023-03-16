import numpy as np 
import pickle
import json

class diabetes_prediction():

    def __init__(self,data) :
        self.data = data

    def loading_files(self):

        with open(r'artifacts\dibetes_model.pkl','rb') as file :
            self.model = pickle.load(file)

        with open(r'artifacts\dibetes_scaler.pkl','rb') as file :
            self.scaler = pickle.load(file)

        with open(r'artifacts\project_data.json','r') as file :
            self.project_data = json.load(file)

    def diab_predict(self):
        self.loading_files()

        Glucose = self.data['html_Glucose']
        BloodPressure = self.data['html_BloodPressure']
        SkinThickness = self.data['html_SkinThickness']
        Insulin = self.data['html_Insulin']
        BMI = self.data['html_BMI']
        DiabetesPedigreeFunction = self.data['html_DiabetesPedigreeFunction']
        Age = self.data['html_Age']

        user_data = np.zeros(len(self.project_data['columns']))
        user_data[0] = Glucose
        user_data[1] = BloodPressure
        user_data[2] = SkinThickness
        user_data[3] = Insulin
        user_data[4] = BMI
        user_data[5] = DiabetesPedigreeFunction
        user_data[6] = Age

        scaler = self.scaler.fit_transform([user_data])
        result = self.model.predict(scaler)[0]
        return result