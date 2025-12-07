import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier   

file = pd.read_csv(r"C:\Users\sumit\OneDrive\Documents\personalised_dataset.csv")
frame = pd.DataFrame(file)
print(frame)

label = LabelEncoder()
frame["Gender"] = label.fit_transform(frame["Gender"])
frame["Smoking_Status"] = label.fit_transform(frame["Smoking_Status"])
frame["Alcohol_Consumption"] = label.fit_transform(frame["Alcohol_Consumption"])
frame["Physical_Activity_Level"] = label.fit_transform(frame["Physical_Activity_Level"])
frame["Diet_Type"] = label.fit_transform(frame["Diet_Type"])
frame["Heart_Disease_Risk"] = label.fit_transform(frame["Heart_Disease_Risk"])
frame["Diabetes_Risk"] = label.fit_transform(frame["Diabetes_Risk"])
frame["Health_Risk"] = label.fit_transform(frame["Health_Risk"])
frame["Diet_Recommendation"] = label.fit_transform(frame["Diet_Recommendation"])

print(frame)

algo = MultiOutputClassifier(XGBClassifier())


inputs = frame[[
    "Age","Gender","BMI","Smoking_Status","Alcohol_Consumption",
    "Physical_Activity_Level","Diet_Type",
    "Systolic_BP","Diastolic_BP","Cholesterol","Glucose_Level","HbA1c"
]]

outputs = frame[[
    "Heart_Disease_Risk","Diabetes_Risk",
    "Health_Risk","Diet_Recommendation"
]]

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size=0.8, random_state=42)

algo.fit(x_train, y_train)

get = algo.predict(x_test)


age = float(input("Enter your Age = "))
gender = int(input("Enter your Gender (0=Female, 1=Male) = "))
bmi = float(input("Enter your BMI = "))

smoking_status = int(input("Enter Smoking Status (0=Non-Smoker, 1=Former, 2=Current) = "))
alcohol = int(input("Enter Alcohol Consumption (0=None, 1=Moderate, 2=Heavy) = "))
physical_activity = int(input("Enter Physical Activity Level (0=Low, 1=Medium, 2=High) = "))
diet_type = int(input("Enter Diet Type (0=Unhealthy, 1=Balanced, 2=Healthy) = "))

systolic_bp = float(input("Enter your Systolic BP = "))
diastolic_bp = float(input("Enter your Diastolic BP = "))
cholesterol = float(input("Enter your Cholesterol Level = "))
glucose = float(input("Enter your Glucose Level = "))
hba1c = float(input("Enter your HbA1c = "))

final = algo.predict([[
    age, gender, bmi, smoking_status, alcohol,
    physical_activity, diet_type, systolic_bp, diastolic_bp,
    cholesterol, glucose, hba1c
]])
print( final)

decode = label.inverse_transform([final[0][3]])[0]
    
label.fit(file["Heart_Disease_Risk"])
decode = label.inverse_transform([final[0][0]])[0]
print("heart_desease_risk", decode)
label.fit(file["Diabetes_Risk"])
decodes = label.inverse_transform([final[0][1]])[0]
print("diabetes_risk:",decodes)
label.fit(file["Diabetes_Risk"])
decodess = label.inverse_transform([final[0][2]])[0]
print("Diabetes_Risk",decodess)
label.fit(file["Diet_Recommendation"])
decodesss = label.inverse_transform([final[0][2]])[0]
print("diet_recommend:" , decodesss)
#done finaly time to watch reels 