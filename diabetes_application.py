import pandas as pd
import numpy as np
import tkinter as TK
from tkinter import messagebox
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data/diabetes_data.csv")

x = data.iloc[:,[1,2,3,4,5,6,7]] #used to predict
y = data.iloc[:,[8]] #value to predict


model = KNeighborsClassifier()

model.fit(x,y)
print("Enter Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age 'with comma':")
my_ins = list(input().split(","))

pred = model.predict([my_ins])

output = ""
if pred == [1]:
	output = "You Have Diabetes."
else:
	output = "You do not Have Diabetes."


root = TK.Tk()
root.withdraw()
messagebox.showinfo("Output", output)
root.withdraw()

