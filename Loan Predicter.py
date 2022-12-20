#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np


# In[3]:


df = pd.read_csv("train_csv.csv", )



# In[3]:





# Male = 0, Female = 1
# No  = 0, Yes = 1
# Not Graduate = 0, Graduate = 1
# Applicant Income in thousaands per anum
# Coapplicant Income in thousaands per anum(0 if no co apllicant)
# Urban = 0, Rural = 1

# In[4]:


df['Gender'] = df['Gender'].map({"Male":0, "Female":1})
df['Married'] = df['Married'].map({"No":0, "Yes":1})
df['Dependents'] = df['Dependents'].map({"0":0, "1":1, "2":2, "3+":3})
df['Education'] = df['Education'].map({"Not Graduate":0, "Graduate":1 })
df['Self_Employed'] = df['Self_Employed'].map({"No":0, "Yes":1})
df['Property_Area'] = df['Property_Area'].map({"Urban":0, "Rural":1})
df['Loan_Status'] = df['Loan_Status'].map({"N":0, "Y":1})



# In[20]:





# In[5]:


df['Gender'] = df['Gender'].fillna(df['Gender'].mean())
df['Married'] = df['Married'].fillna(df['Married'].mean())
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mean())
df['Education'] = df['Education'].fillna(df['Education'].mean())
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mean())
df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mean())
df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean())
df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean())
df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mean())
df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean())
df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean())
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())


# In[33]:





# In[6]:


x_values = df[['Gender','Married','Dependents','Education',"Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Property_Area"]]
y_values = df[['Loan_Status']]


# In[7]:


dt = tree.DecisionTreeClassifier()
dt = dt.fit(x_values,y_values)


# In[8]:


df2 = pd.read_csv("test.csv.csv", )



# In[9]:


df2['Gender'] = df2['Gender'].map({"Male":0, "Female":1})
df2['Married'] = df2['Married'].map({"No":0, "Yes":1})
df2['Dependents'] = df2['Dependents'].map({"0":0, "1":1, "2":2, "3+":3})
df2['Education'] = df2['Education'].map({"Not Graduate":0, "Graduate":1 })
df2['Self_Employed'] = df2['Self_Employed'].map({"No":0, "Yes":1})
df2['Property_Area'] = df2['Property_Area'].map({"Urban":0, "Rural":1})



# In[10]:


df2['Gender'] = df2['Gender'].fillna(df2['Gender'].mean())
df2['Married'] = df2['Married'].fillna(df2['Married'].mean())
df2['Dependents'] = df2['Dependents'].fillna(df2['Dependents'].mean())
df2['Education'] = df2['Education'].fillna(df2['Education'].mean())
df2['Self_Employed'] = df2['Self_Employed'].fillna(df2['Self_Employed'].mean())
df2['Property_Area'] = df2['Property_Area'].fillna(df2['Property_Area'].mean())
df2['ApplicantIncome'] = df2['ApplicantIncome'].fillna(df2['ApplicantIncome'].mean())
df2['CoapplicantIncome'] = df2['CoapplicantIncome'].fillna(df2['CoapplicantIncome'].mean())
df2['Property_Area'] = df2['Property_Area'].fillna(df2['Property_Area'].mean())
df2['ApplicantIncome'] = df2['ApplicantIncome'].fillna(df2['ApplicantIncome'].mean())
df2['CoapplicantIncome'] = df2['CoapplicantIncome'].fillna(df2['CoapplicantIncome'].mean())
df2['LoanAmount'] = df2['LoanAmount'].fillna(df2['LoanAmount'].mean())
df2['Loan_Amount_Term'] = df2['Loan_Amount_Term'].fillna(df2['Loan_Amount_Term'].mean())


# In[25]:





# In[11]:


x_test = df2[['Gender','Married','Dependents','Education',"Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Property_Area"]]


# In[12]:


y_predict = dt.predict(x_test)


# In[13]:


'''accuracy = []
depth = []

# Use ii to cycle through values 1 to 9. This will be the max_depth value for the decision tree. 
for ii in range(2,11):
    # Set max_depth to ii
    dt = tree.DecisionTreeClassifier(min_samples_split=ii)
    # Training or fitting the model with the data
    dt.fit(x_values,y_values)
    # .score provides the accuracy of the model based on the testing data. Store the accuracy into the list.
    accuracy.append(dt.score(x_test,y_predict))
    # Append the max_depth values to a list
    depth.append(ii)

print(accuracy)
print(y_predict)'''


# In[43]:


'''from keras.models import Sequential
from keras.layers import Dense


# In[64]:


model = Sequential()

# Add the first hidden layer with 6 nodes. Input_dim refers to the number of columns/number of features in x_values or the input layer.
# Activation refers to how the nodes/neurons are activated. We will use relu. Other common activations are 'sigmoid' and 'tanh'
model.add(Dense(6,input_dim=10,activation='relu'))

# Add the hidden layer with 6 nodes. 
model.add(Dense(6,activation='relu'))
model.add(Dense(6,activation='relu'))
# Add the output layer with 3 nodes. The activation used has to be 'softmax'. Softmax is used when you are dealing with categorical outputs or targets. 
model.add(Dense(1,activation='softmax'))

# Compile the model together. The optimizer refers to the method to make the adjustment within the model. Loss refers to how the difference between the predicted out 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[65]:


model.fit(x_values,y_values,epochs=20,shuffle=True)'''


# In[13]:


test = pd.DataFrame()
print('''Male = 0, Female = 1
No  = 0, Yes = 1
Not Graduate = 0, Graduate = 1
Applicant Income in thousaands per anum
Coapplicant Income in thousaands per anum(0 if no co apllicant)
Urban = 0, Rural = 1''')
test['Gender'] = [(input("Gender - "))]
test['Married'] = [float(input("Married - "))]
test['Dependents'] = [float(input("Dependents - "))]
test['Education'] = [float(input("Education - "))]
test['Self_Employed'] = [float(input("Self_Employed - "))]
test['ApplicantIncome'] = [float(input("ApplicantIncome - "))]
test['CoapplicantIncome'] = [float(input("CoapplicantIncome - "))]
test['LoanAmount'] = [float(input("LoanAmount - "))]
test['Loan_Amount_Term'] = [float(input("Loan_Amount_Term - "))]
test['Property_Area'] = [float(input("Property_Area - "))]
print(dt.predict(test))
u = (dt.predict(test))
if u == [0]:
   print("loan won't be given")
elif u == [1]:
   print("loan will be given")


# In[31]:




# In[ ]:




