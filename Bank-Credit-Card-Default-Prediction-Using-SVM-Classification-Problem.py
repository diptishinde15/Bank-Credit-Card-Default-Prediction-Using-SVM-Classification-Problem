import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/Dipti/Documents/IMR/Data_SVM/")

FullRaw = pd.read_csv('BankCreditCard.csv')


############################
# Sampling: Divide the data into Train and Testset
############################

from sklearn.model_selection import train_test_split
TrainRaw, TestRaw = train_test_split(FullRaw, train_size=0.7, random_state = 999)


# Create Source Column in both Train and Test
TrainRaw['Source'] = 'Train'
TestRaw['Source'] = 'Test'

# Combine Train and Test
FullRaw = pd.concat([TrainRaw, TestRaw], axis = 0)

# Check for NAs
FullRaw.isnull().sum()

# % Split of 0s and 1s
FullRaw.loc[FullRaw['Source'] == 'Train', 'Default_Payment'].value_counts()/FullRaw.loc[FullRaw['Source'] == 'Train'].shape[0]


# Summarize the data
FullRaw_Summary = FullRaw.describe()

# Lets drop 'Customer ID' column from the data 
FullRaw.drop(['Customer ID'], axis = 1, inplace = True) 

FullRaw.shape


############################
# Conversion of numeric variables to categorical variables
############################

# Categorical variables: Gender, Academic_Qualification, Marital

Variable_To_Update = 'Gender'
FullRaw[Variable_To_Update].value_counts() 
FullRaw[Variable_To_Update].replace({1:"Male", 
                                     2:"Female"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


Variable_To_Update = 'Academic_Qualification'
FullRaw[Variable_To_Update].value_counts()
FullRaw[Variable_To_Update].replace({1:"Undergraduate",
                                     2:"Graduate",
                                     3:"Postgraduate",
                                     4:"Professional",
                                     5:"Others",
                                     6:"Unknown"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


Variable_To_Update = 'Marital'
FullRaw[Variable_To_Update].value_counts()
FullRaw[Variable_To_Update].replace({1:"Married",
                                     2:"Single",
                                     3:"Unknown",
                                     0:"Unknown"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


#########################
## Combining "Academic_Qualification" variable categories using a function
#########################

tempDf = FullRaw[FullRaw["Source"] == "Train"]

# Step 1
propDf = pd.crosstab(tempDf["Academic_Qualification"], tempDf["Default_Payment"], margins = True)


# Step 2: Study the data to combine groups
propDf["Default_Prop"] = round(propDf[1]/propDf["All"],1)

# Step 3
FullRaw["AQ_New"] = np.where(FullRaw["Academic_Qualification"].isin(["Graduate", "Undergraduate", "Postgraduate"]),
                             "Group1", "Group2")
FullRaw["AQ_New"].unique()
del FullRaw["Academic_Qualification"]

FullRaw.columns


############################
# Dummy variable creation
############################

FullRaw2 = FullRaw.copy()
FullRaw2 = pd.get_dummies(FullRaw2, drop_first = False) 
FullRaw2.shape

############################
# Divide the data into Train and Test
############################

# Step 1: Divide into Train and Testest
Train = FullRaw2[FullRaw2['Source_Train'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
Test = FullRaw2[FullRaw2['Source_Test'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()


# Step 2: Divide into Xs (Independents) and Y (Dependent)
depVar = "Default_Payment"
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape


########################
# Modeling
########################

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#############
# Model1
#############

M1 = SVC() # Default kernel is 'rbf'
M1_Model = M1.fit(trainX, trainY) 

Test_Class = M1_Model.predict(testX)
Confusion_Mat = confusion_matrix(testY, Test_Class)
Confusion_Mat

# sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100
print(classification_report(testY, Test_Class))



########################
# Modeling using Random Search
########################

from sklearn.model_selection import RandomizedSearchCV
myCost = [0.1, 1, 2]
myKernel = ['sigmoid','rbf']
myCombo = len(myCost)*len(myKernel)

my_param_grid = {'C': myCost, 'kernel': myKernel}
SVM_RandomSearchCV = RandomizedSearchCV(SVC(), param_distributions = my_param_grid,  
                                        scoring = 'f1', cv = 3, n_jobs = -1, 
                                        n_iter = int(0.7*myCombo),
                                        random_state = 123).fit(trainX, trainY)
# Result in a dataframe
SVM_RandomSearch_Df = pd.DataFrame.from_dict(SVM_RandomSearchCV.cv_results_)


###################
###################
# 1. Modeling with Standardized Dataset
###################
################### 




from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(trainX)
trainX_Std = Train_Scaling.transform(trainX) 
testX_Std  = Train_Scaling.transform(testX) 

# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)



M4 = SVC()
M4_Model = M4.fit(trainX_Std, trainY) 

Test_Class = M4_Model.predict(testX_Std)
Confusion_Mat = confusion_matrix(testY, Test_Class)
Confusion_Mat

# sum(np.diagonal(Confusion_Mat))/testX_Std.shape[0]*100
print(classification_report(testY, Test_Class))


###################
###################
# 2. Stratified random sampling: Class  Imbalance/ Rare Events Handling using Under-sampling
###################
###################  


# Run this command in command prompt (cmd) or anaconda prompt: pip install imblearn
from imblearn.under_sampling import RandomUnderSampler 


# Count of 0s and 1s
trainY.value_counts()

RUS = RandomUnderSampler(sampling_strategy = 0.7, random_state = 123)

trainX_RUS, trainY_RUS = RUS.fit_resample(trainX_Std, trainY)
trainX_RUS = pd.DataFrame(trainX_RUS)
trainY_RUS = pd.Series(trainY_RUS)

# # Rename columns/headers for trainX and trainY



trainY_RUS.value_counts() # Count of 0s and 1s
trainY_RUS.value_counts()[1]/sum(trainY_RUS.value_counts())*100 # Percentage of 1s
trainY_RUS.value_counts()[1]/trainY_RUS.value_counts()[0] # Ratio of 1s to 0s


M5 = SVC()
M5_Model = M5.fit(trainX_RUS, trainY_RUS) 
Test_Class = M5_Model.predict(testX_Std)
Confusion_Mat = confusion_matrix(testY, Test_Class)
Confusion_Mat

# sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100
print(classification_report(testY, Test_Class))





###################
###################
# 3. Data Transformation/ Re-Distribution with Standardization (You can do mix & match!)
###################
###################

# Data transformation involves applying some mathemetical functions on the dataset (usually indep vars) 
           
import seaborn as sns
import numpy as np

trainX.columns
columnsToConsider = ["Credit_Amount", "Age_Years", "Jan_Bill_Amount", "Feb_Bill_Amount"]

# Histogram using seaborn
sns.pairplot(trainX[columnsToConsider])

# Lets consider Area Mean and apply log transformation
trainX_Copy = trainX.copy()
trainX_Copy["Age_Years"] = np.log(np.where(trainX_Copy["Age_Years"] == 0, 1, trainX_Copy["Age_Years"]))
# np.log(trainX_Copy["Age_Years"]) # If there are no 0s present in Age_years

testX_Copy = testX.copy()
testX_Copy["Age_Years"] = np.log(np.where(testX_Copy["Age_Years"] == 0, 1, testX_Copy["Age_Years"]))

# Histogram using seaborn
sns.pairplot(trainX_Copy[columnsToConsider])


###################
# Standardization
###################

Train_Scaling = StandardScaler().fit(trainX_Copy) 
trainX_Std = Train_Scaling.transform(trainX_Copy) 
testX_Std  = Train_Scaling.transform(testX_Copy) 

# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)



###################
# Modeling
###################

M6 = SVC()
M6_Model = M6.fit(trainX_Std, trainY) 
Test_Class = M6_Model.predict(testX_Std)
Confusion_Mat = confusion_matrix(testY, Test_Class)
Confusion_Mat

# sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100
print(classification_report(testY, Test_Class))

    