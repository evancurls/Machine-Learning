#Evan Inrig
#ICP Wavelength Model
#2/3/2025

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR  # imported sklearn tool
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


#file names
for dirname, _, filenames in os.walk('input here'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#assign file paths
train_path = 'filename here'
test_path = 'filename here'

#open specific file
with open(train_path, 'r') as file:
    trainData = json.load(file)


with open(test_path, 'r') as file:
    testData = json.load(file)

#first 5 lines
first_lines = testData[:5]

#for i,entry in enumerate(first_lines, start = 1):
    #print(f"Row {i}: {entry}")
  
  
#reference plots    
numPlot = 0

for i in range(10):
    icp_wave = trainData[i]['icpWave']  # Extract the icpWave list
    plt.plot(icp_wave, label=f'Patient {trainData[i]["patientId"]}')  # Plot each icpWave

# Customize the plot
    plt.title('ICP Waveforms')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.scatter(trainData[i]["l1"], icp_wave[round(trainData[i]["l1"])], color='red', label='L1', s=50, edgecolor='black')   
    plt.scatter(trainData[i]["l2"], icp_wave[round(trainData[i]["l2"])], color='green', label='L2', s=50, edgecolor='black')   
    plt.scatter(trainData[i]["l3"], icp_wave[round(trainData[i]["l3"])], color='orange', label='L3', s=50, edgecolor='black')   

# Show the plot
    plt.show()
    
    
# Find the maximum length of icpWave
max_length = max(len(entry['icpWave']) for entry in trainData)

# Prepare data
X = []  # will hold processed icpWaves
y_l1, y_l2, y_l3 = [], [], []  # Targets (l1, l2, l3)

for entry in trainData:
    # Pad or truncate each icpWave to match max_length
    icp_wave = entry['icpWave']

    padded_wave = np.pad(icp_wave, (0, max_length - len(icp_wave)), mode='constant')[:max_length]
    X.append(padded_wave)  # Append the padded/truncated waveform

    y_l1.append(entry['l1'])  # Append l1 as the target
    y_l2.append(entry['l2'])  # Append l2 as the target
    y_l3.append(entry['l3'])  # Append l3 as the target

# Convert to NumPy arrays
X = np.array(X)
y_l1 = np.array(y_l1)
y_l2 = np.array(y_l2)
y_l3 = np.array(y_l3)

# Create a pipeline for each model 
pipeline_l1 = make_pipeline(StandardScaler(), SVR(kernel='linear'))
pipeline_l2 = make_pipeline(StandardScaler(), SVR(kernel='linear'))

#we want linear for l3
pipeline_l3 = make_pipeline(StandardScaler(), SVR(kernel='linear'))

# Train the models
pipeline_l1.fit(X, y_l1)
pipeline_l2.fit(X, y_l2)
pipeline_l3.fit(X, y_l3)

# Prepare test data
Xtest = []  # Features (icpWave)

for entry in testData:
    # Pad or truncate each icpWave to match max_length
    icp_wave = entry['icpWave']

    #uncomment later
    padded_wave = np.pad(icp_wave, (0, max_length - len(icp_wave)), mode='constant')[:max_length]
    Xtest.append(padded_wave)  # Append the padded/truncated waveform

# Predict on the test set  
y_pred_l1 = pipeline_l1.predict(Xtest)
y_pred_l2 = pipeline_l2.predict(Xtest)
y_pred_l3 = pipeline_l3.predict(Xtest)

test_df = pd.DataFrame(testData)

output = pd.DataFrame({'ID': test_df["ID"], 
                       'l1': y_pred_l1, 
                       'l2': y_pred_l2, 
                       'l3': y_pred_l3})


graph = output
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
    
    
    
numTest = 0
#graph['ID'][numTest]
for numTest in range(15):
    icp_wave = testData[numTest]['icpWave']  # Extract the icpWave list
    plt.plot(icp_wave, label=f'Patient {testData[numTest]["patientId"]}')  # Plot each icpWave



# Customize the plot
    plt.title('ICP Waveforms')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)


#real points on graph in test
#plt.scatter(testData[0]["l1"], icp_wave[round(testData[0]["l1"])], color='green', label='L1', s=50, edgecolor='black')   
#plt.scatter(testData[0]["l2"], icp_wave[round(testData[0]["l2"])], color='green', label='L2', s=50, edgecolor='black')   
#plt.scatter(testData[0]["l3"], icp_wave[round(testData[0]["l3"])], color='green', label='L3', s=50, edgecolor='black')


#predicted points on graph 
    plt.scatter(graph['l1'][numTest], icp_wave[round(graph['l1'][numTest])], color='red', label='L1(predict)', s=50, edgecolor='black')   
    plt.scatter(graph['l2'][numTest], icp_wave[round(graph['l2'][numTest])], color='red', label='L2(predict)', s=50, edgecolor='black')   
    plt.scatter(graph['l3'][numTest], icp_wave[round(graph['l3'][numTest])], color='red', label='L3(predict)', s=50, edgecolor='black')

    plt.show()    