import pandas as pd
import numpy as np
import pickle
import datetime
import json
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print('Reading in data')
df = pd.read_csv('data.csv', skiprows=1, skipfooter=1, header=None, engine='python')

#rename the columns
df = df.iloc[:,0:18]
df.columns = ['HDF', 'date', 'half_hour_increment',
              'CCGT', 'OIL', 'COAL', 'NUCLEAR',
              'WIND', 'PS', 'NPSHYD', 'OCGT',
              'OTHER', 'INTFR', 'INTIRL', 'INTNED',
               'INTEW', 'BIOMASS', 'INTEM']

#Create a new column datetime that represents the starting datetime of the measured increment
df['datetime'] = pd.to_datetime(df['date'], format="%Y%m%d")
df['datetime'] = df.apply(lambda x:x['datetime']+ datetime.timedelta(minutes=30*(int(x['half_hour_increment'])-1)), 
                          axis = 1)
#Filter columns down and rename
df = df[['datetime', 'CCGT']].rename(columns = {'datetime':'ds', 'CCGT':'y'})

#Loading in saved model
print('Loading saved model')
loaded_model = pickle.load(open('/mnt/model.pkl','rb'))

#Take sample of rows from df
n = sys.argv[1]
print('Making predictions on {} rows...'.format(n))
df_samp = df[0:int(n)]

#Create feature and target sets
X = pd.DataFrame(data = df_samp.ds.apply(lambda x : str(x.date())))
y = df_samp.y

#Predict input features
model_predictions = loaded_model.predict(X).yhat

#Compute mean absolute error
print("Mean Absolute Error: {}".format(mean_absolute_error(y, model_predictions).round(2)))

#Compute residuals
residuals = y-model_predictions


#Creating Fitted vs residuals plot


plt.figure(figsize=(6,4))
sns.regplot(residuals, model_predictions,\
scatter_kws={"color": "blue"}, line_kws={"color": "orange"})
plt.title('Fitted vs Residuals plot')
plt.ylabel('Residuals')
plt.xlabel('Fitted Values')
plt.savefig('results/FittedVsResiduals.png')
plt.show()

#Write mean absolute error to dominostats file for plotting in experiment manager
results_dict = {}
results_dict['mean_absolute_error'] = mean_absolute_error(y, model_predictions).round(2)

with open('dominostats.json', 'w') as f:
    f.write(json.dumps(results_dict))
    
print('Done!')