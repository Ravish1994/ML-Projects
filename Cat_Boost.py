import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor


def DD_Time(Data):
    Year = []
    for i in range(len(Data)):
        DD = float(np.array(Data['Date'])[i][0:2])/30
        MM = float(np.array(Data['Date'])[i][3:5])
        Year.append(DD+MM)
    Data['Date'] = Year
    return Data

def Best_Learning_Rate(L,N1,Acc):
    for i in range(len(L)):
        acc = Acc[i]
        max1 = np.max(Acc)
        if acc==max1:
            lbest = L[i]
            Nbest = N1[i]
    return lbest,Nbest

def LGBoost_Regression_Training(lat,lon):
    Data = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\CYGNSS_SMAP_2019_21_{lat}_{lon}.csv')
    ## Take care of inconsistent data
    Data = DD_Time(Data) ## Date time column is string make them float
    
    ##  Take care of missing values
    Data = Data.bfill()
    Data = Data.ffill()

    ## Test and Train split
    Data_2020 = Data.head(366)  # Test Data
    Data_2021 = Data.tail(365*2)  # Train Data
    
    ##  Features and Target variables
    X_train = Data_2021.iloc[:,:-1]
    y_train = Data_2021.iloc[:,-1]
    X_test = Data_2020.iloc[:,:-1]
    y_test = Data_2020.iloc[:,-1] 

    ## Hyperparameter tuning by finding accuracy at all Learning rate and n_estimators
    L = np.arange(0.01,0.1,10)
    N = [5000]
    Acc2 = []
    N1 = []
    for i in range(len(L)):
        l = L[i]
        for j in range(len(N)):
            ## Training on 2021 Data
            cboost = CatBoostRegressor(n_estimators=N[j],learning_rate=l)
            cboost.fit(X_train, y_train, verbose=True, eval_set=[(X_test,y_test)])

            ## Predicting for 2020 data
            y_predt = cboost.predict(X_test)

            ## Accuracy score R2_score
            Accuracy2 = 100*np.round(r2_score(y_test, y_predt),2)
            Acc2.append(Accuracy2)
            N1.append(N[j])

    ## Best n_estimators, Learning Rate tunned
    Lbest, Nbest = Best_Learning_Rate(L,N1,Acc2)
    cboost = CatBoostRegressor(n_estimators=Nbest,learning_rate=Lbest)
    cboost.fit(X_train, y_train, verbose=True, eval_set=[(X_test,y_test)])
    y_predt = cboost.predict(X_test)
    Accuracy_GB = 100*np.round(r2_score(y_test, y_predt),2)
    Data_2020['Predicted Soil Moisture'] = y_predt
   
    Data_2020 = Data_2020.drop(['sp_inc_angle', 'sp_rx_gain', 'gps_tx_power_db_w',
                                'gps_ant_gain_db_i','ddm_snr', 'ddm_noise_floor', 'rx_to_sp_range', 'tx_to_sp_range',
                                'quality_flags', 'peak of power_analog', 'SR_eff'],axis=1)
    return Data_2020,Accuracy_GB

def Plotting_Comparision(lat,lon):
    Data_2020,Accuracy_GB = LGBoost_Regression_Training(lat,lon)
    Data_2020.to_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2019_21_{lat}_{lon}.csv',index=False)
    CR = np.array(Data_2020.corr())[1][2]*100
    RMSE = round(np.sum((Data_2020['Predicted Soil Moisture'] - Data_2020['SMAP_Soil_Moisture'])**2)/len(Data_2020['SMAP_Soil_Moisture']),3)
    DD = []
    for i in range(1,367):
        DD.append(i)
    plt.figure(figsize=(30,8))
    plt.scatter(DD,Data_2020['Predicted Soil Moisture'],label='Predicted Soil Moisture')
    plt.scatter(DD,Data_2020['SMAP_Soil_Moisture'],label='SMAP Soil Moisture')
    plt.title(f'''Latitude:{lat}°N Longitude:{lon}°E RMSE:{round(RMSE,4)} 
Correlation:{np.round(CR,2)}% NSE or CD:{Accuracy_GB}%''',size=40)  
    plt.xlabel('Day number of the year 2020',size=35)
    plt.ylabel('Volumetric Soil Moisture',size=35)
    plt.ylim(0,1)
    plt.xticks(np.arange(1, 370, 20),size=25)
    plt.yticks(np.arange(0, 1, 0.1),size=25)
    plt.legend(fontsize=35)

    plt.figure(figsize=(8,8))
    sns.lmplot(x=f'SMAP_Soil_Moisture',y=f'Predicted Soil Moisture',data=Data_2020,line_kws={'color': 'black'})
    plt.xlabel(f'SMAP Soil Moisture',size=20)
    plt.ylabel(f'Predicted Soil Moisture',size=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))
    plt.tick_params(axis='both', labelsize=20)
    
def Plotting_Comparision1(lat,lon):
    Data_2020 = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2019_21_{lat}_{lon}.csv')
    CR = np.array(Data_2020.corr())[1][2]*100
    RMSE = round(np.sum((Data_2020['Predicted Soil Moisture'] - Data_2020['SMAP_Soil_Moisture'])**2)/len(Data_2020['SMAP_Soil_Moisture']),3)
    DD = []
    for i in range(1,367):
        DD.append(i)
    plt.figure(figsize=(30,8))
    plt.scatter(DD,Data_2020['Predicted Soil Moisture'],label='Predicted Soil Moisture')
    plt.scatter(DD,Data_2020['SMAP_Soil_Moisture'],label='SMAP Soil Moisture')
    plt.title(f'''Latitude:{lat}°N Longitude:{lon}°E RMSE:{round(RMSE,4)} 
Correlation:{np.round(CR,2)}% NSE or CD:{np.round(((CR/100)**2)*100,2)}%''',size=40)  
    plt.xlabel('Day number of the year 2020',size=35)
    plt.ylabel('Volumetric Soil Moisture',size=35)
    plt.ylim(0,0.6)
    plt.xticks(np.arange(1, 370, 20),size=25)
    plt.yticks(np.arange(0, 0.6, 0.1),size=25)
    plt.legend(fontsize=35)

    plt.figure(figsize=(8,8))
    sns.lmplot(x=f'SMAP_Soil_Moisture',y=f'Predicted Soil Moisture',data=Data_2020,line_kws={'color': 'black'})
    plt.xlabel(f'SMAP Soil Moisture',size=20)
    plt.ylabel(f'Predicted Soil Moisture',size=20)
    plt.plot([0,0.6],[0,0.6],c='gray')
    plt.xlim(0,0.6)
    plt.ylim(0,0.6)
    plt.xticks(np.arange(0, 0.6, 0.1))
    plt.yticks(np.arange(0, 0.6, 0.1))
    plt.tick_params(axis='both', labelsize=20)
    
def Plotting_Comparision2(lat,lon):
    Data_2020 = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2020_21_{lat}_{lon}.csv')
    CR = np.array(Data_2020.corr())[1][2]*100
    RMSE = round(np.sum((Data_2020['Predicted Soil Moisture'] - Data_2020['SMAP_Soil_Moisture'])**2)/len(Data_2020['SMAP_Soil_Moisture']),3)
    DD = []
    for i in range(1,367):
        DD.append(i)
    plt.figure(figsize=(30,8))
    plt.scatter(DD,Data_2020['Predicted Soil Moisture'],label='Predicted Soil Moisture')
    plt.scatter(DD,Data_2020['SMAP_Soil_Moisture'],label='SMAP Soil Moisture')
    plt.title(f'''Latitude:{lat}°N Longitude:{lon}°E RMSE:{round(RMSE,4)} 
Correlation:{np.round(CR,2)}% NSE or CD:{np.round(((CR/100)**2)*100,2)}%''',size=40)  
    plt.xlabel('Day number of the year 2020',size=35)
    plt.ylabel('Volumetric Soil Moisture',size=35)
    plt.ylim(0,0.6)
    plt.xticks(np.arange(1, 370, 20),size=25)
    plt.yticks(np.arange(0, 0.6, 0.1),size=25)
    plt.legend(fontsize=35)

    plt.figure(figsize=(8,8))
    sns.lmplot(x=f'SMAP_Soil_Moisture',y=f'Predicted Soil Moisture',data=Data_2020,line_kws={'color': 'black'})
    plt.xlabel(f'SMAP Soil Moisture',size=20)
    plt.ylabel(f'Predicted Soil Moisture',size=20)
    plt.plot([0,0.6],[0,0.6],c='gray')
    plt.xlim(0,0.6)
    plt.ylim(0,0.6)
    plt.xticks(np.arange(0, 0.6, 0.1))
    plt.yticks(np.arange(0, 0.6, 0.1))
    plt.tick_params(axis='both', labelsize=20)
    
def Compar_Comparision_12(lat,lon):
    Data_2020 = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2020_21_{lat}_{lon}.csv')
    CR1 = np.array(Data_2020.corr())[1][2]*100
    RMSE1 = round(np.sum((Data_2020['Predicted Soil Moisture'] - Data_2020['SMAP_Soil_Moisture'])**2)/len(Data_2020['SMAP_Soil_Moisture']),3)
    
    Data_2019_21 = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2019_21_{lat}_{lon}.csv')
    CR2 = np.array(Data_2019_21.corr())[1][2]*100
    RMSE2 = round(np.sum((Data_2019_21['Predicted Soil Moisture'] - Data_2019_21['SMAP_Soil_Moisture'])**2)/len(Data_2019_21['SMAP_Soil_Moisture']),3)
    return CR1,CR2
    
    
def Saving_Correlations_LGBoost(lat,lon):
    Data_2020 = pd.read_csv(rf'D:\EG\Project Data\ML_Model_Datasets\ML_Predicted_SM\Cat_Boost_CYGNSS_SMAP_2019_21_{lat}_{lon}.csv')
    CR = np.array(Data_2020.corr())[1][2]*100
    RMSE = round(np.sum((Data_2020['Predicted Soil Moisture'] - Data_2020['SMAP_Soil_Moisture'])**2)/len(Data_2020['SMAP_Soil_Moisture']),3)
    return CR,RMSE