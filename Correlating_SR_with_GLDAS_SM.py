import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
   Input : Path : Path of the file containing CYGNSS SR and GLDAS SM
           lat and lon: Centroid of latitude and longitude of the GLDAS single grid cell
           
   Output : Visualization of correlation

'''
        
def SR_GLDAS_SM_Corr(path,lat,lon):
    Dataset = pd.read_csv(path)
    Dataset = Dataset.dropna()
    
    y_pred = np.array((Dataset.drop(['Day_No','GLDAS_SM'],axis=1))/20)     # Independet variable
    y_act  = np.array((Dataset['GLDAS_SM'])/1000)                          # dependent variable
    
    # Calculation of RMSE
    act   = np.array(pd.DataFrame(y_act).replace(to_replace=np.nan,value=0))
    pred  = np.array(pd.DataFrame(y_pred).replace(to_replace=np.nan,value=0))
    
    if len(act)>0:
        rmse  = mean_squared_error(act, pred, squared=False)
    else:
        rmse  = np.nan
    
    # R_square
    if len(act)>0:
        R2_score = r2_score(act,pred)
    else:
        R2_score = np.nan
        
    Dataset['Cygnss_SR'] = Dataset['Cygnss_SR']/20
    Dataset['GLDAS_SM']  = Dataset['GLDAS_SM']/1000
    df1 = (Dataset.drop(['Day_No'],axis=1))
    df1  = df1.dropna()        
    CR = (np.array(df1.corr()))*100
    CR1 = np.round(CR[0][1],2)
    
    a1 = Dataset['Cygnss_SR'] # Observed Data
    a2 = Dataset['GLDAS_SM']  # Simulated Data
    denominator = np.sum((a1 - np.mean(a1))**2)
    numerator   = np.sum((a2 - a1)**2)
    nse_val     = 1 - (numerator/denominator)
    
    plt.figure(figsize=(30,10))
    plt.scatter(Dataset['Day_No'],Dataset['Cygnss_SR'],label = 'Normalized CYGNSS Backscatter')
    plt.scatter(Dataset['Day_No'],Dataset['GLDAS_SM'],label  = 'GLDAS Soil Moisture')
    y_lab = '''Normalized Surface reflectiviy 
and Soil Moisture'''
    plt.ylabel(y_lab,fontsize=35)
    plt.xlabel('Days of Year 2020',fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.title(f'Latitude:{lat}??N Longitude:{lon}??E Correlation:{CR1}% RMSE:{round(rmse,4)} R2 Score: {np.round(CR1**2/10000,2)}',fontsize=40)
    plt.legend(fontsize=30) 
    
    import seaborn as sns
    plt.figure(figsize=(8,8))
    sns.lmplot(x='Cygnss_SR',y='GLDAS_SM',data=Dataset,line_kws={'color': 'black'})
    plt.xlabel('Normalized Surface Reflectivity',fontsize=20)
    plt.ylabel('GLDAS Soil Moisture',fontsize=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))  
    plt.tick_params(axis='both', labelsize=16)