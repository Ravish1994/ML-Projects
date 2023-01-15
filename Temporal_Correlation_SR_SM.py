from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statistics

'''--------------------------------------------Function to Determine Surface Reflectivity-------------------------------------------'''

def Surface_Reflectivity2(Data):
    P1  =  Data['gps_ant_gain_db_i'][:]           # Gain of the transmitting antenna:(in dBi(Decibel Isotropic))    
    P2  =  Data['sp_rx_gain'][:]                  # Gain of the recieving antenna:(in dBi)    
    P3  =  Data['tx_to_sp_range'][:]              # Distance(m), specular point to the transmitter    
    P4  =  Data['rx_to_sp_range'][:]              # Distance(m), specular point to the reciever    
    P5  =  Data['gps_tx_power_db_w'][:]           # GPS transmitting power(RHCP Power in dB)           
    DDM_peak =  Data['peak of power_analog'][:]   # 17 Delay × 11 Doppler(in Watt) bins corresp. to 50 km^2    
    P_r = DDM_peak*90
    T_rl = (P_r*(4*np.pi*(P3+P4))**2)/(P5*P1*P2*(0.19)**2)  # Effective Surface Reflectivity in dB
    return T_rl


'''
   Input : path: Path of the CYGNSS Daily Raw files
           lat and lon: Centroid of latitude and longitude of the SMAP single grid cell
   
   Output : Averaged Surface Reflectivity and its Standard Deviation within that SMAP grid cell

'''
def SR_CYGNSS(path,lat,lon):
    
    ## Reading raw data file of CYGNSS
    Cygnss_csv1 = pd.read_csv(path)
    
    ## Masking the Raw data files by removing outliers
    # DDM_SNR        > 2
    # SP_Rx_Gain     > 0
    # SP_inclination < 40
    
    mask = ((Cygnss_csv1['ddm_snr']>2) & (Cygnss_csv1['sp_rx_gain']>0) & (Cygnss_csv1['sp_inc_angle']<40))
    Cygnss_csv = Cygnss_csv1[mask]
    Data = Cygnss_csv
    
    # Calculating Effective Surface Reflectivity
    SR2 = Surface_Reflectivity2(Data)                   
    DF2 = pd.DataFrame(Cygnss_csv['sp_lat'])
    DF2.columns = ['sp_lat']
    DF2['sp_lon'] = Cygnss_csv['sp_lon']
    DF2['SR_eff'] = SR2
    
    # Masking SR values between 0 and 20 to remove outliers
    mask1 = ((DF2['SR_eff']<20) & (DF2['SR_eff']>0))    
    DF3 = DF2[mask1]
    S_1 = DF3
    
    # Averaging surface reflectivity values within SMAP grid cell of 36 Km with lat,lon as its centroid
    mask = ((S_1['sp_lat']>(lat-0.18)) & (S_1['sp_lat']<(lat+0.18)) & (S_1['sp_lon']>(lon-0.18)) & (S_1['sp_lon']<(lon+0.18)))
    S_1  = S_1[mask]
    SR1   = S_1['SR_eff']
    if len(SR1)>1:                             ## Ignoring the cells having less than 2 data points
        SR  = np.mean(np.array(SR1))
        STD = np.std(np.array(SR1))
    else:
        SR  = np.nan
        STD = np.nan
    return SR,STD


'''----------------------------------Taking out single SMAP Soil Moisture Data within that particular grid cell-------------------------'''

'''
   Input : path: Path of the SMAP Daily Raw files
           lat and lon: Centroid of latitude and longitude of the SMAP single grid cell
   
   Output : Single SMAP soil moisture within that SMAP grid cell

'''

def SMAP_Data(path,lat,lon):
    
    # Importing the SMAP hdf files 
    SMAP_data = h5py.File(path,'r')  
    
    # PM data
    df1     = SMAP_data['Soil_Moisture_Retrieval_Data_PM']
    SM_lat1 = np.array(pd.DataFrame(df1['latitude_pm']))
    SM_lon1 = np.array(pd.DataFrame(df1['longitude_pm']))
    SM1     = np.array(pd.DataFrame(df1['soil_moisture_pm']))
    S1      = pd.DataFrame(SM_lat1.reshape(406*964,1))
    S2      = pd.DataFrame(SM_lon1.reshape(406*964,1))
    SM_1    = pd.DataFrame(SM1.reshape(406*964,1))
    
    # AM data
    df2     = SMAP_data['Soil_Moisture_Retrieval_Data_AM']
    SM_lat2 = np.array(pd.DataFrame(df2['latitude']))
    SM_lon2 = np.array(pd.DataFrame(df2['longitude']))        
    SM2     = np.array(pd.DataFrame(df2['soil_moisture']))
    T1      = pd.DataFrame(SM_lat2.reshape(406*964,1))
    T2      = pd.DataFrame(SM_lon2.reshape(406*964,1))  
    SM_2    = pd.DataFrame(SM2.reshape(406*964,1))

    # Concatenating AM and PM data  
    S_1 = pd.concat([S1,T1])
    S_2 = pd.concat([S2,T2])
    SM  = pd.concat([SM_1,SM_2]) 
    
    # Creating dataframe of SMAP_Lat, SMAP_Lon and SMAP_SM
    S_1.columns = ['Lat']
    S_1['lon']  = S_2
    S_1['SM']   = SM  
    
    # Masking and taking out SMAP soil moisture of that particular cell
    mask = ((S_1['Lat']>(lat-0.18)) & (S_1['Lat']<(lat+0.18)) & (S_1['lon']>(lon-0.18)) & (S_1['lon']<(lon+0.18)) & (S_1['SM']>0))
    Df = S_1[mask]
    SM2 = np.array(Df['SM'])
    if len(SM2)>0:
        SM1 = np.mean(SM2)
    else:
        SM1 = np.nan
    return SM1*100


'''-------------------------Temporal Correlation between CYGNSS surface reflectivity with SMAP soil moisture---------------------------'''

'''
   Input : m: starting day of the year
           n: ending day of the year
           lat and lon: Centroid of latitude and longitude of the SMAP single grid cell
           
   Output : Saving CSV file for a single pixel containing Day_no, Standard deviation of SR in a particular cell with day and SM

'''

def CYGNSS_SMAP_Data_availability(m,n,lat,lon):
    CYGNSS_Data = []
    CYGNSS_STD = []
    SMAP_Data1 = []
    Day_No = []
    for i in range(m,n+1):
        Day_No1 = i
        Day_No.append(Day_No1)
        if i<=31:
            d = i
            if d <= 9:                
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_00{i}.csv'
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_01\SMAP_L3_SM_P_2020010{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM    = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_01\SMAP_L3_SM_P_202001{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=60:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
            d = i-31
            if d <= 9:                
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_02\SMAP_L3_SM_P_2020020{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_02\SMAP_L3_SM_P_202002{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=91:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
            d = i-60
            if d <= 9:                
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_03\SMAP_L3_SM_P_2020030{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_03\SMAP_L3_SM_P_202003{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=121:
            d = i-91
            if i <= 99:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_04\SMAP_L3_SM_P_2020040{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
                if d<=9:
                    path2 = f'D:\EG\Project Data\SMAP_DATA\Month_04\SMAP_L3_SM_P_2020040{d}_R17000_001.h5'
                    
                    SR,SD = SR_CYGNSS(path1,lat,lon)
                    SM = SMAP_Data(path2,lat,lon)
                    CYGNSS_Data.append(SR)
                    CYGNSS_STD.append(SD)
                    SMAP_Data1.append(SM)
                    
                else:
                    path2 = f'D:\EG\Project Data\SMAP_DATA\Month_04\SMAP_L3_SM_P_202004{d}_R17000_001.h5'
                    
                    SR,SD = SR_CYGNSS(path1,lat,lon)
                    SM = SMAP_Data(path2,lat,lon)
                    CYGNSS_Data.append(SR)
                    CYGNSS_STD.append(SD)
                    SMAP_Data1.append(SM)
                    
        elif i<=152:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-121
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_05\SMAP_L3_SM_P_2020050{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_05\SMAP_L3_SM_P_202005{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=182:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-152
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_06\SMAP_L3_SM_P_2020060{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_06\SMAP_L3_SM_P_202006{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=213:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-182
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_07\SMAP_L3_SM_P_2020070{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_07\SMAP_L3_SM_P_202007{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=244:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-213
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_08\SMAP_L3_SM_P_2020080{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_08\SMAP_L3_SM_P_202008{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=274:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-244
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_09\SMAP_L3_SM_P_2020090{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_09\SMAP_L3_SM_P_202009{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=305:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-274
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_10\SMAP_L3_SM_P_2020100{d}_R17000_001.h5'
                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_10\SMAP_L3_SM_P_202010{d}_R17000_001.h5'                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=335:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-305
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_11\SMAP_L3_SM_P_2020110{d}_R17000_001.h5'                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_11\SMAP_L3_SM_P_202011{d}_R17000_001.h5'                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
        elif i<=366:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-335
            if d <= 9:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_12\SMAP_L3_SM_P_2020120{d}_R17000_001.h5'               
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM    = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\SMAP_DATA\Month_12\SMAP_L3_SM_P_202012{d}_R17000_001.h5'                
                SR,SD = SR_CYGNSS(path1,lat,lon)
                SM    = SMAP_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                CYGNSS_STD.append(SD)
                SMAP_Data1.append(SM) 
    
    Day_Nodf              = pd.DataFrame(Day_No)
    Day_Nodf.columns      = ['Day_No']
    Day_Nodf['Cygnss_SR'] = pd.DataFrame(CYGNSS_Data)
    Day_Nodf['Cygnss_SD'] = pd.DataFrame(CYGNSS_STD)
    Day_Nodf['SMAP_SM']   = pd.DataFrame(SMAP_Data1)
    Day_Nodf.to_csv(f'Annual_Variations_{lat}_{lon}.csv',index = False)


'''
   Input : path of all csv files saved previously
           n: numbers of files 
           
   Output : Visualizing Correlation with band plots of standard deviation

''' 
    
def Plotting_Variations_with_band(lat,lon):
    Path1          = f'D:\EG\Project Data\Variability_SMAP_SM_CYGNSS_SReff_2020\Annual_Variations_{lat}_{lon}.csv'
    df             = pd.read_csv(Path1)
    Day_No         = df['Day_No']
    CYGNSS_Data    = (df['Cygnss_SR']-np.min(df['Cygnss_SR']))/(np.max(df['Cygnss_SR'])-np.min(df['Cygnss_SR']))
    std1           = (df['Cygnss_SD']-np.min(df['Cygnss_SD']))/(np.max(df['Cygnss_SD'])-np.min(df['Cygnss_SD']))
    SMAP_Data1     = df['SMAP_SM']    
    df1            = pd.DataFrame(CYGNSS_Data)
    df1.columns    = ['CYGNSS_SR']
    SMAP_Data1     = df['SMAP_SM']/100
    df1['SMAP_SM'] = SMAP_Data1
    df1            = df1.dropna()
    CR             = (np.array(df1.corr()))*100
    CR1            = np.round(CR[0][1],2)
        
    act   = pd.DataFrame(SMAP_Data1).replace(to_replace=np.nan,value=0)
    pred  = pd.DataFrame(CYGNSS_Data).replace(to_replace=np.nan,value=0)
    rmse  = mean_squared_error(act/100, pred, squared=False)
    
    act = np.array(act)
    a1 = (act-np.min(act))/(np.max(act)-np.min(act))    
    a2 = np.array(pred)  
    denominator = np.sum((a1 - np.mean(a1))**2)
    numerator   = np.sum((a2 - a1)**2)
    nse_val     = 1 - (numerator/denominator)
        
    plt.figure(figsize=(30,10))
    plt.plot(Day_No,CYGNSS_Data,label    = 'Normalized CYGNSS Derived Surface Reflectivity',color='blue')
    plt.fill_between(Day_No,(CYGNSS_Data-std1),(CYGNSS_Data+std1),color='grey')
    plt.scatter(Day_No,SMAP_Data1,label = 'SMAP Soil Moisture',color='black')
    y_lab = '''Normalized Surface reflectiviy 
and Soil Moisture'''
    plt.ylabel(y_lab,fontsize=35)
    plt.xlabel('Days of Year 2020',fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.title(f'Latitude:{lat}°N Longitude:{lon}°E Correlation:{CR1}% RMSE:{round(rmse,4)} R2_Score: {np.round((CR1/100)**2,2)}',fontsize=40)
    plt.legend(fontsize=30) 
        
    plt.figure(figsize=(8,8))
    sns.lmplot(x='SMAP_SM',y='CYGNSS_SR',data=df1,line_kws={'color': 'black'})
    plt.ylabel('''Normalized Surface 
Reflectivity''',fontsize=20)
    plt.xlabel('SMAP Soil Moisture',fontsize=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))  
    plt.tick_params(axis='both', labelsize=16)
    
def Plotting_Variations(lat,lon):
    Path1             = f'D:\EG\Project Data\CYGNSS_Obs_Chambal_{lat}_{lon}\Annual_Variations_{lat}_{lon}.csv'
    df_1              = pd.read_csv(Path1)
    df_1   = df_1.head(365)
    Path2             = f'D:\EG\Project Data\CYGNSS_Data_in_0p36Dg\SMAP_RF_SM\SMAP_SM_Variations_{lat}_{lon}.csv'
    df_2              = pd.read_csv(Path2)
    df_2   = df_2.head(365)
    
    Keys1             = df_1.keys()
    for i in range(0,1):
        if 'Cygnss_SR' in Keys1:
            SR = 'Cygnss_SR'
        else:
            SR = 'CYGNSS_Backscatter'
    
    Day_No            = df_1['Day_No']
    sr                = df_1[f'{SR}']
    df_1['SMAP_SM']   = np.array(df_2['SMAP_SM']/100)
    df_1[f'{SR}']     = (sr-np.min(sr))/(np.max(sr)-np.min(sr)) 
    CR   = np.round(np.array(df_1.corr())[1][3]*100,2)
    RMSE = round((np.sum((df_1[f'{SR}'] - df_1['SMAP_SM'])**2)/len(df_1[f'{SR}']))**0.5,3)
    
    a1 = (df_1[f'{SR}']-np.min(df_1[f'{SR}']))/(np.max(df_1[f'{SR}'])-np.min(df_1[f'{SR}']))    # Observed Data
    a2 = df_1['SMAP_SM']  # Simulated Data
    denominator = np.sum((a1 - np.mean(a1))**2)
    numerator   = np.sum((a2 - a1)**2)
    nse_val     = 1 - (numerator/denominator)

    plt.figure(figsize=(30,10))
    plt.scatter(Day_No,df_1[f'{SR}'],label     = 'Normalized CYGNSS Derived Surface Reflectivity',color='blue')
    plt.scatter(Day_No,df_1['SMAP_SM'],label = 'SMAP Soil Moisture',color='black')
    y_lab = '''Normalized Surface reflectiviy 
and Soil Moisture'''
    plt.ylabel(y_lab,fontsize=30)
    plt.xlabel('Days of Year 2020',fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(f'Latitude:{lat} Longitude:{lon} Correlation:{CR}% RMSE:{round(RMSE,4)} NSE: {np.round(nse_val,2)}',fontsize=30)
    plt.legend(fontsize=35) 

    plt.figure(figsize=(2,2))
    sns.lmplot(x='SMAP_SM',y=f'{SR}',data=df_1,line_kws={'color': 'black'})
    plt.ylabel('''Normalized Surface 
Reflectivity''',fontsize=20)
    plt.xlabel('SMAP Soil Moisture',fontsize=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))    
    plt.tick_params(axis='both', which='major', labelsize=20)
    
def CR_RMS(lat,lon):
    Path1  = f'D:\EG\Project Data\CYGNSS_Obs_Chambal_{lat}_{lon}\Annual_Variations_{lat}_{lon}.csv'
    df_1   = pd.read_csv(Path1)
    df_1   = df_1.head(365)
    Path2  = f'D:\EG\Project Data\CYGNSS_Data_in_0p36Dg\SMAP_RF_SM\SMAP_SM_Variations_{lat}_{lon}.csv'
    df_2   = pd.read_csv(Path2)
    df_2   = df_2.head(365)
    
    path3  = f'D:\EG\Project Data\LAI_2018\Rolling_LAI30Days\LAI_Data_Within_36_Km_Resolution_Cell_{lat}_{lon}.csv'
    df_3   = pd.read_csv(path3)
    df_3   = df_3.head(365)
    
    Keys1             = df_1.keys()
    for i in range(0,1):
        if 'Cygnss_SR' in Keys1:
            SR = 'Cygnss_SR'
        else:
            SR = 'CYGNSS_Backscatter'
    
    Day_No            = df_1['Day_No']
    sr                = df_1[f'{SR}']
    df_1['SMAP_SM']   = np.array(df_2['SMAP_SM']/100)
    df_1[f'{SR}']     = (sr-np.min(sr))/(np.max(sr)-np.min(sr)) 
    RMSE = round((np.sum((sr - df_1['SMAP_SM'])**2)/len(sr))**0.5,3)
    CR   = np.round(np.array(df_1.corr())[1][3]*100,2)
    RMS  = round((np.sum((df_1['SMAP_SM'])**2)/len(df_1[f'{SR}']))**0.5,3)
    RMS_LAI = round((np.sum((df_3['LAI'])**2)/len(df_1[f'{SR}']))**0.5,3)
    return CR,RMS,RMS_LAI,RMSE