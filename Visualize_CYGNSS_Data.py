from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
import geopandas as gpd 
import xarray as xr

'''String definition for the day number in month(28 or 30 or 31)'''
def fun4(i):
    if i<=9:
        b = '0'+str(i)
        return b
    elif i>=10 and i <=31:
        b = str(i)
        return b
    
'''String definition for the month number'''
def fun3(i):
    if i<=9:
        b = '0'+str(i)
        return b
    elif i>=10 and i <=12:
        b = str(i)
        return b    
'''String definition for the day number in month(28 or 30 or 31)'''
def fun4(i):
    if i<=9:
        b = '0'+str(i)
        return b
    elif i>=10 and i <=31:
        b = str(i)
        return b
    
''' Formatting Date '''    
def fun5(i):
    if (i>=1) and (i<=31):
        b1 = fun3(1)                           ## Month Number(Jan)
        b2 = fun4(i)                           ## Day number in month
        c = b1+b2
                
    elif (i>=32) and (i<=59):
        b1 = fun3(2)                           ## Month Number(Feb)
        b2 = fun4(i-31)                           ## Day number in month
        c = b1+b2
                
    elif (i>=60) and (i<=90):
        b1 = fun3(3)                           ## Month Number(Mar)
        b2 = fun4(i-59)                        ## Day number in month
        c = b1+b2
                
    elif (i>=91) and (i<=120):
        b1 = fun3(4)                           ## Month Number(Apr)
        b2 = fun4(i-90)                        ## Day number in month
        c = b1+b2
        
    elif (i>=121) and (i<=151):
        b1 = fun3(5)                           ## Month Number(May)
        b2 = fun4(i-120)                       ## Day number in month
        c = b1+b2
        
    elif (i>=152) and (i<=181):
        b1 = fun3(6)                           ## Month Number(Jun)
        b2 = fun4(i-151)                       ## Day number in month
        c = b1+b2
        
    elif (i>=182) and (i<=212):
        b1 = fun3(7)                           ## Month Number(Jul)
        b2 = fun4(i-181)                       ## Day number in month
        c = b1+b2
        
    elif (i>=213) and (i<=243):
        b1 = fun3(8)                           ## Month Number(Aug)
        b2 = fun4(i-212)                       ## Day number in month
        c = b1+b2
        
    elif (i>=244) and (i<=273):
        b1 = fun3(9)                           ## Month Number(Sept)
        b2 = fun4(i-243)                       ## Day number in month
        c = b1+b2
        
    elif (i>=274) and (i<=304):
        b1 = fun3(10)                          ## Month Number(Oct)
        b2 = fun4(i-273)                       ## Day number in month 
        c = b1+b2
        
    elif (i>=305) and (i<=334):
        b1 = fun3(11)                          ## Month Number(Nov)
        b2 = fun4(i-304)                       ## Day number in month
        c = b1+b2
        
    elif (i>=335) and (i<=365):
        b1 = fun3(12)                          ## Month Number(Dec)
        b2 = fun4(i-334)                       ## Day number in month
        c = b1+b2
    return c

def CYGNSS_Daily_Data(d):
    data = pd.DataFrame()
    Path = rf'D:\EG\Project Data\CYGNSS_Raw_Ganga_Catchment_2021\Day_{d}'
    for i in range(1,9):
        nc = xr.open_dataset(rf'{Path}\cyg0{i}.ddmi.s2021{fun5(d)}-000000-e2021{fun5(d)}-235959.l1.power-brcs.a30.d31.nc.nc')
        nc.to_dataframe().to_csv(rf'{Path}\cyg0{i}.ddmi.s2021{fun5(d)}-000000-e2021{fun5(d)}-235959.l1.power-brcs.a30.d31.csv')
        data1 = pd.read_csv(rf'{Path}\cyg0{i}.ddmi.s2021{fun5(d)}-000000-e2021{fun5(d)}-235959.l1.power-brcs.a30.d31.csv')
        data  = pd.concat([data, data1], axis=0)
    return data  

fp = r'indian_districts.shp'
map_df = gpd.read_file(fp) 
map_df_msk = map_df[((map_df['latitude']>22) & (map_df['latitude']<27)) & ((map_df['longitude']>72) & (map_df['longitude']<79))]
DF3 = map_df_msk
DF3 = DF3[DF3['state nam0']!='Uttar Pradesh']
DF3 = DF3[DF3['state nam0']!='Chhattisgarh']

def SR_gridded_Avg(Lat,Lon,DF,Var):
    mask = ((DF['sp_lat']>(Lat-0.18)) & (DF['sp_lat']<(Lat+0.18))) & ((DF['sp_lon']>(Lon-0.18)) & (DF['sp_lon']<(Lon+0.18)))
    Df1 = DF[mask]  
    SR  = np.array(Df1[f'{Var}'])
    SR1 = np.mean(SR)
    if SR1==np.nan:
        SR2 = 0
    else:
        SR2 = SR1
    return SR2

def Averaging_Gridd(DF1):
    Lat1 = np.arange(22,27,0.36)
    Lon1 = np.arange(73,79,0.36)

    SP_lat = []
    SP_lon = []
    SP_SR  = []  # CYGNSS values within 10 Km grid cell
    for i in range(len(Lat1)):
        Lat = Lat1[i]
        for j in range(len(Lon1)):
            Lon = Lon1[j]
            SR  = SR_gridded_Avg(Lat,Lon,DF1,'ddm_peak')
            
            SP_lat.append(Lat)
            SP_lon.append(Lon)
            SP_SR.append(SR)
            
    DF = pd.DataFrame(SP_lat)
    DF.columns = ['SP_lat']
    DF['SP_lon'] = SP_lon
    DF['SP_SR'] = pd.DataFrame(SP_SR).replace(np.nan,0)
    return DF

def Plotting_CYGNSS(d):
    data = CYGNSS_Daily_Data(d)
    DDM = np.array(data['ddm_peak'])
  
    fig , ax1 = plt.subplots(figsize=(30, 15))
    DF3.plot(color = 'white', edgecolor = 'black',axes=ax1)
    for i in range(len(data)):
        plt.scatter(np.array(data['sp_lon'])[i],np.array(data['sp_lat'])[i],s=5,c='black',alpha=0.5)
    plt.xlabel('Longitude',fontsize=25)
    plt.ylabel('Lattitude',fontsize=25)
    plt.title(f'CYGNSS Track:2021-{fun5(d)[0:2]}-{fun5(d)[2:]},Number of Samples: {len(data)}',fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=15)
    for i in range(len(DF3)):
        X = DF3['longitude'].iloc[i]
        Y = DF3['latitude'].iloc[i]
        S = DF3['district n'].iloc[i]
        S = S.replace("*","")
        if (X>73) & (X<78.5):
            plt.text(X,Y,S[9:],fontsize=15,alpha=1)

    data1 = Averaging_Gridd(data)
    m = len(np.arange(22,27,0.36))
    n = len(np.arange(73,79,0.36))
    Lat = np.array(data1['SP_lat']).reshape(m,n)
    Lon = np.array(data1['SP_lon']).reshape(m,n)
    DDM = np.array(data1['SP_SR']).reshape(m,n)

    fig , ax1 = plt.subplots(figsize=(30, 15))
    DF3.plot(color = 'white', edgecolor = 'black',axes=ax1)
    plt.pcolor(Lon,Lat,DDM,cmap="gist_ncar",alpha=0.8)
    DF3.plot(color = 'white', edgecolor = 'black',axes=ax1,alpha=0.1)
    cbar = plt.colorbar(shrink=1)
    cbar.set_label(f'gps_tx_power_db_w in Watt')
    plt.xlabel('Longitude',fontsize=25)
    plt.ylabel('Lattitude',fontsize=25)
    plt.title(f'CYGNSS data withing 36 Km Grid:2021-{fun5(d)[0:2]}-{fun5(d)[2:]}',fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=15)

    for i in range(len(DF3)):
        X = DF3['longitude'].iloc[i]
        Y = DF3['latitude'].iloc[i]
        S = DF3['district n'].iloc[i]
        S = S.replace("*","")
        if (X>73) & (X<78.5):
            plt.text(X,Y,S[9:],fontsize=20,alpha=1)