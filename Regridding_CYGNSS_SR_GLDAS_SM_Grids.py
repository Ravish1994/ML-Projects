from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def SR_CYGNSS(path,lat,lon):
    Cygnss_csv1 = pd.read_csv(path)
    mask = ((Cygnss_csv1['ddm_snr']>2) & (Cygnss_csv1['sp_rx_gain']>0) & (Cygnss_csv1['sp_inc_angle']<40))
    Cygnss_csv = Cygnss_csv1[mask]
    Data = Cygnss_csv
    SR2 = Surface_Reflectivity2(Data)             # Calculating Effective Surface Reflectivity
    DF2 = pd.DataFrame(Cygnss_csv['sp_lat'])
    DF2.columns = ['sp_lat']
    DF2['sp_lon'] = Cygnss_csv['sp_lon']
    DF2['SR_eff'] = SR2
    mask1 = ((DF2['SR_eff']<20) & (DF2['SR_eff']>0))
    DF3 = DF2[mask1]
    S_1 = DF3
    mask = ((S_1['sp_lat']>(lat-0.18)) & (S_1['sp_lat']<(lat+0.18)) & (S_1['sp_lon']>(lon-0.18)) & (S_1['sp_lon']<(lon+0.18)))
    S_1 = S_1[mask]
    SR = S_1['SR_eff']
    if len(SR)>0:
        SR = (np.sum(SR))/len(SR)
    else:
        SR = np.nan
    return SR

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

def GLDAS_Data(path,lat1,lon1):
    SM = Dataset(path,'r')                 # Importing the SM files  
    
    lon = np.array(SM['lon'])
    lat = np.array(SM['lat'])
    SoilMoist_S_tavg = pd.DataFrame(np.array(SM['SoilMoist_S_tavg'][0]))
    SoilMoist_S_tavg = np.array(SoilMoist_S_tavg.replace(to_replace=-9999,value=np.nan))

    Lon,Lat = np.meshgrid(lon,lat)  
    S_1 = pd.DataFrame(Lat.flatten())
    S_2 = Lon.flatten()
    SM  = SoilMoist_S_tavg.flatten() 
    
    # Creating dataframe of GLDAS_Lat, GLDAS_Lon and GLDAS_SM
    S_1.columns = ['Lat']
    S_1['lon'] = S_2
    S_1['SM'] = SM  
    mask = ((S_1['Lat']>(lat1-0.18)) & (S_1['Lat']<(lat1+0.18)) & (S_1['lon']>(lon1-0.18)) & (S_1['lon']<(lon1+0.18)) & (S_1['SM']>0))
    Df = S_1[mask]
    SM2 = np.array(Df['SM'])
    if len(SM2)>0:
        SM1 = float((np.sum(SM2))/len(SM2))
    else:
        SM1 = np.nan
    return SM1*100

def CYGNSS_GLDAS_Data_availability(m,n,lat,lon):
    CYGNSS_Data = []
    GLDAS_Data1 = []
    Day_No = []
    for i in range(m,n+1):
        Day_No1 = i
        Day_No.append(Day_No1)
        if i<=31:
            d = i
            if d <= 9:                
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_00{i}.csv'
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020010{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202001{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=60:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
            d = i-31
            if d <= 9:                
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020020{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202002{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=91:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
            d = i-60
            if d <= 9:                
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020030{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202003{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=121:
            d = i-91
            if i <= 99:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_0{i}.csv'
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020040{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
                if d<=9:
                    path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020040{d}.022.nc4.SUB.nc4'
                    
                    SR = SR_CYGNSS(path1,lat,lon)
                    SM = GLDAS_Data(path2,lat,lon)
                    CYGNSS_Data.append(SR)
                    GLDAS_Data1.append(SM)
                    
                else:
                    path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202004{d}.022.nc4.SUB.nc4'
                    
                    SR = SR_CYGNSS(path1,lat,lon)
                    SM = GLDAS_Data(path2,lat,lon)
                    CYGNSS_Data.append(SR)
                    GLDAS_Data1.append(SM)
                    
        elif i<=152:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-121
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020050{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202005{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=182:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-152
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020060{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202006{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=213:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-182
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020070{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202007{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=244:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-213
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020080{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202008{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=274:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-244
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020090{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202009{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=305:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-274
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020100{d}.022.nc4.SUB.nc4'
                
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202010{d}.022.nc4.SUB.nc4'               
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=335:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-305
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020110{d}.022.nc4.SUB.nc4'              
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202011{d}.022.nc4.SUB.nc4'              
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
        elif i<=366:
            path1 = f'D:\EG\Project Data\CYGNSS_CSV_Files\Day_{i}.csv'
            d = i-335
            if d <= 9:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A2020120{d}.022.nc4.SUB.nc4'              
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
                
            else:
                path2 = f'D:\EG\Project Data\GLDAS_SM\GLDAS_SM_Ganga_Catchment\GLDAS_CLSM025_DA1_D.A202012{d}.022.nc4.SUB.nc4'              
                SR = SR_CYGNSS(path1,lat,lon)
                SM = GLDAS_Data(path2,lat,lon)
                CYGNSS_Data.append(SR)
                GLDAS_Data1.append(SM)
    
    Day_Nodf = pd.DataFrame(Day_No)
    Day_Nodf.columns = ['Day_No']
    Day_Nodf['Cygnss_SR'] = pd.DataFrame(CYGNSS_Data)
    Day_Nodf['GLDAS_SM'] = pd.DataFrame(GLDAS_Data1)/1000
    Day_Nodf.to_csv(rf'D:\EG\Project Data\CYGNSS_Obs_Chambal_{lat}_{lon}\Annual_Variations_{lat}_{lon}.csv',index = False)
    Day_Nodf.to_csv(rf'D:\EG\Project Data\CYGNSS_Obs_Chambal_{lat}_{lon}\GLDAS_SM_{lat}_{lon}_Day_1-366.csv',index = False)