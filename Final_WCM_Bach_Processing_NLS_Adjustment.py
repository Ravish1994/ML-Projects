import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import sympy
from sympy import *
import matplotlib.pyplot as plt

# Creating Variables
Sv, l, c, d, Mvt, Sgma = symbols('Sv l c d Mvt Sgma')

# Modified Water Cloud Model Equation
f   = Sgma-(Sv*l+(c*Mvt+d)*(1-l))

# Model Parameters (n, u, r, P)
def Adjustment_Parameters1(Sgma0,DD):
    P = np.identity(len(Sgma0))         # Weight matrix (Precision matrix)
    n = len(Sgma0)                      # As total n erroneous observations are there   
    # We have to calculate the model parameters and soil moisture 
    u = 2+2*len(DD) # C,D,Svt6,Svt8,...,Svt365,Mvt6,Mvt8,...,Mvt365 Assume those day SM only having LAI
    r = n-u         # Redundancy
    return P,n,u,r

# Generating Design Matrix
def Der(f,var):
    A1 = (Derivative(f,var)).doit()
    return A1

def Design_Matrix1(Df1,Sv,c,d,Mvt,DD):
    C1 = Der(f,Sv)
    C2 = Der(f,c)
    C3 = Der(f,d)
    C4 = Der(f,Mvt)
    A1 = []
    for j in range(len(Df1)):
        Single_Row_A = []
        for i in range(len(DD)):
            d1 = np.array(DD)[i]
            d2 = int(np.array(Df1['Day_No'])[j])
            if d1 == d2:
                Single_Row_A += [C1]
            else:
                Single_Row_A += [0]
                
        Single_Row_A += [C2]
        Single_Row_A += [C3]
        A1 += Single_Row_A
    A2 = np.array(A1)
    A1 = A2.reshape(len(Df1),(len(DD)+2))  
    return A1

def Design_Matrix2(Df1,Sv,c,d,Mvt,DD):
    C4 = Der(f,Mvt)
    A1 = []
    for j in range(len(Df1)):
        Single_Row_A = []
        for i in range(len(DD)):
            d1 = np.array(DD)[i]
            d2 = int(np.array(Df1['Day_No'])[j])
            if d1 == d2:
                Single_Row_A += [C4]
            else:
                Single_Row_A += [0]
        A1 += Single_Row_A
    A2 = np.array(A1)
    A1 = A2.reshape(len(Df1),len(DD))  
    return A1

def Design_Matrix(Df1,Sv,c,d,Mvt,DD):
    A1 = Design_Matrix1(Df1,Sv,c,d,Mvt,DD)
    A1 = pd.DataFrame(A1)
    
    A2 = Design_Matrix2(Df1,Sv,c,d,Mvt,DD)
    A2 = pd.DataFrame(A2)
    
    A  = pd.concat([A1, A2], axis=1)
    A  = (np.array(A))
    return A

# Substituting Values in Design Matrix
def Resubstituting_Values(A1,Sgma0,L,S_v,SM_GLDAS,c0,d0,lat,lon):
    n1,n2 = A1.shape
    A_1   = []
    for i in range(n1):
        Sgma1  = np.array(Sgma0)[i]
        L1     = np.array(L)[i]
        Mv0    = np.array(SM_GLDAS)[i]
        Sv0    = np.array(S_v)[i]
        for j in range(n2):
            a1 = A1[i][j]
            if a1!=0:
                aij = a1.subs({Sgma:Sgma1,l:L1,Sv:Sv0,Mvt:Mv0,c:c0,d:d0})
                A_1.append(aij)
            else:
                A_1.append(0)
    A_1 = np.array(A_1)    
    A1  = A_1.reshape(n1,n2)
    A   = np.array(A1)
    A = pd.DataFrame(A)
    A.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\A_{lat}_{lon}.csv',index = False)
    A = pd.read_csv(rf'D:\EG\Project Data\Adjusted_Parameters\A_{lat}_{lon}.csv')
    A = (np.array(A))
    return A

## Residual vector
def Misclosure(Sgma0,L,S_v,SM_GLDAS,c0,d0,f,lat,lon):
    W_0 = []
    for i in range(len(Sgma0)):
        Sgma1  = np.array(Sgma0)[i]
        L1     = np.array(L)[i]
        Mv0    = np.array(SM_GLDAS)[i]
        Sv0    = np.array(S_v)[i]
        f5 = f.subs({Sgma:Sgma1,l:L1,Sv:Sv0,Mvt:Mv0,c:c0,d:d0})
        W_0.append(f5)
    W_0 = np.array(W_0)
    W1 = W_0.reshape(len(Sgma0),1)
    W = pd.DataFrame(W1)
    W.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\W_{lat}_{lon}.csv',index = False)
    W = pd.read_csv(rf'D:\EG\Project Data\Adjusted_Parameters\W_{lat}_{lon}.csv')
    W = (np.array(W))
    return W

# Converting the elements of the matrix from the object to the float
def Object_toFloat(Matrix):
    m,n = Matrix.shape
    M1 = []
    for i in range(m):
        for j in range(n):
            M1.append(float(Matrix[i,j]))
    M2 = np.array(M1)
    M = M2.reshape(m,n)
    return M

# Updating Soil Moisture
def New_SM(DF_batch,X_Sv,X_SM,idx):
    Mv0 = X_SM
    DF_batch1 = DF_batch.drop(['GLDAS_SM'], axis=1)
    SM_initialized = []
    for i in range(len(DF_batch1)):
        D  = int(np.array(idx)[i])
        SM = Mv0[D]
        SM_initialized.append(SM)
    SM_initialized = np.array(SM_initialized).reshape(len(SM_initialized),) 
    
    Mv0 = X_Sv
    DF_batch2 = DF_batch.drop(['Sv'], axis=1)
    Sv_initialized = []
    for j in range(len(DF_batch2)):
        D  = int(np.array(idx)[j])
        Sv = Mv0[D]
        Sv_initialized.append(Sv)
    Sv_initialized = np.array(Sv_initialized).reshape(len(Sv_initialized),) 
    return SM_initialized, Sv_initialized

# Performing more Iteration
def Adjustment_Parameters(DF_batch,idx,A1,Sgma0,L,X,f,P,u1,lat,lon):
    X_SM = X[u1+2:]
    X_Sv = X[:u1+2]
    GLDAS_SM,S_v = New_SM(DF_batch,X_Sv,X_SM,idx)
    A        = Resubstituting_Values(A1,Sgma0,L,S_v,GLDAS_SM,X[u1][0],X[u1+1][0],lat,lon)
    W        = Misclosure(Sgma0,L,S_v,GLDAS_SM,X[u1][0],X[u1+1][0],f,lat,lon)
    N        = np.array(np.transpose(A)@P@A)
    N        = pd.DataFrame(N)
    N.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv',index = False)
    N        = pd.read_csv(rf'''D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv''')
    N        = (np.array(N))
    U        = (np.transpose(A))@P@W
    N        = Object_toFloat(N)
    Part1    = np.linalg.pinv(N)       ## Inverse of N
    Part2    = Object_toFloat(Part1)   ## Converting elements to float
    dX       = -Part2@U                ## Change in X
    return A,W,N,U,dX

def Plotting_Variations(Df,Var1,Var2,label1,label2,lat,lon,RMSE,CR):
    plt.figure(figsize=(30,8))
    plt.scatter(Df['Day_No'],Df[f'{Var1}'],label = label1)
    plt.scatter(Df['Day_No'],Df[f'{Var2}'],label = label2)
    
    plt.title(f'''Latitude:{lat}°N Longitude:{lon}°E RMSE:{round(RMSE,4)}
Correlation:{CR}% NSE or CD:{np.round(CR**2/100,2)}%''',size=40) 
    plt.xlabel('Day number of the year 2020',size=35)
    plt.ylabel('Volumetric Soil Moisture',size=35)
    plt.ylim(0,1)
    plt.xticks(np.arange(1, 370, 20),size=25)
    plt.yticks(np.arange(0, 1, 0.1),size=25)
    plt.legend(fontsize=35)

    plt.figure(figsize=(3,3))
    plt.scatter(Df[f'{Var1}'],Df[f'{Var2}'],s=40)
    plt.xlabel(f'{Var1}')
    plt.ylabel(f'{Var2}')
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))
    
    plt.figure(figsize=(8,8))
    sns.lmplot(x=f'{Var2}',y=f'{Var1}',data=Df,line_kws={'color': 'black'})
    plt.ylabel(f'{Var1.replace("_SM"," Soil Moisture")}',size=20)
    plt.xlabel(f'{Var2.replace("_SM"," Soil Moisture")}',size=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))
    plt.tick_params(axis='both', labelsize=20)

# Correlation of WCM with GLDAS soil moisture
def New_SM1(DF_batch,X_SM,idx):
    Mv0 = X_SM
    SM_initialized = []
    for i in range(len(DF_batch)):
        D  = int(np.array(idx)[i])
        SM = float(Mv0[D])
        SM_initialized.append(SM)
    SM_initialized = np.array(SM_initialized).reshape(len(SM_initialized),) 
    DF_batch['WCM_SM'] = SM_initialized
    return DF_batch

def Minimum(DF_batch,D):
        DF = DF_batch[DF_batch['Day_No']==D]
        DF1 = DF['SR_eff']
        return np.min(DF1)
    
def Plotting_Var(Df,lat,lon):
    CR = np.array(Df.corr())[1][2]*100
    RMSE = round(np.sum((Df['SMAP_SM'] - Df['Improved_SM'])**2)/len(Df['SMAP_SM']),3)
    
    plt.figure(figsize=(30,8))
    plt.scatter(Df['Day_No'],Df['SMAP_SM'],label='SMAP Soil Moisture on the vegetated and Barren Land')
    plt.scatter(Df['Day_No'],Df['Improved_SM'],label='WCM Soil Moisture after adjustment')
    plt.title(f'''Latitude: {lat} Longitude: {lon} RMSE: {round(RMSE,4)} Correlation: {np.round(CR,2)} % NSE or CD: {np.round(CR**2/100,2)} %''',size=35) 
    plt.xlabel('Day number of the year 2020',size=35)
    plt.ylabel('Volumetric Soil Moisture',size=35)
    plt.ylim(0,1)
    plt.xticks(np.arange(1, 370, 20),size=35)
    plt.yticks(np.arange(0, 1, 0.1),size=35)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.legend(fontsize=35)

    plt.figure(figsize=(3,3))
    plt.scatter(Df['SMAP_SM'],Df['Improved_SM'],s=40)
    plt.xlabel(f'SMAP_SM')
    plt.ylabel(f'Improved_SM')
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))
   
    plt.figure(figsize=(8,8))
    sns.lmplot(x=f'SMAP_SM',y=f'Improved_SM',data=Df,line_kws={'color': 'black'})
    plt.xlabel(f'SMAP Soil Moisture',size=20)
    plt.ylabel(f'Adjusted WCM Soil Moisture',size=20)
    plt.plot([0,1],[0,1],c='gray')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 1, 0.2))
    plt.tick_params(axis='both', labelsize=20)
    

def Bach_Adjustment_For_366_Days_WCM(lat,lon):
    # Collecting whole year data for a single Pixel Point
    import Preparing_Dataset_for_BatchProcessing
    import imp
    imp.reload(Preparing_Dataset_for_BatchProcessing)
    import Preparing_Dataset_for_BatchProcessing as r5

    DF_batch = r5.Data_Batch_2020(lat,lon)
    DF_batch = DF_batch[DF_batch['LAI']<1]

    # GLDAS Soil Moisture Data for initialization purpose of 366 days of 2020 within a pixel of 36 x 36 Km of SMAP
    GLDAS_Path = f'D:\EG\Project Data\CYGNSS_Obs_Chambal_{lat}_{lon}'+f'\GLDAS_SM_{lat}_{lon}_Day_1-366.csv'
    GLDAS_SM1  = pd.read_csv(GLDAS_Path)
    GLDAS_SM1  = GLDAS_SM1.drop(np.array(GLDAS_SM1.keys())[:-1],axis=1)
    GLDAS_SM   = np.array(GLDAS_SM1['GLDAS_SM'])
    Mv0        = GLDAS_SM

    SM_initialized = []
    for i in range(len(DF_batch)):
        D = np.array(DF_batch['Day_No'])[i]-1
        SM_initialized.append(Mv0[int(D)])
    DF_batch['GLDAS_SM'] = SM_initialized

    Sv_initialized = []
    for j in range(len(DF_batch)):
        D = np.array(DF_batch['Day_No'])[j]
        DF_batch1 = DF_batch[(DF_batch['Day_No']==D)]
        Sv1 = Minimum(DF_batch,D)
        Sv_initialized.append(Sv1)
    DF_batch['Sv'] = Sv_initialized

    # Unique days
    Df1 = DF_batch
    DD  = (Df1['Day_No'])
    DD  = DD.drop_duplicates()

    # Collecting observations
    Sgma0    = Df1['SR_eff']
    L        = Df1['LAI']
    GLDAS_SM = Df1['GLDAS_SM']
    S_v      = Df1['Sv']
    Day_No   = Df1['Day_No']

    P,n,u,r  = Adjustment_Parameters1(Sgma0,DD)
    u1       = int((u-2)/2)
    A1       = Design_Matrix(Df1,Sv,c,d,Mvt,DD)
    c0       = 20
    d0       = -1

    DFT        = DF_batch.drop_duplicates(subset=['Day_No'])
    SM_initial = np.array(DFT['GLDAS_SM'])
    Sv_initial = np.array(DFT['Sv'])
    
    X0         = []
    for i in range(len(Sv_initial)):
        X0.append(Sv_initial[i])
        
    X0.append(c0)
    X0.append(d0)
    
    for i in range(len(SM_initial)):
        X0.append(SM_initial[i])

    A = Resubstituting_Values(A1,Sgma0,L,S_v,GLDAS_SM,c0,d0,lat,lon)
    W = Misclosure(Sgma0,L,S_v,GLDAS_SM,c0,d0,f,lat,lon)

    # Finding Normal Matrix
    N = np.array(np.transpose(A)@P@A)
    N        = pd.DataFrame(N)
    N.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv',index = False)
    N        = pd.read_csv(rf"D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv")
    N        = (np.array(N))
    N = Object_toFloat(N)
    U = (np.transpose(A))@P@W

    # Change in Parameters
    Part1  = np.linalg.pinv(N)       ## Inverse of N
    Part2  = Object_toFloat(Part1)   ## Converting elements to float
    dX     = -Part2@U                ## Change in X
    X0     = np.array(X0)
    X0     = X0.reshape(len(X0),1)
    X      = X0+dX

    # Creating Index for updating soil moisture
    Df  = pd.DataFrame(DF_batch['Day_No'].value_counts(sort=False))['Day_No']
    DD  = Df.values
    idx = []
    m   = 0
    for i in range(len(DD)):
        j = DD[i]
        for k in range(j):
            idx.append(m)
        m = m+1

    # 1st Iteration
    X_SM = X[u1+2:]
    X_Sv = X[:u1+2]
    GLDAS_SM,S_v = New_SM(DF_batch,X_Sv,X_SM,idx)
    A            = Resubstituting_Values(A1,Sgma0,L,S_v,GLDAS_SM,X[u1][0],X[u1+1][0],lat,lon)
    W            = Misclosure(Sgma0,L,S_v,GLDAS_SM,X[u1][0],X[u1+1][0],f,lat,lon)
    N            = np.array(np.transpose(A)@P@A)
    N        = pd.DataFrame(N)
    N.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv',index = False)
    N        = pd.read_csv(rf'D:\EG\Project Data\Adjusted_Parameters\N_{lat}_{lon}.csv')
    N        = (np.array(N))
    U            = (np.transpose(A))@P@W
    N            = Object_toFloat(N)
    Part1        = np.linalg.pinv(N)       ## Inverse of N
    Part2        = Object_toFloat(Part1)   ## Converting elements to float
    dX1          = -Part2@U                ## Change in X

    X = X+dX1
    dX = dX1
    for i in range(1,200):
        A,W,N,U,dX1 = Adjustment_Parameters(DF_batch,idx,A1,Sgma0,L,X,f,P,u1,lat,lon)
        Thres       = np.max(abs(dX-dX1))
        if Thres<10**(-5):
            A_adjusted  = A
            W_adjusted  = W
            N_adjusted  = N
            U_adjusted  = U
            X_adjusted  = X
            dX_adjusted = dX
            Number_Iteration = i
            break
        else:
            dX = dX1
            X  = X+dX
    A = pd.DataFrame(A_adjusted)
    W = pd.DataFrame(W_adjusted) 
    N = pd.DataFrame(N_adjusted)
    X = pd.DataFrame(X_adjusted)
    dX = pd.DataFrame(dX_adjusted)
    DD1 = DF_batch['Day_No'].drop_duplicates()
    DD  = DD1.values
    SM_adjusted = X[u1+2:].values
    Thres_Iter  = pd.DataFrame(np.array([Thres,Number_Iteration+1]))
    Thres_Iter.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\Thres_Iter_{lat}_{lon}.csv',index = False)
    dX.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\dX_adjusted_{lat}_{lon}.csv',index = False)
    X.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\X_adjusted_{lat}_{lon}.csv',index = False)
    
    Par_I = pd.DataFrame(['C','D'])
    Par_I.columns = ['Parameters']
    Par_I['Initialized Values'] = np.array([c0,d0])
    Par_I['Adjusted Values'] = np.array(X[u1:u1+2])
    Param = Par_I
    Param.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\Param_adjusted_{lat}_{lon}.csv',index = False)
    
    # Hypothesis Testing
    S   = np.transpose(W)@P@W/r       ## Sum of the square of the residuals
    S02 = np.std(Sgma0)               ## Standard deviation in original observed backscatter
    
    Test_Statistic = r*S02/S[0][0]
    Pr = r-1
    Hypothesis = pd.DataFrame([f'Test Statistic: {Test_Statistic}',f'Critical_Value: {Pr}'])
    Hypothesis.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\Hypothesis_{lat}_{lon}.csv',index = False)
    
    # Orthogonality Check
    V = np.transpose(A)@W 
    Orthogonality = pd.DataFrame([f'Orthogonality Check: ATW: {np.sum(V**2)} is approximately zero'])
    Orthogonality.to_csv(rf'D:\EG\Project Data\Adjusted_Parameters\Orthogonality_{lat}_{lon}.csv',index = False)
    
    # Correlation of WCM with SMAP soil moisture
    SMAP_SM = pd.read_csv(f'D:\EG\Project Data\CYGNSS_Data_in_0p36Dg\SMAP_RF_SM\SMAP_SM_Variations_{lat}_{lon}.csv')
    SMAP_SM['SMAP_SM'] = SMAP_SM['SMAP_SM']/100
    x2  = np.array(SMAP_SM['SMAP_SM'])
    
    # Taking pixel of SMAP on the vegetated and Barren Land Both region
    SMAP_SM_filter = []
    for j in range(len(DD)):
        D2 = int(DD[j])
        SMAP_SM_filter.append(np.array(SMAP_SM['SMAP_SM'])[D2-1])

    Df = pd.DataFrame(DD)
    Df.columns = ['Day_No'] 
    Df['SMAP_SM'] = SMAP_SM_filter
    Df['Improved_SM'] = SM_adjusted
    Df = Df[(Df['Improved_SM']<1) & (Df['Improved_SM']>=0)]
    Df.to_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\WCM_Modified_Adjusted_SM_{lat}_{lon}.csv',index = False)
    Plotting_Var(Df,lat,lon)
    
    # LAI index ploting
    d1   = 31
    LAI = pd.read_csv(f'D:\EG\Project Data\LAI_2018\Rolling_LAI30Days\LAI_Data_Within_36_Km_Resolution_Cell_{lat}_{lon}.csv')
    DF2  = LAI
    DF2[f'LAI{d1}']  = DF2['LAI'].rolling(d1).mean()
    DF2['Day_No']    = DF2['Day_No']-30
    DF2 = DF2.dropna()
    DF2.to_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\LAI_Rolling_Avg_31Day_{lat}_{lon}.csv',index = False)
    
    plt.figure(figsize=(30,8))
    plt.scatter(DF2['Day_No'],DF2[f'LAI{d1}'],label=f'{d1} Days Rolling Average LAI for Lat:{lat} Lon: {lon}')
    plt.scatter(DF2['Day_No'],DF2['LAI'],label='LAI')
    plt.xlabel('Day number of the year 2020',size=20)
    plt.ylabel('LAI in m2/m2',size=20)
    plt.ylim(0,1)
    plt.xticks(np.arange(1, 370, 10),size=15)
    plt.yticks(np.arange(0, 1, 0.2),size=15)
    plt.legend(fontsize=30)

    # Correlation of WCM with GLDAS soil moisture
    X_SM     = X[u1+2:].values
    GLDAS_WCM_SM  = New_SM1(DF_batch,X_SM,idx)
    GLDAS_WCM_SM1 = GLDAS_WCM_SM.drop(['SR_eff','SP_I','sp_lat','sp_lon','LAI'], axis=1)
    GLDAS_WCM_SM1 = GLDAS_WCM_SM1.drop_duplicates()
    GLDAS_WCM_SM1 = GLDAS_WCM_SM1[(GLDAS_WCM_SM1['WCM_SM']<1)]
    GLDAS_WCM_SM1 = GLDAS_WCM_SM1[(GLDAS_WCM_SM1['WCM_SM']>0)]
    GLDAS_WCM_SM1.to_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\GLDAS_WCM_Modified_Adjusted_SM_{lat}_{lon}.csv',index = False)
    CR = np.array(GLDAS_WCM_SM1.corr())
    CR = round((CR[1][3])*100,3)
    RMSE = round(np.sum((GLDAS_WCM_SM1['GLDAS_SM']-GLDAS_WCM_SM1['WCM_SM'])**2)/len(GLDAS_WCM_SM1['GLDAS_SM']),3)
    Plotting_Variations(GLDAS_WCM_SM1
                        ,'GLDAS_SM'
                        ,'WCM_SM'
                        ,'GLDAS Soil Moisture on Barren Land'
                        ,'After removing vegetation attenuation WCM soil moisture',lat,lon,RMSE,CR)

    # Correlation of SMAP with GLDAS soil moisture
    GLDAS_SM1['Day_No']  = pd.DataFrame(np.array(SMAP_SM['Day_No']))
    GLDAS_SM1['SMAP_SM'] = SMAP_SM['SMAP_SM']
    GLDAS_SM1.to_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\GLDAS_SMAP_SM_{lat}_{lon}.csv',index = False)
    CR = np.array(GLDAS_SM1.corr())
    CR = round((CR[0][2])*100,3)
    RMSE = round(np.sum((GLDAS_SM1['GLDAS_SM'] - GLDAS_SM1['SMAP_SM'])**2)/len(GLDAS_SM1['GLDAS_SM']),3)

    Plotting_Variations(GLDAS_SM1
                        ,'SMAP_SM'
                        ,'GLDAS_SM'
                        ,'SMAP Soil Moisture on Barren Land'
                        ,'GLDAS Soil Moisture on Barren Land'
                        ,lat,lon,RMSE,CR)
    
def Bach_Adjustment_Visualization(lat,lon):
    Df1    = pd.read_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\WCM_Modified_Adjusted_SM_{lat}_{lon}.CSV')
    Plotting_Var(Df1,lat,lon)
    GLDAS_SM1 = pd.read_csv(rf'D:\EG\Project Data\WCM_Adjusted_SM\WCM_Modified_Model_SM\GLDAS_SMAP_SM_{lat}_{lon}.csv')
    CR   = np.array(GLDAS_SM1.corr())
    CR   = round((CR[2][0])*100,3)
    print(CR)
    RMSE = round(np.sum((GLDAS_SM1['GLDAS_SM'] - GLDAS_SM1['SMAP_SM'])**2)/len(GLDAS_SM1['GLDAS_SM']),3)
    Plotting_Variations(GLDAS_SM1
                        ,'GLDAS_SM'
                        ,'SMAP_SM'
                        ,'GLDAS Soil Moisture on Barren Land'
                        ,'SMAP Soil Moisture on Barren Land',lat,lon,RMSE,CR)