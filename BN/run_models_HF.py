# basic stuff
import numpy as np
import pandas as pd
import matplotlib.dates as dates
# z score stuff
import scipy.stats as stats
import scipy.stats as ss
# plots for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import random
from ms_toolbox import tools as tls

# models to test out
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.dummy import DummyRegressor
from sklearn import tree
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
#from sklearn.model_selection import
from sklearn.utils import shuffle
from sklearn.neural_network._multilayer_perceptron import MLPRegressor
from sklearn.model_selection import train_test_split
import sys
import matplotlib.dates as dates
sys.path.append("/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/tools/ML")
from ML_functions import *
# sys.path.append('/Users/miks/Desktop/MOP/TorreyPines/Yates')
# from yates_paper_functions import *
sys.path.append('/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/tools/Yates')
from yates2008_functions import *
sys.path.append('/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/tools/general')
from tools import *

plt.ion()
# import the data into the workspace

mopnum = 551
froot = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/BN/"
figpath = froot + "run_model/figures/MOP%s/"%mopnum #figure directory
fpath = froot + "features/CSV_Nino-HF/" #csv directory
showvalidation = False
fpathparams = froot + "optimization/optimized_parameters/HF/"

wpath = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/waves/"
smpath = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/BW_obs/data_processed/"

########################################### Import Dataset ###########################################
# data = pd.read_csv(fpath + "MOP581_MLfiles.csv")
data = pd.read_csv(fpath + "MOP%s_MLfiles_ver_eof.csv" %mopnum)

############################################# modify data ############################################
# high frequency obs
kkHF = (data["IHF"] == True)
# low frequency obs
kkLF = (data["IHF"] == False)

dataV = data[kkHF].copy() #daily obs period, data for testing
data = data[kkLF] #pre daily obs period

######## model parameters
randomstatemodel = 261
train_size= .8
############################################ Import Optimized Parameters ###################################

# paramroots_titles = ["Group I","Group II","Group III","Group IV"]
# paramroots = ["EOFMean","EOFStd","EOFMeanStd","EOFMeanStdEq"]
paramroots_titles = ["Group I","Group II","Group III"]
paramroots = ["EOFMean","EOFStd","EOFMeanStdHF"]

modelname = "LR"
pLR = read_params(fpathparams,modelname,paramroots)

modelname = "Ridge"
pRidge = read_params(fpathparams,modelname,paramroots)

modelname = "Lasso"
pLasso = read_params(fpathparams,modelname,paramroots)

modelname = "DT"
pDT = read_params(fpathparams,modelname,paramroots)

modelname = "ET"
pET = read_params(fpathparams,modelname,paramroots)

modelname = "GBR"
pGBR = read_params(fpathparams,modelname,paramroots)

modelname = "RF"
pRF = read_params(fpathparams,modelname,paramroots)

modelname = "SVR"
pSVR = read_params(fpathparams,modelname,paramroots)
#
# modelname = "NN1"
# pNN1 = read_params(fpathparams,modelname,paramroots)
#
# modelname = "NN2"
# pNN2 = read_params(fpathparams,modelname,paramroots)
#
# modelname = "NN3"
# pNN3 = read_params(fpathparams,modelname,paramroots)
#
# modelname = "NN4"
# pNN4 = read_params(fpathparams,modelname,paramroots)

# ############################################ select features ########################################
output_cols = ['bwidth']
modelnames = ['LR','Ridge','Lasso','DT','RF','ET','GBR','SVR']

#####################################################################################################
#####################################################################################################
############################################### Train Models ########################################
#####################################################################################################
#####################################################################################################

"""
These models are trained on data (80% randomly selected) before daily obs period
"""
########################################### 9030 Mean Std ################################################
modelstruct = {}
for ii,jm in enumerate(paramroots):
    paramroot = jm
    features_cols = pRidge[paramroot]["Features"]
    print(features_cols)
    ######## train a number of models
    lr = train_models_optimized_params(data,output_cols,features_cols,"LR",pLR[paramroot],train_size)
    ridge = train_models_optimized_params(data,output_cols,features_cols,"Ridge",pRidge[paramroot],train_size)
    lasso = train_models_optimized_params(data,output_cols,features_cols,"Lasso",pLasso[paramroot],train_size)
    svr = train_models_optimized_params(data,output_cols,features_cols,"SVR",pSVR[paramroot],train_size)
    dt = train_models_optimized_params(data,output_cols,features_cols,"DT",pDT[paramroot],train_size,randomstatemodel=randomstatemodel)
    rf = train_models_optimized_params(data,output_cols,features_cols,"RF",pRF[paramroot],train_size,randomstatemodel=randomstatemodel)
    et = train_models_optimized_params(data,output_cols,features_cols,"ET",pET[paramroot],train_size,randomstatemodel=randomstatemodel)
    gbr = train_models_optimized_params(data,output_cols,features_cols,"GBR",pGBR[paramroot],train_size,randomstatemodel=randomstatemodel)
    # NN1 = train_models_optimized_params(XTrain_data,YTrain_data,"NN1",params=pNN1[paramroot],randomstatemodel=randomstatemodel)
    # NN2 = train_models_optimized_params(XTrain_data,YTrain_data,"NN2",params=pNN2[paramroot],randomstatemodel=randomstatemodel)
    # NN3 = train_models_optimized_params(XTrain_data,YTrain_data,"NN3",params=pNN3[paramroot],randomstatemodel=randomstatemodel)
    # NN4 = train_models_optimized_params(XTrain_data,YTrain_data,"NN4",params=pNN4[paramroot],randomstatemodel=randomstatemodel)

    modelstruct[paramroot] = [lr,ridge,lasso,dt,rf,et,gbr,svr]

#sanity check for DT
# dttest = modelstruct["9030MeanStd"][3]
# YVal = dttest.predict(XVal_data)
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
############################################ Validation dataset #####################################
################################### Run Equilibrium ########################################

sm = {}
sm["Mopnum"] = mopnum
sm["xmhw_anom"] = dataV["bwidth"].values
sm["t64"] = tls.num2dt64(dataV["tnum"].values)
sm["tnum"] = dataV["tnum"].values

####################### Equilibrium dataset for correct initialization ###########################

tstart = np.datetime64('2022-01-20 00:00:00') #YMD
tend =   np.datetime64('2022-12-06 00:00:00') #YMD
smeq = read_npz_sm(smpath,"sm_MOP%s.npz"%mopnum,t1=tstart,t2=tend)

fparams_eq = "/Users/miks/Desktop/MOP/TorreyPines/Yates/paper/optimize/params/"
peq_name = "eq_params_MOP%s_HF.pkl"%mopnum

peq = read_pickle(fparams_eq,peq_name)
b=peq["b"]
a=peq["a"]
Cminusog=peq["Cminus"]
Cplusog=peq["Cplus"]

inS,waves = get_waves_SS_yates(mopnum,smeq,np.datetime64('2015-01-01T00:00:00'),tend,wpath=wpath)
ya2015 = yates_model(mopnum,smeq,waves,a,b,Cminusog,Cplusog,inS,dt=2)
Sogi2015 = np.interp(sm["tnum"],dates.date2num(ya2015["t"]),ya2015["Sog"])

inS,waves = get_waves_SS_yates(mopnum,smeq,tstart,tend,wpath=wpath)
ya = yates_model(mopnum,smeq,waves,a,b,Cminusog,Cplusog,inS,dt=2)
Sogi = np.interp(sm["tnum"],dates.date2num(ya["t"]),ya["Sog"])

b=0.07
a=-0.0045
Cminusog=-1.38
Cplusog=-1.16

yaOG = yates_model(mopnum,smeq,waves,a,b,Cminusog,Cplusog,inS,dt=2)
SogOGi = np.interp(sm["tnum"],dates.date2num(ya["t"]),yaOG["Sog"])

##################################### Make structured dictionary of model output ################################################
plot_ts(dataV["tnum"]-dataV["tnum"].iloc[0],dataV[features_cols],saveas=True,figpath=figpath,svname="pl_timeseries_LTMOP581_test.png")
modeloutput = {}
for jr in paramroots:
    modeloutput[jr] = {}
    for modelname in modelnames:
        features_cols = pRidge[jr]["Features"]
        if modelname == "LR":
            randomstate = pLR[jr]["RandomState"]
        elif modelname == "Ridge":
            randomstate = pRidge[jr]["RandomState"]
        elif modelname == "Lasso":
            randomstate = pLasso[jr]["RandomState"]
        elif modelname == "SVR":
            randomstate = pSVR[jr]["RandomState"]
        elif modelname == "DT":
            randomstate = pDT[jr]["RandomState"]
        elif modelname == "RF":
            randomstate = pRF[jr]["RandomState"]
        elif modelname == "ET":
            randomstate = pET[jr]["RandomState"]
        elif modelname == "GBR":
            randomstate = pGBR[jr]["RandomState"]

        XTrain_data, XVal_data, YTrain_data, YVal_data, mu, sig = return_XY_normalized(data,output_cols,features_cols,randomstate,train_size)
        dataT = dataV[features_cols]
        XTest_data2 = (dataT - mu)/sig
        YTestdata = np.squeeze(dataV["bwidth"])
        modeloutput[jr][modelname] = {}
        mm = find_modind(modelnames,modelname)
        pY = predict_y(modelstruct[jr],mm,modelnames[mm],XTest_data2,YTestdata,[-13,13],[-13,13],saveas=False,figpath=figpath,svname="sc_testdata_LTMOP581_dailytest_et_allvars")
        modeloutput[jr][modelname]["pY"] = pY
        modeloutput[jr][modelname]["skill"]= model_skill_statistics(sm["xmhw_anom"],pY)




##################################### Make Timeseries Plot ################################################
ntrials = len(paramroots)

if ntrials == 1:
    jf = paramroots[0]
    fig,ax=plt.subplots(figsize=(10,4),ncols=1,sharex=True)

    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["et"]["pY"],color="tab:blue",label="ET",markersize=3)
    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["svr"]["pY"],"*-",label="SVR",color="cyan",markersize=3)
    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["lr"]["pY"],"*-",label="LR",color="orange",markersize=3)
    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["lasso"]["pY"],"D-",label="Lasso",color="green",markersize=3)
    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["dt"]["pY"],"P-",label="DT",color="gold",markersize=3)
    # ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["gbr"]["pY"],"v-",label="GBR",color="red",markersize=3)
    ax.plot(sm["tnum"]-sm["tnum"][0],sm["xmhw_anom"],".-k",label="Observations",markersize=3)
    ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,"--",label="Equilibrium Model",color="tab:orange",markersize=3)
    # ax.set_title(paramroots_titles[ii],fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("$X_{MHW}$ Anomaly [m]", fontsize=16)
    ax.set_ylim([-15,20])
    ax.set_xlabel("Days Since February 08, 2022",fontsize=20)
    ax.legend(loc="upper right",fontsize=16)
    ax.set_xlim([-1,160])
    ax.hlines(0,-1,160,color="k",linewidth=1)
    ax.set_title("Torrey Pines (MOP 581)",fontsize=16)
    ax.legend(loc="upper right",fontsize=16)
else:
    fig,axs=plt.subplots(figsize=(10,12),ncols=1,nrows=ntrials,sharex=True)

    for ii,jf in enumerate(paramroots):
        ax=axs[ii]
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["SVR"]["pY"],"*-",label="SVR",color="cyan",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["LR"]["pY"],"*-",label="LR",color="orange",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["Lasso"]["pY"],"D-",label="Lasso",color="green",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["DT"]["pY"],"P-",label="DT",color="gold",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["GBR"]["pY"],"v-",label="GBR",color="red",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["ET"]["pY"],"s-",color="tab:pink",label="ET",markersize=3)
        ax.plot(sm["tnum"]-sm["tnum"][0],sm["xmhw_anom"],".-k",label="Observations",markersize=5,zorder=1000)
        # ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,"--",label="Equilibrium Model",color="tab:orange",markersize=3)
        ax.set_title(paramroots_titles[ii],fontsize=16)
        ax.tick_params(labelsize=16)
        ax.set_ylabel("$X_{MHW}$ Anomaly [m]", fontsize=16)
        ax.set_ylim([-15,20])
        if ii == ntrials-1:
            ax.set_xlabel("Days Since February 08, 2022",fontsize=20)
        if ii == 0:
            # ax.legend(loc="upper right",fontsize=14,ncol=2)
            fig.suptitle("Black's North (MOP 551)",fontsize=20,y=.94)
        ax.set_xlim([-1,300])
        ax.hlines(0,-1,300,color="k",linewidth=1)

fig.savefig(figpath + "pl_ts_HF_change_eof_revision.png",dpi=300,bbox_inches="tight",pad_inches=.5)

###################################### make skill tables ###########################################
for ii,jm in enumerate(paramroots):
    paramroot = jm
    df = table_skill_full(modeloutput[paramroot],modelnames,paramroots_titles[ii],saveas=True,figpath=figpath,svname="tb_skill_%s_HF.png"%paramroot)

eqskill = model_skill_statistics(sm["xmhw_anom"],Sogi)
table_skill_full_eq(eqskill,"Eq","Equilibrium",saveas=True,figpath=figpath,svname="tb_skill_eq.png")
# #
# paramroot = "9030Std"
# df = table_skill_full(modeloutput[paramroot],modelnames,"Only Standard Deviation",saveas=True,figpath=figpath,svname="tb_skill_%s.png"%paramroot)
#
# paramroot = "9030Mean"
# df = table_skill_full(modeloutput[paramroot],modelnames,"Only Mean",saveas=True,figpath=figpath,svname="tb_skill_%s.png"%paramroot)
#
# paramroot = "9030Mean"
# df = table_skill_full(modeloutput[paramroot],modelnames,"Only Mean",saveas=True,figpath=figpath,svname="tb_skill_%s.png"%paramroot)

######################################### EXTRAAA ####################################################
jf = paramroots[2]

fig,axs=plt.subplots(nrows = 1,ncols=5,figsize=(15,4))
ax=axs[0]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
ax.set_ylabel("Predicted [m]",fontsize=16)
ax.plot(sm["xmhw_anom"],modeloutput[jf]["LR"]["pY"],".",color="orange",markersize=7)
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Linear",fontsize=16)
ax=axs[1]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
# ax.set_ylabel("Predicted [m]",fontsize=16)
ax.yaxis.set_ticklabels([])
ax.plot(sm["xmhw_anom"],modeloutput[jf]["SVR"]["pY"],".",color="cyan",markersize=7)
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Support Vector",fontsize=16)

ax=axs[2]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
# ax.set_ylabel("Predicted [m]",fontsize=16)
ax.plot(sm["xmhw_anom"],modeloutput[jf]["DT"]["pY"],".",color="gold",markersize=7,label="DT")
# ax.plot(sm["xmhw_anom"],pYrf,".",markersize=7,color="orchid",label="RF")
# ax.plot(sm["xmhw_anom"],pYgbr,".",markersize=7,color="limegreen",label="GBR")
# ax.plot(sm["xmhw_anom"],pYet,".",color="magenta",markersize=7,label="ET")
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.yaxis.set_ticklabels([])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Decision Tree",fontsize=16)
# ax.legend(loc="lower right")
ax=axs[3]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
# ax.set_ylabel("Predicted [m]",fontsize=16)
# ax.plot(sm["xmhw_anom"],pYdt,".",color="slateblue",markersize=7,label="DT")
# ax.plot(sm["xmhw_anom"],pYrf,".",markersize=7,color="orchid",label="RF")
ax.plot(sm["xmhw_anom"],modeloutput[jf]["GBR"]["pY"],".",markersize=7,color="red",label="GBR")
# ax.plot(sm["xmhw_anom"],pYet,".",color="magenta",markersize=7,label="ET")
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.yaxis.set_ticklabels([])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Gradient Boosting",fontsize=16)
# ax.legend(loc="lower right")
ax=axs[4]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
# ax.set_ylabel("Predicted [m]",fontsize=16)
# ax.plot(sm["xmhw_anom"],pYdt,".",color="slateblue",markersize=7,label="DT")
# ax.plot(sm["xmhw_anom"],pYrf,".",markersize=7,color="orchid",label="RF")
ax.plot(sm["xmhw_anom"],modeloutput[jf]["ET"]["pY"],".",markersize=7,color="tab:blue",label="GBR")
# ax.plot(sm["xmhw_anom"],pYet,".",color="magenta",markersize=7,label="ET")
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.yaxis.set_ticklabels([])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Extra Trees",fontsize=16)
# ax.legend(loc="lower right")
fig.suptitle("Black's North (MOP 551)",fontsize=20,y=.94)
fig.tight_layout()
fig.savefig(figpath + "pl_scatter_methods_HF.png",dpi=300)


######################################## One Model and EQ plot #######################################################
jf = paramroots[2]
modname = "ET"

rmseeq,req = RMSE_r(sm["xmhw_anom"],Sogi)
rmseml,rml = RMSE_r(sm["xmhw_anom"],modeloutput[jf][modname]["pY"])
rmseeq2,req2 = RMSE_r(sm["xmhw_anom"],SogOGi)
mlskill= model_skill_statistics(sm["xmhw_anom"],modeloutput[jf][modname]["pY"])

rmseeq,req = RMSE_r(sm["xmhw_anom"],Sogi)
rmseml,rml = RMSE_r(sm["xmhw_anom"],modeloutput[jf][modname]["pY"])
rmseeq2,req2 = RMSE_r(sm["xmhw_anom"],SogOGi)
rmseeq2_2015,req2_2015 = RMSE_r(sm["xmhw_anom"],Sogi2015)

mlskill= model_skill_statistics(sm["xmhw_anom"],modeloutput[jf][modname]["pY"])
eqskill= model_skill_statistics(sm["xmhw_anom"],Sogi)

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(sm["tnum"]-sm["tnum"][0],sm["xmhw_anom"],".-k",label="Observations")
ax.plot(sm["tnum"]-sm["tnum"][0],SogOGi,".--",label="Y09 ($r^2$= %.03f, RMSE = %.03f)"%(req2**2,rmseeq2),color="tab:orange")
ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,">--",label="Eq (Optimized) ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange",markersize=4)
ax.plot(sm["tnum"]-sm["tnum"][0],Sogi2015,"s--",label="Eq (Optimized, 2015) ($r^2$= %.03f, RMSE = %.03f)"%(req2_2015**2,rmseeq2_2015),color="tab:orange",markersize=4)

# ax.plot(dates.date2num(ya["t"])-sm["tnum"][0],ya["Sog"],".-",label="Equilibrium Model ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange")
ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf][modname]["pY"],".-",label="ML Model (ET) ($r^2$= %.03f, RMSE = %.03f)"%(rml**2,rmseml),color="tab:blue")
ax.tick_params(labelsize=16)
ax.set_ylabel("$X_{MHW}$ Anomaly [m]",fontsize=16)
ax.set_xlabel("Days Since February 08, 2022",fontsize=20)
ax.legend(loc="upper left",fontsize=13)
ax.set_title("Black's North (MOP 551)",fontsize=16)
ax.set_xlim([-1,300])
ax.hlines(0,-1,300,color="k",linewidth=1)
ax.set_ylim([-15,20])

fig.savefig(figpath + "pl_MLETVEQ_BN_stats"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)
##################################################################################################################
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(sm["tnum"]-sm["tnum"][0],sm["xmhw_anom"],".-k",label="Observations")
ax.plot(sm["tnum"]-sm["tnum"][0],SogOGi,"--",label="Eq (Y09)",color="tab:orange")
ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,"--",label="Eq (Optimized)",color="mediumslateblue",markersize=4)
ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf][modname]["pY"],"-",label="ET",color="tab:pink")

ax.tick_params(labelsize=16)
ax.set_ylabel("$X_{MHW}$ Anomaly [m]",fontsize=16)
ax.set_xlabel("Days Since February 08, 2022",fontsize=16)
# ax.legend(loc="upper left",fontsize=14)
ax.set_title("Black's North (MOP 551)",fontsize=16)
ax.set_xlim([-1,300])
ax.hlines(0,-1,300,color="k",linewidth=1)
ax.set_ylim([-15,20])

fig.savefig(figpath + "pl_MLETVEQ_BN_nostats"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)
####################################################################################################################
f = paramroots[0]
mm = find_modind(modelnames,"et") #lasso
# pY,lr = predict_y(modelstruct[jf],mm,modelnames[mm],XTest_data2,YTestdata,[-13,13],[-13,13],saveas=False,figpath=figpath,svname="sc_testdata_LTMOP581_dailytest_et_allvars")

# jf = paramroots[4]
rmseeq,req = RMSE_r(sm["xmhw_anom"],Sogi)
rmseml,rml = RMSE_r(sm["xmhw_anom"],modeloutput[jf]["et"]["pY"])
eqskill= model_skill_statistics(sm["xmhw_anom"],Sogi)
eqskill= model_skill_statistics(sm["xmhw_anom"],modeloutput[jf]["et"]["pY"])

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(sm["tnum"]-sm["tnum"][0],sm["xmhw_anom"],".-k",label="Observations")
# ax.plot(dates.date2num(waves["t"])-sm["tnum"][0],waves["E"]*30,".-k",label="Observations")

ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,"--k",label="Equilibrium Model ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange")
ax.plot(dates.date2num(ya["t"])-sm["tnum"][0],ya["Sog"],".-",label="Equilibrium Model ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange")
ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["et"]["pY"],"-.",label="ML Model (ET) ($r^2$= %.03f, RMSE = %.03f)"%(rml**2,rmseml),color="tab:blue")
# ax.plot(sm["tnum"]-sm["tnum"][0],Sogi,"--",label="Equilibrium Model",color="tab:orange")
# ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["et"]["pY"],"-.",label="ML Model (ET)",color="tab:blue")
jf = paramroots[2]
# ax.plot(sm["tnum"]-sm["tnum"][0],modeloutput[jf]["svr"]["pY"],"-.",label="ML Model (ET)",color="tab:blue")

ax.tick_params(labelsize=16)
ax.set_ylabel("$X_{MHW}$ Anomaly [m]",fontsize=16)
ax.set_xlabel("Days Since October 11, 2021",fontsize=16)
ax.legend(loc="upper right",fontsize=14)
ax.set_title("MOP 581",fontsize=16)
# ax.set_xlim([-1,160])
ax.hlines(0,-1,160,color="k",linewidth=1)
fig.savefig(figpath + "pl_MLETVEQ_TP"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)
