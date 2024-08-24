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
sys.path.append('/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/tools/Yates')
from yates2008_functions import *
sys.path.append('/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/tools/general')
from tools import *

plt.ion()
# import the data into the workspace


mopnum = 581
froot = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/TP/"
fpathparams = froot + "optimization/optimized_parameters/Nino/"

figpath = froot + "run_model/figures/MOP%s/"%mopnum #figure directory
fpath = froot + "features/CSV_Nino-HF/" #csv directory
wpath = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/waves/"
smpath = "/Users/miks/Desktop/MOP/TorreyPines/ML_revisions/BW_obs/data_processed/"

########################################### Import Dataset ###########################################
data = pd.read_csv(fpath + "MOP%s_MLfiles_ver_eof.csv" %mopnum)

############################################# modify data ############################################
# high frequency obs
kkNino = (data["Inino"] == True)
# low frequency obs
kkNNino = (data["Inino"] == False)

dataV = data[kkNino].copy() #daily obs period, data for testing
data = data[kkNNino] #pre daily obs period

######## model parameters
randomstatemodel = 261
train_size= .8
############################################ Import Optimized Parameters ###################################
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


################################### Model Initialized with observation ########################################
###################################### iterates forward with modeled beach width ##############################
# test daily data


# wpath = "/Users/miks/Desktop/MOP/TorreyPines/MOP_578_585/waves/"
#
# sm = {}
# sm["Mopnum"] = mopnum
# sm["xmhw_anom"] = dataV["bwidth"].values
# sm["t64"] = tls.num2dt64(dataV["tnum"].values)
# sm["tnum"] = dataV["tnum"].values
# t1 = dates.date2num(np.datetime64('2021-10-10')) #YMD
# t2 =   dates.date2num(np.datetime64('2022-02-08')) #YMD
#
# tstart = '20150101' #YMD
# tend =   '20190101' #YMD
#
#
# fparams_eq = "/Users/miks/Desktop/MOP/TorreyPines/Yates/paper/optimize/params/"
# peq_name = "eq_params_MOP%s_Nino.pkl"%mopnum
# # pe1 = read_params(fparams_eq,"eq",peq_root)
# peq = read_pickle(fparams_eq,peq_name)
# b=peq["b"]
# a=peq["a"]
# Cminusog=peq["Cminus"]
# Cplusog=peq["Cplus"]
# ya581all,yar581all = run_yates_og(mopnum,sm,a,b,Cplusog,Cminusog,wpath=wpath,tstart=tstart,tend=tend,dt=2,saveas=False,figpath=figpath,rsave="MOP582og",plotting=False)
# Sogi = np.interp(sm["tnum"],dates.date2num(ya581all["t"]),ya581all["Sog"])
#



####################### Validation dataset ###########################
sm = {}
sm["Mopnum"] = mopnum
sm["xmhw_anom"] = dataV["bwidth"].values
sm["t64"] = tls.num2dt64(dataV["tnum"].values)
sm["tnum"] = dataV["tnum"].values

####################### Equilibrium dataset for correct initialization ###########################
tstart = np.datetime64('2014-11-30T23:00:00') #YMD
tend =   np.datetime64('2019-01-01T00:00:00') #YMD
smeq = read_npz_sm(smpath,"sm_MOP%s.npz"%mopnum,t1=tstart,t2=tend)


fparams_eq = "/Users/miks/Desktop/MOP/TorreyPines/Yates/paper/optimize/params/"
peq_name = "eq_params_MOP%s_Nino.pkl"%mopnum

peq = read_pickle(fparams_eq,peq_name)
b=peq["b"]
a=peq["a"]
Cminusog=peq["Cminus"]
Cplusog=peq["Cplus"]

inS,waves = get_waves_SS_yates(mopnum,smeq,tstart,tend,wpath=wpath)
ya = yates_model(mopnum,smeq,waves,a,b,Cminusog,Cplusog,inS,dt=2)
Sogi = np.interp(sm["tnum"],dates.date2num(ya["t"]),ya["Sog"])

b=0.07
a=-0.0045
Cminusog=-1.38
Cplusog=-1.16

yaOG = yates_model(mopnum,smeq,waves,a,b,Cminusog,Cplusog,inS,dt=2)
SogOGi = np.interp(sm["tnum"],dates.date2num(ya["t"]),yaOG["Sog"])

###################################################
# model_test = {}
# paramroot = "9030Std"
# features_cols = pRidge[paramroot]["Features"]
# dataT = dataV[features_cols]
# XTest_data2 = (dataT - mu)/sig
# YTestdata = np.squeeze(dataV["bwidth"])
#
# mm = find_modind(modelnames,"et")
# pYet = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
# model_test["et"] = pYet
# mm = find_modind(modelnames,"lasso")
# pYlasso = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"ridge")
# pYridge = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"svr")
# pYsvr = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"lr")
# pYlr = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"dt")
# pYdt = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"rf")
# pYrf = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# mm = find_modind(modelnames,"gbr")
# pYgbr = run_initialize(modelstruct[paramroot],mm,modelnames,XTest_data2,YTestdata,dataV["tnum"],mu,sig,plotsc=True,saveas = False,figpath=figpath,svnamesc="sc_et_predict",svnamepl="ls_et_predict")
#
# eqskill= model_skill_statistics(sm["xmhw_anom"],pYgbr)

##################################### Make structured dictionary of model output ################################################


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

    ax.plot(sm["t64"],modeloutput[jf]["ET"]["pY"],"s-",color="tab:blue",label="ET",markersize=3)
    # ax.plot(sm["t64"],modeloutput[jf]["SVR"]["pY"],"*-",label="SVR",color="cyan",markersize=3)
    # ax.plot(sm["t64"],modeloutput[jf]["LR"]["pY"],"*-",label="LR",color="orange",markersize=3)
    # ax.plot(sm["t64"],modeloutput[jf]["Lasso"]["pY"],"D-",label="Lasso",color="green",markersize=3)
    # ax.plot(sm["t64"],modeloutput[jf]["DT"]["pY"],"P-",label="DT",color="gold",markersize=3)
    # ax.plot(sm["t64"],modeloutput[jf]["GBR"]["pY"],"v-",label="GBR",color="red",markersize=3)
    ax.plot(sm["t64"],sm["xmhw_anom"],".-k",label="Observations",markersize=3)
    ax.plot(sm["t64"],Sogi,"--",label="Equilibrium Model",color="tab:orange",markersize=3)
    # ax.set_title(paramroots_titles[ii],fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("$X_{MHW}$ Anomaly [m]", fontsize=16)
    ax.legend(loc="lower right",fontsize=14)
    ax.set_xlim([sm["t64"][0],sm["t64"][-1]])
    ax.hlines(0,sm["t64"][0],sm["t64"][-1],color="k",linewidth=1)
    ax.set_ylim([-45,15])
    ax.tick_params("x",rotation=45)
    ax.set_title("Torrey Pines (MOP 581)",fontsize=16)

else:
    fig,axs=plt.subplots(figsize=(10,12),ncols=1,nrows=ntrials,sharex=True)

    for ii,jf in enumerate(paramroots):
        ax=axs[ii]
        ax.plot(sm["t64"],modeloutput[jf]["SVR"]["pY"],"*-",label="SVR",color="cyan",markersize=3)
        ax.plot(sm["t64"],modeloutput[jf]["LR"]["pY"],"*-",label="LR",color="orange",markersize=3)
        ax.plot(sm["t64"],modeloutput[jf]["Lasso"]["pY"],"D-",label="Lasso",color="green",markersize=3)
        ax.plot(sm["t64"],modeloutput[jf]["DT"]["pY"],"P-",label="DT",color="gold",markersize=3)
        ax.plot(sm["t64"],modeloutput[jf]["GBR"]["pY"],"v-",label="GBR",color="red",markersize=3)
        ax.plot(sm["t64"],modeloutput[jf]["ET"]["pY"],"s-",color="tab:pink",label="ET",markersize=3)
        ax.plot(sm["t64"],sm["xmhw_anom"],".-k",label="Observations",markersize=5,zorder=1000)
        ax.set_title(paramroots_titles[ii],fontsize=16)
        ax.tick_params(labelsize=16)
        ax.set_ylabel("$X_{MHW}$ Anomaly [m]", fontsize=16)
        ax.set_xlim([sm["t64"][0],sm["t64"][-1]])
        # if ii == ntrials-1:
            # ax.set_xlabel("Days Since October 11, 2021",fontsize=16)
        if ii == 0:
            ax.legend(loc="lower right",fontsize=11,ncol=2)
        ax.hlines(0,sm["t64"][0],sm["t64"][-1],color="k",linewidth=1)
        ax.set_ylim([-45,15])
        ax.tick_params("x",rotation=45)
        fig.suptitle("Torrey Pines (MOP 581)",fontsize=24,y=.95)


fig.savefig(figpath + "pl_MLETVEQ_NINO_revision_eof"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)


# fig.savefig(figpath + "pl_ts_HFnoeq_change_et_svr_MeanSTD.png",dpi=300,bbox_inches="tight",pad_inches=.5)

###################################### make skill tables ###########################################
for ii,jm in enumerate(paramroots):
    paramroot = jm
    df = table_skill_full(modeloutput[paramroot],modelnames,paramroots_titles[ii],saveas=True,figpath=figpath,svname="tb_skill_%s_nino.png"%paramroot)

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

fig,axs=plt.subplots(nrows = 1,ncols=3,figsize=(10,4))
ax=axs[0]
ax.plot([-10,20],[-10,20],"--k")
ax.set_xlabel("Observed [m]",fontsize=16)
ax.set_ylabel("Predicted [m]",fontsize=16)
ax.plot(sm["xmhw_anom"],pYlr,".",color="orange",markersize=7)
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
ax.plot(sm["xmhw_anom"],pYsvr,".",color="cyan",markersize=7)
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
ax.plot(sm["xmhw_anom"],pYdt,".",color="slateblue",markersize=7,label="DT")
# ax.plot(sm["xmhw_anom"],pYrf,".",markersize=7,color="orchid",label="RF")
ax.plot(sm["xmhw_anom"],pYgbr,".",markersize=7,color="limegreen",label="GBR")
ax.plot(sm["xmhw_anom"],pYet,".",color="magenta",markersize=7,label="ET")
ax.tick_params(labelsize=16)
ax.set_ylim([-10,20])
ax.set_xlim([-10,20])
ax.yaxis.set_ticklabels([])
ax.set_aspect("equal")
ax.grid("on")
ax.set_title("Decision Trees & Ensemble",fontsize=16)
ax.legend(loc="lower right")
fig.savefig(figpath + "pl_scatter_methods_Nino.png",dpi=300)


######################################## One Model and EQ plot #######################################################
######################################## One Model and EQ plot #######################################################
jf = paramroots[2]
modname = "ET"
rmseeq,req = RMSE_r(sm["xmhw_anom"],Sogi)
rmseml,rml = RMSE_r(sm["xmhw_anom"],modeloutput[jf][modname]["pY"])
rmseeq2,req2 = RMSE_r(sm["xmhw_anom"],SogOGi)


fig,ax=plt.subplots(figsize=(10,4))
ax.plot(sm["t64"],sm["xmhw_anom"],".-k",label="Observations")
ax.plot(sm["t64"],SogOGi,".--",label="Y09 ($r^2$= %.03f, RMSE = %.03f)"%(req2**2,rmseeq2),color="tab:orange")
ax.plot(sm["t64"],Sogi,">--",label="Eq (Optimized) ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange",markersize=4)
# ax.plot(dates.date2num(ya["t"])-sm["t64"][0],ya["Sog"],".-",label="Equilibrium Model ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange")
ax.plot(sm["t64"],modeloutput[jf][modname]["pY"],".-",label="ML Model (ET) ($r^2$= %.03f, RMSE = %.03f)"%(rml**2,rmseml),color="tab:blue")


ax.tick_params(labelsize=16)
ax.set_ylabel("$X_{MHW}$ Anomaly [m]",fontsize=16)
ax.legend(loc="lower right",fontsize=12)
ax.set_title("MOP 581",fontsize=16)
# ax.set_xlim([-1,160])
# ax.hlines(0,-1,160,color="k",linewidth=1)
ax.set_ylim([-45,20])
ax.set_xlim([sm["t64"][0],sm["t64"][-1]])
ax.hlines(0,sm["t64"][0],sm["t64"][-1],color="k",linewidth=1)
ax.set_ylim([-45,15])
ax.tick_params("x",rotation=45)

fig.savefig(figpath + "pl_MLETVEQ_TP_statsNINO"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)
##################################################################################################################
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(sm["t64"],sm["xmhw_anom"],".-k",label="Observations")
# ax.plot(dates.date2num(waves["t"])-sm["t64"][0],waves["E"]*30,".-k",label="Observations")
ax.plot(sm["t64"],SogOGi,"--",label="Eq (Y09)",color="tab:orange")
ax.plot(sm["t64"],Sogi,"--",label="Eq (Optimized)",color="tab:red",markersize=4)
# ax.plot(dates.date2num(ya["t"])-sm["t64"][0],ya["Sog"],".-",label="Equilibrium Model ($r^2$= %.03f, RMSE = %.03f)"%(req**2,rmseeq),color="tab:orange")
ax.plot(sm["t64"],modeloutput[jf][modname]["pY"],"-",label="ET",color="tab:pink")
# ax.plot(sm["t64"]-sm["t64"][0],Sogi,"--",label="Equilibrium Model",color="tab:orange")
# ax.plot(sm["t64"]-sm["t64"][0],modeloutput[jf]["et"]["pY"],"-.",label="ML Model (ET)",color="tab:blue")
# ax.plot(sm["t64"]-sm["t64"][0],modeloutput[jf]["svr"]["pY"],"-.",label="ML Model (ET)",color="tab:blue")
ax.legend(loc="lower right",fontsize=14)
ax.tick_params(labelsize=16)
ax.set_ylabel("$X_{MHW}$ Anomaly [m]",fontsize=16)
# ax.legend(loc="lower right",fontsize=14)
ax.set_title("Torrey Pines (MOP 581)",fontsize=16)
ax.set_xlim([-1,160])
ax.hlines(0,-1,160,color="k",linewidth=1)
ax.set_ylim([-45,20])
ax.set_xlim([sm["t64"][0],sm["t64"][-1]])
ax.hlines(0,sm["t64"][0],sm["t64"][-1],color="k",linewidth=1)
ax.set_ylim([-45,15])
ax.tick_params("x",rotation=45)

fig.savefig(figpath + "pl_MLETVEQ_TP_nostatsNINO"  + ".png",dpi=300,bbox_inches="tight",pad_inches=.5)
