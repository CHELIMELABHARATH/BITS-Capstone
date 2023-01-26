# -*- coding: utf-8 -*-

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://www.machinelearningplus.com/time-series/time-series-analysis-python/

import sys, os, gc, traceback

import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os, gc, traceback
from os import chdir
from dateutil.parser import parse
# Systematic: Components of the time series that have consistency or recurrence and can be described and modeled.
# Non-Systematic: Components of the time series that cannot be directly modeled.
import pmdarima as pm
from pmdarima.utils import array
from dfply import *
from sklearn.model_selection import learning_curve

# Level: The average value in the series.
# Trend: The increasing or decreasing value in the series.
# Seasonality: The repeating short-term cycle in the series.
# Noise: The random variation in the series.


import datetime
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dython.nominal import associations
from plotnine import *
import category_encoders as ce

#from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD
from keras.constraints import maxnorm

import numpy as np

import missingno as msno
#from missingpy import MissForest
from impyute.imputation.cs import mice

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, RandomizedSearchCV, KFold

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import signal

from keras.models import Sequential
import keras.backend as K

from keras.layers import Dense, Input, Dropout, LSTM
from keras.optimizers import SGD, Adam
from keras.models import Model 
from keras.models import load_model 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from xgboost.sklearn import XGBRegressor

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingRegressor, BaggingClassifier
from mlens.visualization import corrmat

class ErrorHandler(object):
    def handleErr(self, err):
        tb = sys.exc_info()[-1]
        stk = traceback.extract_tb(tb, 1)
        functionName = stk[0][2]
        # print(" i am in handle err", functionName)
        return functionName + ":" + err


class PreProcessor():
    def __init__(self, parent=None):
        try:
            # self.filename = filename

            #path = os.path.dirname(os.path.abspath(__file__))
            #lFilename = os.path.join(path, filename)
            # Move to the path that holds our CSV files


            
            accidentDir = 'F:\RajivKhemka\Bits Pilani 2\Code\Road Accident - Satyaki\Accident'
            vehicleDir = 'F:\RajivKhemka\Bits Pilani 2\Code\Road Accident - Satyaki\vehicle'
            
            self.accident = self.produceSingleCSV(accidentDir, 'accident.csv')

            self.accident['AccidentCount'] = 1 # used to count number of accident during aggregation
            #self.vehicle = self.produceSingleCSV(vehicleDir, 'vehicle.csv')
            self.getUniqueValues(self.accident)
            
            self.timeSeriesCols = ['AccidentCount', 'FATALS']
            self.datetimeCols = ['DAY', 'MONTH', 'YEAR', 'DAY_WEEK', 'HOUR', 'MINUTE', 'NOT_HOUR', 
                                 'NOT_MIN', 'ARR_HOUR', 'ARR_MIN', 'HOSP_HR', 'HOSP_MN']
            self.numericCols = ['VE_TOTAL', 'VE_FORMS', 'PVH_INVL', 'PEDS', 'PERNOTMVIT', 'PERMVIT',
                                'PEESONS', 'FATALS', 'DRUNK_DR', 'AccidentCount'] # VE_TOTAL and VE_FORMS may be correlated
            
            self.CategoricalCols = ['STATE', 'NHS', 'RUR_URB', 'FUNC_SYS', 'RD_OWNER', 'ROUTE',
                                    'TWAY_ID2', 'SP_JUR', 'HARM_EV', 'MAN_COLL', 'RELJCT1', 'RELJCT2',
                                    'TYP_INT', 'WRK_ZONE', 'REL_ROAD', 'LGT_COND', 'WEATHER2', 'SCH_BUS', 
                                    'RAIL', 'CF1']
            
            self.aggregationCols = {'STATE':'sum', 'NHS':'sum', 'RUR_URB':'sum', 
                                    'FUNC_SYS':'sum', 'RD_OWNER':'sum', 'ROUTE':'sum',
                                    'TWAY_ID2':'sum', 'SP_JUR':'sum', 'HARM_EV':'sum', 
                                    'MAN_COLL':'sum', 'RELJCT1':'sum', 'RELJCT2':'sum',
                                    'TYP_INT':'sum', 'WRK_ZONE':'sum', 'REL_ROAD':'sum', 
                                    'LGT_COND':'sum', 'WEATHER2':'sum', 'SCH_BUS':'sum', 
                                    'RAIL':'sum', 'CF1':'sum', 'VE_TOTAL':'sum', 
                                    'VE_FORMS':'sum', 'PVH_INVL':'sum', 'PEDS':'sum', 
                                    'PERNOTMVIT':'sum', 'PERMVIT':'sum', 'PEESONS':'sum', 
                                    'FATALS':'sum', 'DRUNK_DR':'sum', 'AccidentCount':'sum'}
            
            # find difference in hours between (NOT_HOUR', 'NOT_MIN') and ('ARR_HOUR', 'ARR_MIN')
            # also find difference in hours between ('ARR_HOUR', 'ARR_MIN') and ('HOSP_HR', 'HOSP_MN')
            # we can drop 'NOT_HOUR', 'NOT_MIN', 'ARR_HOUR', 'ARR_MIN', 'HOSP_HR', 'HOSP_MN'
            
            self.colToDrop = ['ST_CASE', 'COUNTY', 'CITY', 'TWAY_ID', 'MILEPT', 'WEATHER1', 
                              'WEATHER', 'CF2', 'CF3', 'LATITUDE', 'LONGITUD']
            self.accident = self.dropColumn(self.accident, self.colToDrop)
            
            self.accident['AccidentDate']=pd.to_datetime(self.accident[["YEAR", "MONTH", "DAY"]])
            #self.accident['AccidentDate'] = [pd.to_datetime(str(a)+str(b)+str(c), format='%m%d%Y') 
             #for a,b,c in zip(self.accident.MONTH, self.accident.DAY, self.accident.YEAR)]
            
            #self.accident = (self.accident >> arrange(X.AccidentDate))
            #self.accident['AccidentDateTime'] = [pd.to_datetime(str(a)+str(b)+str(c)+str(d)+str(e), format='%m%d%Y%H%M') 
             #for a,b,c,d,e in zip(self.accident.MONTH, self.accident.DAY, self.accident.YEAR, self.accident.HOUR, self.accident.MINUTE)]
                        
            #self.vehicle = pd.read_csv('vehicle.csv', na_values = '-')
            
            self.accident = self.convertColsCategorical(self.accident, self.CategoricalCols)
            self.accidentDecode, self.importantCatCols = self.accidentDecodeCatCols(self.accident)
            #self.accidentDecode = self.convertColsCategorical(self.accidentDecode, importantCatCols)
            
            
            #self.timeSeriesCols = self.geNumericCols(self.data)
            

            print("Data Counts: \n\n{}".format(self.accident.count()))
            print("\nBasic Stat of data: \n{}".format(self.accident.describe()))
            print("\nBasic Info of data: \n{}".format(self.accident.info()))
            print("\nUnique value in each column \n{}".format(self.accident.nunique(axis=0)))

            print(pd.DataFrame([[self.accident.shape],
                                      [self.accident.isnull().sum()],
                                      [self.accident.duplicated().sum()]],
                                     columns = ['Summary'],
                                     index = ['Shape', 'Missing Value', 'Duplicates']))
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    
    def getUniqueValues(self, data):
        data.select_dtypes(np.number).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
        plt.xlabel('Number of Unique Values'); plt.ylabel('Count')
        plt.title('Count of Unique Values in number Columns')
        
        data.select_dtypes(np.number).nunique().value_counts().sort_index().plot.bar(color = 'blue', 
                                                                             figsize = (8, 6),
                                                                            edgecolor = 'k', linewidth = 2);
        plt.xlabel('Number of Unique Values'); plt.ylabel('Count')
        plt.title('Count of Unique Values in number Columns')
    
    def produceSingleCSV(self, csvFilePath, fileOut):
        try:
            #csvFilePath = 'F:\RajivKhemka\Bits Pilani 2\Road Accident - Satyaki\Accident'
            chdir(csvFilePath)
            os.getcwd()
            # List all CSV files in the working dir
            filePattern = '.csv'
            listOfFiles = [file for file in glob('*{}'.format(filePattern))]
            # print(listOfFiles)
            
            #fileOut = "accident.csv"
            # Consolidate all CSV files into one object
            data = pd.concat([pd.read_csv(file, encoding = 'unicode_escape') for file in listOfFiles])
            # Convert the above object into a csv file and export
            data.to_csv(fileOut, index=False, encoding="utf-8")
            data = pd.read_csv(fileOut, na_values = '-')
            return data
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))


    def accidentDecodeCatCols(self, data):
        try:
            #csvFilePath = 'F:\RajivKhemka\Bits Pilani 2\Road Accident - Satyaki\Accident'
            d1 = data.copy()
            weatherNames = {0: 'No Additional Atmospheric Conditions', 1: 'Clear', 
                       2: 'Rain', 3: 'Sleet, Hail', 4: 'Snow', 5: 'Fog, Smog, Smoke', 
                       6: 'Severe Crosswinds', 7: 'Blowing Sand, Soil, Dirt', 
                       8: 'Other', 10: 'Cloudy', 11: 'Blowing Snow', 
                       12: 'Freezing Rain or Drizzle', 
                       98: 'Not Reported', 99: 'Unknown'}
            """
            weatherNames = {'WEATHER2':[0,1,2,3,4,5,6,7,8,10,11,12,98,99],
                            'WeatherDesc':['No Additional Atmospheric Conditions',
                                           'Clear','Rain','Sleet, Hail','Snow',
                                           'Fog, Smog, Smoke','Severe Crosswinds',
                                           'Blowing Sand, Soil, Dirt','Other', 'Cloudy',
                                           'Blowing Snow','Freezing Rain or Drizzle',
                                           'Not Reported','Unknown']}
            """
            #weatherNamesDF = pd.DataFrame.from_dict(weatherNames)
            stateNames = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 
                      6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 
                      11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 
                      16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 
                      21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 
                      25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 
                      28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 
                      32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 
                      36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 
                      40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 43: 'Puerto Rico', 
                      44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 
                      48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 52: 'Virgin Islands', 
                      53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'}
            
            harmEventNames = {1: 'Rollover Overturn', 2: 'Fire Explosion', 3: 'Immersion', 4: 'Gas Inhalation',
                         5: 'Fell/Jumped from Vehicle', 6: 'Injured in Vehicle', 7: 'Other Non-Collision', 
                         8: 'Pedestrian', 9: 'Pedalcyclist', 10: 'Railway Train', 11: 'Animal', 
                         12: 'Motor Vehicle in Transport', 13: 'Motor Vehicle in Transport', 14: 'Parked Motor Vehicle', 
                         15: 'Non-Motorist on Personal Conveyance ', 16: 'Thrown or Falling Object', 17: '', 18: 'Boulder',
                         19: 'Building', 20: 'Crash Cushion', 21: 'Bridge Pier', 22: 'Bridge Parapet End', 
                         23: 'Bridge Rail', 24: 'Guardrail Face', 25: 'Concrete Traffic Barrier', 26: 'Other Traffic Barrier', 
                         27: 'Traffic Sign Post', 28: 'Overhead Sign Support', 29: 'Light Support', 30: 'Utility Pole', 
                         31: ' Other Pole', 32: 'Culvert', 33: 'Curb', 34: 'Ditch', 35: 'Embankment', 36: 'Concrete', 
                         37: 'Embankment – Material Type Unknown', 38: 'Fence', 39: 'Wall', 40: 'Fire Hydrant', 
                         41: 'Shrubbery', 42: 'Tree', 43: ' Fixed Object', 44: 'Pavement Surface', 45: 'Working Motor Vehicle', 
                         46: 'Traffic Signal Support', 47: ' Run Over by Own Vehicle', 48: 'Snow Bank', 49: 'Ridden Anima',
                         50: 'Bridge Overhead Structure', 51: 'Jackknife', 52: 'Guardrail End', 53: 'Mail Box', 
                         54: 'Motor Vehicle In-Transport Strikes',
                         55: 'Motor Vehicle in Motion', 57: 'Cable Barrier', 58: 'Ground', 59: 'Traffic Sign Support', 
                         72: 'Equipment Loss', 73: 'Object Fell From Motor Vehicle', 74: 'Road Vehicle on Rails', 
                         91: 'Unknown Object Not Fixed', 93: 'Unknown Fixed Object', 98: 'Not Reported',
                         99: 'Unknown'}


            mannerCollisionNames = {0: 'Not Collision with Motor Vehicle in Transport', 1: 'Front-to-Rea', 
                                    2: 'Front-to-Front', 3: 'Front-to-Side, Same Direction', 4: 'Front-to-Side, Opposite Direction', 
                                    5: '– Front-to-Side, Right Angle', 6: '– Front-to-Side Angle-Direction Not Specified', 
                                    7: 'Sideswipe – Same Direction', 8: 'Sideswipe – Opposite Direction', 
                                    9: 'Rear-to-Side', 10: 'Rear-to-Rear', 11: 'Other', 98: 'Not Reported', 
                                    99: 'Unknown'}


            catCols = {'StateDesc':stateNames, 'WeatherDesc':weatherNames, 'HarmEventDesc': harmEventNames,
                       'MannerCollisionDESC':mannerCollisionNames}
            #df = [weatherNamesDF]
            dfCols = ['STATE', 'WEATHER2', 'HARM_EV', 'MAN_COLL']
            #for i in range(len(dfCols)):
             #   d1=d1.merge(df[i], on=dfCols[i], how='left')
            for i, (name, m) in enumerate(catCols.items()):

                #d1['StateName']=d1['STATE'].apply(lambda x: stateNames[x])
                d1[name]=d1[dfCols[i]].apply(lambda x: m[x])
            return d1, dfCols
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def dropColumn(self, data, colsToDrop):
        try:
            d1 = data.copy()
            d1 = data.drop(colsToDrop, axis=1)
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def getAccident(self):
        return self.accident
    
    def getVehicle(self):
        return self.vehicle
    
    def getAggregationCols(self):
        return self.aggregationCols
    
    def getTimeSeriesCols(self):
        return self.timeSeriesCols
    
    def getDateTimeCols(self):
        return self.datetimeCols
    
    def getNumericData(self, data):
        numericCols = self.geNumericCols(data)
        return data[numericCols]
    
    def getFloatData(self, data):
        floatCols = self.getFloatCols(data)
        return data[floatCols]

    def getFloatCols(self, data):
        return data.select_dtypes(include=["float64"]).columns  
    
    def getCategoricalCols(self, data):
        return data.select_dtypes(include=["category"]).columns  

    def geNumericCols(self, data):
        return data.select_dtypes(include=["number"]).columns    
    
    def convertColsCategorical(self, data, cols):
        d1 = data.copy()
        d1[cols] = d1[cols].astype('category')
        return d1
    
    def convertFloatToNumeric(self, data):
        d1 = data.copy()
        floatCols = self.getFloatCols(d1)
        d1[floatCols] = d1[floatCols].astype('int64')
        return d1
    
    def getAggregateData(self, data, groupbyCols, aggrCols):
        return data.groupby(groupbyCols).agg(aggrCols)
    
    def createAggregationDict(self, data, fnNames):
        numericalCols = self.geNumericCols(data).tolist()
        d1 = dict()
        for i in numericalCols:
            d1.update({i:'sum'})
        return d1
    


class visualize:
    def __init__(self, parent=None):
        self.errObj = ErrorHandler()

    def pltSetFullScreen(self):
        backend = str(plt.get_backend())
        mgr = plt.get_current_fig_manager()
        if backend == 'TkAgg':
            if os.name == 'nt':
                mgr.window.state('zoomed')
            else:
                mgr.resize(*mgr.window.maxsize())
        elif backend == 'wxAgg':
            mgr.frame.Maximize(True)
        elif backend == 'Qt4Agg':
            mgr.window.showMaximized()

    def pandaPlot(self, data, numericCols, categoricalCols, response='TARGET_deathRate'):
        try:
            plt.close('all')
            self.pltSetFullScreen()
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            # plt.figure(figsize=(80, 60))
            x = data.plot(x=numericCols[0], y=numericCols[1], style='o')
            # data.plot(x=data[0], y=data[1], style='o')
            plt.title('Graph')
            plt.xlabel(numericCols[0])
            plt.ylabel(numericCols[1])

            plt.show()
            """
            for i in range((len(numericCols)-1)):
                data.plot.hexbin(x=response, y=numericCols[i+1], gridsize=25)
                plt.close()
            """
            scaler = MinMaxScaler()
            d2 = scaler.fit_transform(data[numericCols])
            scaledData = pd.DataFrame(d2, columns=data[numericCols].columns)
            # x = preprocessing.scale(data[numericCols])
            # scaledData = pd.DataFrame(x)
            # scaledData.columns = data[numericCols].columns
            plt.figure()
            scaledData.iloc[:, 0:4].plot.hist(alpha=0.5)

            plt.show()
            x = scaledData[numericCols[0:30]].plot.kde()
            x.figure.savefig('pandaKDE.png')
            scaledData[numericCols[0:2]].plot.area()

            pd.crosstab(data.binnedInc, data.PctWhiteBin).plot(kind='bar')
            plt.title('Frequency of White % Category for Income Group')
            plt.xlabel('Income')
            plt.ylabel('Frequency of White %')
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def snsPlot(self, data, numericCols, categoricalCols, response='TARGET_deathRate'):
        try:


            selectedCols = []
            hueCols = []
            Cols = []
            for p in categoricalCols:
                #plt.close('all')
                plt.figure(figsize=(4, 4))
                # plt.subplot(121)
                x = data[p].value_counts().reset_index()
                ax = sns.barplot(p, "index", data=x, palette="husl")
                #for i, j in enumerate(
                 #       np.around((x[p][:10].values * 100 / (x[p][:10].sum())))):
                  #  ax.text(.7, i, j, weight="bold")
                plt.xlabel("Count")
                plt.ylabel('code of ' + p)
                plt.title("Frequency Distribution")
                #ax.figure.savefig(p + 'count.png')
                plt.show()

            """
            for i in numericCols:
                plt.figure(figsize=(15, 10))
                plt.tight_layout()
                x = sns.distplot(data[i])
                x.figure.savefig(i+'.png')

            sns.set(style="ticks")

            indx1 = numericCols[1:10]
            indx1 = indx1.append(pd.Index([numericCols[0]]))
            data[indx1.values]
            sns.pairplot(data[indx1.values])

            indx1 = numericCols[11:21]
            indx1 = indx1.append(pd.Index([numericCols[0]]))
            data[indx1.values]
            sns.pairplot(data[indx1.values])

            indx1 = numericCols[22:25]
            indx1 = indx1.append(pd.Index([numericCols[0]]))
            data[indx1.values]
            sns.pairplot(data[indx1.values])
            plt.close()
            """

            # results = associations(data, nominal_columns=catcols, return_results=True)

            # sns.pairplot(data[numericCols[0:4]])
            """
            hueCols = ['medIncomeBin', 'popEst2015Bin', 'povertyPercentBin', 'MedianAgeMaleBin', 'MedianAgeFemaleBin', 'PctPrivateCoverage', 'PctPublicCoverage', 'PctWhiteBin', 'PctBlackBin']
            for i in categoricalCols:
                indx1 = numericCols[0:6]
                indx1 = indx1.append(pd.Index([i]))
                x = sns.pairplot(data[indx1.values], hue=i)
                plt.show()
                #x.figure.savefig(i+'.png')
                #plt.close()
            plt.close()
            """
            selectedCols = ['medIncome', 'popEst2015', 'povertyPercent', 'MedianAgeMale', 'MedianAgeFemale',
                            'PctPrivateCoverage', 'PctPublicCoverage', 'PctWhite', 'PctBlack']
            for i in numericCols:
                sns.jointplot(x=response, y=i, data=data)

                # for i in numericCols[1:len(numericCols)]:
            #   sns.paidata[[indx1]]rplot(data[[response, numericCols[i]]])

            ################################
            selectedCols = ['medIncomeBin', 'popEst2015Bin', 'povertyPercentBin', 'MedianAgeMaleBin',
                            'MedianAgeFemaleBin',
                            'PctPrivateCoverage', 'PctPublicCoverage', 'PctWhiteBin', 'PctBlackBin']
            hueCols = ['YEAR', 'MONTH']
            for i in categoricalCols:
                for j in hueCols:
                    plt.figure(figsize=(15, 7))
                    plt.subplot(121)
                    graph = sns.countplot(y=data[i],
                                          palette="Set2",
                                          order=data[i].value_counts().index[:10])
                    plt.title("Distribution of " + i)

                    plt.subplot(122)
                    sns.countplot(y=data[i],
                                  hue=data[j], palette="Set2",
                                  order=data[i].value_counts().index[:10])
                    plt.ylabel("")
                    plt.title("Distribution of " + i + " by " + j)

                    plt.subplots_adjust(wspace=.4)
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def ggplotPlot(self, data, numericCols, categoricalCols, response='TARGET_deathRate'):
        plt.close('all')
        try:
            selectedCols = []
            fillCols = []
            facetCols = []

            myTheme = theme(axis_text_x=element_text(color="grey", size=10, angle=90, hjust=.5),
                            axis_text_y=element_text(color="grey", size=10))
            for i in selectedCols:
                for j in fillCols:
                    for k in facetCols:
                        # plt.figure(figsize=(15, 10))
                        # plt.tight_layout()
                        x = ggplot(data=data, mapping=aes(x=i, fill=j, show_legend=True)) + \
                            scale_color_brewer(type='diverging', palette=4) + \
                            xlab(i) + ggtitle("Frequency Distribution") + \
                            geom_bar(position=position_dodge()) + myTheme + facet_wrap(k)

                        ggsave(plot=x, filename=i + 'bin',
                               path="F:\RajivKhemka\Bits Pilani\Code\Cancer Mortality\ggplot")
                        # plt.close()

            print(ggplot(data=data, mapping=aes(x='povertyPercentBin', fill='medIncomeBin', show_legend=True)) +
                  scale_color_brewer(type='diverging', palette=4) +
                  geom_bar(position=position_dodge()) + myTheme + facet_wrap('MedianAgeBin'))

            print(ggplot(data, aes(x=response, y=numericCols[4], color='povertyPercentBin')) + geom_point() +
                  stat_smooth(colour='blue', span=0.2) +
                  scale_color_brewer(type='diverging', palette=4) +
                  facet_wrap('MedianAgeBin'))

            for i in selectedCols:
                for j in fillCols:
                    # for k in facetCols:
                    # plt.figure(figsize=(15, 10))
                    # plt.tight_layout()
                    print(ggplot(data=data, mapping=aes(x=i, fill=j, show_legend=True)) + \
                          scale_color_brewer(type='diverging', palette=4) + \
                          xlab(i) + ggtitle("Frequency Distribution") + \
                          geom_bar(position=position_dodge()) + myTheme + \
                          facet_grid('PctPrivateCoverageBin ~ PctPublicCoverageBin'))

            print(ggplot(data, aes(x='povertyPercent', y='TARGET_deathRate', color='binnedInc')) + geom_point() +
                  labs(title='Correlation', x='Poverty %', y='Death Rate') + stat_smooth(colour='blue', span=0.2) +
                  scale_color_brewer(type='diverging', palette=4) +
                  facet_wrap('binnedInc'))
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

        
    def plotTimeSeries(self, data, timeSeriesCols, dpi=100):
        try:
            #plt.figure(figsize=(16,5), dpi=dpi)
            fig, axes = plt.subplots(nrows=round(len(timeSeriesCols)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
            x = data.index
            colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
            
            for i in range(len(timeSeriesCols)):  
                m = timeSeriesCols[i]
                y=data[m]
                y.index = x
                c = colors[i % (len(colors))]
                #plt.plot(x, y, color='tab:red')time series plot of 
                ax = y.plot(ax = axes[i // 2, i % 2],rot=25, color=c, title = 'time series plot of ' + m)
                ax.set_ylabel(m)
                #plt.gca().set(title='time series plot of ' + m, xlabel='Date', ylabel=m)
                #rot=25
                
            plt.tight_layout()
            plt.show()
            data[timeSeriesCols].plot(figsize=(20,10), linewidth=5, fontsize=20)
            plt.xlabel('Year', fontsize=20)
            
            """ https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-one/
            import warnings
            import matplotlib.pyplot as plt
            y = df['Orders']
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
            ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
            ax.set_ylabel('Orders')
            ax.legend();
            """
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def plotSeasonality(self, data, timeSeriesCols, dpi=100):
        try:
            #plt.figure(figsize=(16,5), dpi=dpi)
            data['Date'] = data.index
            data['Year'] = pd.DatetimeIndex(data.Date).year
            data['Month'] = pd.DatetimeIndex(data.Date).strftime('%b')
            cols = ['Year', 'Month']
            data[cols] = data[cols].astype('category')

            years = data['Year'].unique()
            
            #colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
            
            fig, axes = plt.subplots(nrows=round(len(timeSeriesCols)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
            for i in range(len(timeSeriesCols)):  

                m = timeSeriesCols[i]
                dataWide = data.pivot("Year", "Month", m)
                sns.lineplot(data=data, x="Month", y=m, hue='Year', sort=False,
                             ax = axes[i // 2, i % 2])
                
            fig, axes = plt.subplots(nrows=round(len(timeSeriesCols)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
            for i in range(len(timeSeriesCols)):   
                m = timeSeriesCols[i]
                sns.boxplot(x='Year', y=m, data=data, ax = axes[i // 2, i % 2])

                
            fig, axes = plt.subplots(nrows=round(len(timeSeriesCols)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
            for i in range(len(timeSeriesCols)): 
                m = timeSeriesCols[i]
                sns.boxplot(x='Month', y=m, data=data, ax = axes[i // 2, i % 2])
                
            plt.tight_layout()
            plt.show()
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def visualizeCorrelation(self, data):
        try:
            d1 = data.copy()
            categoricalCols = d1.select_dtypes(include=["category"]).columns 
            categoricalCols = categoricalCols.tolist()
            for i in categoricalCols:

                d1[i] = d1[i].astype('object')
            # http://shakedzy.xyz/dython/getting_started/examples/
            associations(d1[categoricalCols], theil_u=True, figsize=(15, 15))
            
            corr = data.corr()
            g = sns.heatmap(corr, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,
                            fmt='.2f', cmap='coolwarm')
            sns.despine()
            g.figure.set_size_inches(24, 20)
            plt.show()
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))            

class MissingValue:
    def __init__(self, parent=None):
        self.errObj = ErrorHandler()
        # self.data = data
        ## self.data = self.dropColumn(colName)

    def getMissingCount(self, data):
        try:
            missingCount = data.isnull().sum() * 100 / len(data)
            return missingCount
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def visualizeMissingValue(self, data):
        try:
            missingCount = self.getMissingCount(data)
            # missingCount = data.isnull().sum()/ len(data)
            print('missing value in each column', missingCount)
            plt.figure(figsize=(16, 8))
            plt.xticks(np.arange(len(missingCount)) + 0.5, missingCount.index, rotation='vertical')
            plt.ylabel('fraction of rows with missing data')
            plt.bar(np.arange(len(missingCount)), missingCount)
            plt.show()

            plt.figure(figsize=(15, 20))
            sns.heatmap(pd.DataFrame(data.isnull().sum() / data.shape[0] * 100), annot=True,
                        cmap=sns.color_palette("cool"), linewidth=1, linecolor="white")
            plt.title("Missing Value")
            plt.show()
            # missingColumn = data.columns[data.isna().any()].tolist()
            # print(missingColumn)
            # msno.matrix(data.loc[:,['PctPrivateCoverageAlone']])
            # msno.bar(data.loc[:,['PctPrivateCoverageAlone']], color="blue", log=False, figsize=(30, 18))
            # msno.heatmap(data[missingColumn], figsize=(20, 20))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def visualizeMissingCount(self, data):
        try:
            # set up aesthetic design
            plt.style.use('seaborn')
            sns.set_style('whitegrid')
            
            # create NA plot for train data
            plt.figure(figsize = (15,3)) # positioning for 1st plot
            data.isnull().sum().sort_values(ascending = False).plot.bar(color = 'blue')
            plt.axhline(y=mean(data.isnull().sum()), color='r', linestyle='-')
            plt.title('Missing values average per columns in data', fontsize = 20)
            plt.show()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def identifyColsToDrop(self, data, threshold=30.0):
        try:
            missingCols = self.getMissingCount(data)
            colsToDrop = missingCols[missingCols > threshold].index.values
            # colsToDrop = [col for (col, perc) in missingCols[missingCols > threshold].items()]
            return colsToDrop
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def dropColumn(self, data, threshold):
        try:
            colsToDrop = self.identifyColsToDrop(data, threshold)
            data1 = data.drop(colsToDrop, axis=1)
            return data1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def dropRow(self, data, rowToDrop):
        try:
            d1 = data.dropna(0, subset=rowToDrop).reset_index(drop=True)
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def imputeByMean(self, data, numericCols):
        try:
            imputedMean = SimpleImputer(strategy='mean')  # for median imputation replace 'mean' with 'median'
            cols = data.columns
            nonNumeric = set(cols) - set(numericCols)
            dataNumeric = data.drop(nonNumeric, axis=1)
            imputedMean.fit(dataNumeric)
            imputedData = imputedMean.transform(dataNumeric)

            d1 = pd.DataFrame(imputedData)
            d1.columns = dataNumeric.columns
            finalImputedData = pd.concat([d1, data.loc[:, nonNumeric]], axis=1)
            ## imputedData.columns = data.columns
            return finalImputedData
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def imputeByMice(self, data, numericCols):
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            # numericCols = data.select_dtypes(include=["number"]).columns
            dataNumeric = data[numericCols]
            nonNumericCols = [col for col in data.columns if col not in numericCols]
            d2 = data[nonNumericCols] # d2 is data frame with date time index
            # start the MICE training
            imputedDataMice = mice(dataNumeric.values)
            d1 = pd.DataFrame(imputedDataMice) # index of d1 is running number
            d1.columns = dataNumeric.columns
            if d2.shape[1] != 0:
                d1.index = d2.index # index of d1 needs to be converted to date time from d2
                imputedData = pd.concat([d1, d2], axis=1)
            else:
                imputedData = d1
            ## imputedData.columns = data.columns
            imputedData = imputedData.set_index(dataNumeric.index)
            return imputedData
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def imputeByKNN(self, data, numericCols):
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            # start the KNN training
            dataNumeric = data[numericCols]
            nonNumericCols = [col for col in data.columns if col not in numericCols]
            d2 = data[nonNumericCols]

            imputer = KNNImputer(n_neighbors=2)
            imputedDataKNN = imputer.fit_transform(dataNumeric.values)
            d1 = pd.DataFrame(imputedDataKNN)
            d1.columns = dataNumeric.columns
            imputedData = pd.concat([d1, d2], axis=1)
            ## imputedData.columns = data.columns
            return imputedData
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def imputeByMissForest(self, data, numericCols):
        # https://pypi.org/project/missingpy/
        try:
            # from missingpy import MissForest
            # dataNumeric = data.drop(nonNumeric, axis=1)
            dataNumeric = data[numericCols]
            nonNumericCols = [col for col in data.columns if col not in numericCols]
            d2 = data[nonNumericCols]

            # start the KNN training
            imputer = MissForest()
            imputedDataMissForest = imputer.fit_transform(dataNumeric.values)
            d1 = pd.DataFrame(imputedDataMissForest)
            d1.columns = dataNumeric.columns
            imputedData = pd.concat([d1, d2], axis=1)
            ## imputedData.columns = data.columns
            return imputedData
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

class Outlier:
    def __init__(self, parent=None):
        self.errObj = ErrorHandler()

    def visualizeOutlier(self, scaledData, catCols):
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            sns.boxplot(data=scaledData.iloc[:, 0:15])
            for i in range(len(catCols)):
                sns.catplot(x=catCols[i], y="VE_TOTAL", kind="box", data=scaledData)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def zScoreOutlier(self, data):
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            zScore = np.abs(stats.zscore(data))
            print(np.where(zScore > 3))
            outlierData = data[(zScore < 3).all(axis=1)]

            return outlierData
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def IQROutlier(self, data):
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            quantile1 = data.quantile(0.25)
            quantile3 = data.quantile(0.75)
            IQR = quantile3 - quantile1
            print(IQR)
            outlierData = data[~((data < (quantile1 - 1.5 * IQR)) | (data > (quantile3 + 1.5 * IQR))).any(axis=1)]
            outlierData.shape
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def DBScanOutlier(self, data):  # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        try:
            # dataNumeric = data.drop(nonNumeric, axis=1)
            d1 = data.copy()
            # d1 = pd.get_dummies(d1[numericCols])

            scaler = MinMaxScaler()
            d2 = scaler.fit_transform(d1)
            d2 = pd.DataFrame(d2, columns=d1.columns)

            self.findDBScanEPS(d2, 5)
            # epsfloat, default=0.5
            # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            # This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN
            # parameter to choose appropriately for your data set and distance function.
            # min_samplesint, default=5
            # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            # This includes the point itself.

            db = DBSCAN(eps=0.3, min_samples=5).fit(d2)

            #from sklearn import metrics
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # detectedCluster = DBSCAN(eps=3.0, metric='euclidean', min_samples=10, n_jobs=-1)
            # clusters = detectedCluster.fit_predict(d2)

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)
            # print("Homogeneity: %0.3f" % metrics.homogeneity_score(len(d2), labels))
            # print("Completeness: %0.3f" % metrics.completeness_score(len(d2), labels))
            # print("V-measure: %0.3f" % metrics.v_measure_score(len(d2), labels))
            # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(len(d2), labels))
            # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(len(d2), labels))
            # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(d2, labels))

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            plt.figure(figsize=(8, 8))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = d2[class_member_mask & core_samples_mask]
                plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
                         markersize=14)

                xy = d2[class_member_mask & ~core_samples_mask]
                plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k',
                         markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()

            # cmap = cm.get_cmap('Set1')
            # d2.plot.scatter(x='avgDeathsPerYear', y='avgAnnCount', c=clusters, cmap=cmap, colorbar=False)
            # plt.close()
            outlierRow = []
            for i in range(len(labels)):
                if (labels[i] == -1):
                    # print(list((i, labels[i])))
                    outlierRow.append(i)

            return outlierRow

        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def findDBScanEPS(self, data, n_neighbors):
        try:
            nn = NearestNeighbors(n_neighbors=2)
            nbrs = nn.fit(data)
            distances, indices = nbrs.kneighbors(data)

            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            plt.plot(distances)

        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def KmeansOutlier(self, scaledData):
        try:
            # d1 = pd.get_dummies(scaledData)
            wcss = []
            for i in range(1, 30):
                k1 = KMeans(i)
                k1.fit(scaledData)
                wcss.append(k1.inertia_)
            plt.plot(range(1, 30), wcss)
            plt.xlabel('Number of Clusters')
            plt.ylabel('wcss')
            plt.show()
            kmeansCluster = KMeans(4)
            kmeansCluster.fit(scaledData)
            scaledDataCopy = scaledData.copy()
            scaledDataCopy['ClusterPred'] = kmeansCluster.fit_predict(scaledData)
            scaledDataCopy
            plt.scatter(scaledDataCopy['medIncome'], scaledDataCopy['TARGET_deathRate'],
                        c=scaledDataCopy['ClusterPred'], cmap='rainbow')
            plt.xlabel('Median Income')
            plt.ylabel('Death Rate')
            plt.show()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def vifOutlier(self, scaledData):
        try:
            # from statsmodels.stats.outliers_influence import variance_inflation_factor
            # from statsmodels.tools.tools import add_constant
            x = add_constant(scaledData)
            vif = pd.Series([variance_inflation_factor(x.values, i)
                             for i in range(x.shape[1])],
                            index=x.columns)
            print(vif)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def lofOutlier(self, scaledData, k):
        try:

            var1, var2 = 1, 2
            clf = LocalOutlierFactor(n_neighbors=k, contamination=.1)
            y_pred = clf.fit_predict(scaledData)
            LOF_Scores = clf.negative_outlier_factor_

            plt.title('Local Outlier Factor(LOF), K = {}'.format(k))
            plt.scatter(scaledData.iloc[:, var1], x1.iloc[:, var2], color='k', s=3., label='Data points')
            radius = (LOF_Scores.max() - LOF_Scores) / (LOF_Scores.max() - LOF_Scores.min())
            plt.scatter(scaledData.iloc[:, var1], scaledData.iloc[:, var2], s=1000 * radius, edgecolors='r',
                        facecolors='none',
                        label='Outlier scores')
            plt.axis('tight')
            plt.ylabel('{}'.format(scaledData.columns[var1]))
            plt.xlabel('{}'.format(scaledData.columns[var2]))
            legend = plt.legend(loc='upper left')
            legend.legendHandles[0]._sizes = [10]
            legend.legendHandles[1]._sizes = [20]
            plt.show();

        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def ifOutlier(self, data, contamination, var1, var2, response):  # Isolation Forest
        try:

            #var1, var2 = 1, 2
            clf = IsolationForest(n_estimators=100, max_samples=len(
                data), contamination=0.03, max_features=1.0,)
            y_pred = clf.fit_predict(data)
            #LOF_Scores = clf.negative_outlier_factor_

            # LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
            IF_anomalies = data[y_pred == -1]

            print('Estimated number of noise points: %d' % len(IF_anomalies))

            # plot the line, the samples, and the nearest vectors to the plane
            #xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
            #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            #Z = Z.reshape(xx.shape)

            plt.title("IsolationForest")
            #plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

            b1 = plt.scatter(
                data.loc[:, var1], data.loc[:, var2], c="white", s=20, edgecolor="k")
            #b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="green", s=20, edgecolor="k")
            c = plt.scatter(
                IF_anomalies.loc[:, var1], IF_anomalies.loc[:, var2], c="red", s=20, edgecolor="k")
            plt.axis("tight")
            #plt.xlim((-5, 5))
            #plt.ylim((-5, 5))
            plt.legend([b1, c], ["training observations", "new abnormal observations"], loc="upper left", )
            plt.show()

            level0Outlier = data.loc[IF_anomalies.index, response].value_counts()[0]
            level1Outlier = data.loc[IF_anomalies.index, response].value_counts()[1]

            return IF_anomalies, level0Outlier, level1Outlier

        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
    def compareOutliers(self, data):
        try:
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(x="Model", y="OutlierCount",
                             hue="TargetValue", data=data)
            plt.show()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))


class modeler:
    def __init__(self):
        self.errObj = ErrorHandler()
        
    ### plot for Rolling Statistic for testing Stationarity
    def testStationarity(data, title):
        try:
            #Determing rolling statistics
            rolmean = pd.Series(data).rolling(window=12).mean() 
            rolstd = pd.Series(data).rolling(window=12).std()
            
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(data, label= title)
            ax.plot(rolmean, label='rolling mean');
            ax.plot(rolstd, label='rolling std (x10)');
            ax.legend()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    
    def checkADF(self, data): # Augmented Dickey-Fuller Test
        try:
            print('Augmented Dickey-Fuller Test:')
            result = adfuller(data.dropna(),autolag='AIC') # .dropna() handles differenced data
        
            labels = ['ADF test statistic','p-value','# lags used','# observations']
            out = pd.Series(result[0:4],index=labels)
    
            for key, value in result[4].items():
                out[f'critical value ({key})']=value
                
            print(out.to_string())          # .to_string() removes the line "dtype: float64"
            
            if result[1] <= 0.05:
                print("Data has no unit root and is stationary")
            else:
                print("Data has a unit root and is non-stationary")
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))   
            
    def checkKPSS(self, data): # Kwiatkowski-Phillips-Schmidt-Shin Test
        try:
            print('Kwiatkowski-Phillips-Schmidt-Shin Test:')
            result = kpss(data.values, regression='c') 
        
        
            print('\nKPSS Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            for key, value in result[3].items():
                print('Critial Values:')
                print(f'   {key}, {value}')
        
            labels = ['ADF test statistic','p-value','# lags used','# observations']
            out = pd.Series(result[0:4],index=labels)
    
            for key,val in result[4].items():
                out[f'critical value ({key})']=val
                
            print(out.to_string())          # .to_string() removes the line "dtype: float64"
            
            if result[1] <= 0.05:
                print("Data has no unit root and is stationary")
            else:
                print("Data has a unit root and is non-stationary")
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err)) 
            
    def decomposeTimeSeries(self, data, freq): # data is 1d series
        try:
            # ETS - error, trend, seasonality graph
            # https://github.com/neelabhpant/Deep-Learning-in-Python/blob/master/ClimateChange_Prediction.ipynb
            # https://www.machinelearningplus.com/time-series/time-series-analysis-python/
            plt.rcParams.update({'figure.figsize': (10,10)})
            # Setting extrapolate_trend='freq' takes care of any missing values in the trend and residuals at the beginning of the series.
            resultAdditive = seasonal_decompose(data, model='add', 
                                                period=freq, extrapolate_trend=freq)
            resultMultiplicative = seasonal_decompose(data, model='multi', 
                                                period=freq, extrapolate_trend=freq)
            resultAdditive.plot().suptitle('Additive Decompose', fontsize=22)
            resultMultiplicative.plot().suptitle('Multiplicative Decompose', fontsize=22)
            plt.show()
            resultAdditive.seasonal.loc['2010-01-01':'2012-01-01'].plot()
            plt.show()
            
            # Extract the Components ---- Actual Values = Product of (Seasonal * Trend * Resid)
            
            MultiValue = self.getDecompositionValue(resultMultiplicative)
            
            """ https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-one/
            import statsmodels.api as sm

            # graphs to show seasonal_decompose
            def seasonal_decompose (y):
                decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
                fig = decomposition.plot()
                fig.set_size_inches(14,7)
                plt.show()
            """
            
            return resultMultiplicative, resultAdditive, MultiValue
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))   

    def getDecompositionValue(self, multiplicativeData):
        try:
            # Actual Values = Product of (Seasonal * Trend * Resid)
            d1 = pd.concat([multiplicativeData.seasonal, multiplicativeData.trend, 
                            multiplicativeData.resid, multiplicativeData.observed], axis=1)
            d1.columns = ['seas', 'trend', 'resid', 'actual_values']
            print(d1.head())
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))   
            
    def detrendLSF(self, data, title):
        # Detrending a time series is to remove the trend component from a time series
        # Subtract the line of best fit from the time series. The line of best fit may 
        # be obtained from a linear regression model with the time steps as the predictor
        try:
            detrended = signal.detrend(data)
            plt.plot(detrended)
            plt.title(title, fontsize=16)
            plt.show()
            
            return detrended
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))   

    def detrendTrendComponent(self, data, trend, title):
        # Using statmodels: Subtracting the Trend Component.        
        try:
            detrended = data.values - trend
            plt.plot(detrended)
            plt.title(title, fontsize=16)
            plt.show()
            detrended = pd.DataFrame(detrended)   
            
            """
            y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()


            """
            
            return detrended
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err)) 
            
    def deseasonalize(self, data, timeSeriesCols, freq):
        # 1. Take a moving average with length as the seasonal window. This will smoothen in series in the process.

        # 2. Seasonal difference the series (subtract the value of previous season from the current value)

        # 3. Divide the series by the seasonal index obtained from STL decomposition   
        try:
            rollingDF = pd.DataFrame() 
            deseasonalizedDF = pd.DataFrame()
            for m in timeSeriesCols:
                rollingDF[[m]]= data[[m]].rolling(12).mean() # create dataframe with 12 months rolling average
            
                resultMultiplicative = seasonal_decompose(data[m], model='multi', 
                                                period=freq, extrapolate_trend=freq)
                #resultMultiplicative, resultAdditive, MultiValue = \
                 #   self.decomposeTimeSeries(data[m], freq)
                deseasonalizedDF[[m]] = pd.DataFrame(data[m].values / resultMultiplicative.seasonal)
            plt.plot(deseasonalizedDF)
            plt.title('Deseasonalize by Dividing the series by the seasonal index', fontsize=16)
            plt.show() 
            plt.plot(rollingDF)
            plt.title('Deseasonalize using rolling window', fontsize=16)
            plt.xlabel('Year', fontsize=20)
            plt.show()    
            
            return deseasonalizedDF
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err)) 

    def getDiffCount(self, data):
        return pm.arima.ndiffs(data)
    
    def getSeasonalDiffCount(self, data, freq):
        return pm.arima.nsdiffs(data, freq)

    def getSeasonalDiffTerm(self, data, timeSeriesCols, freq):
        # Estimate the seasonal differencing term,  
        # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.nsdiffs.html#pmdarima.arima.nsdiffs
        try:
            diffDF = pd.DataFrame() # holds diff term for each timeseries col
            
            for m in timeSeriesCols:
                diffDF[[m]]= pm.arima.nsdiffs(data[m], freq) # create dataframe with diff
                
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err)) 


    def makeDiff(self, data, timeSeriesCols, title, diffNo):
        # https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial       
        try:
            diffDF = data[timeSeriesCols].diff(diffNo)
            plt.plot(diffDF)
            plt.title(title, fontsize=16)
            plt.xlabel('Year', fontsize=20)
            plt.show()
                        
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))  
            
    def createLag(self, data, lag, timeSeriesCols): 
        try:
            d1 = data.copy()
            colPrefix = "Lag0"
            prevCol = ''
            #lag_incremental_paid_05
            #numb = 10
            
            for m in timeSeriesCols:
                for x in range(1, lag): 
                    colName = m + colPrefix + str(x)
                    if x == 1:
                       d1[colName] = d1[m].shift(1)
                       prevCol =  colName
                    else:
                       d1[colName] = d1[prevCol].shift(1) #shift(x) is not needed as shifting is based on new column
                       prevCol =  colName            

            print('createLag success')
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))   
            
    def makePredictorTarget(self, data, forecastPeriod):
        try:
            """ 
            Input:  
                   data: original time series 
                   forecastPeriod: number of time steps in the regressors 
            Output:  
                   predictor: 2-D array of regressors 
                   target: 1-D array of target  
            """ 
            predictor = [] 
            target = [] 
            for i in range(forecastPeriod, data.shape[0]): 
                predictor.append(list(data.loc[i-forecastPeriod:i-1])) 
                target.append(data.loc[i]) 
            predictor, target = np.array(predictor), np.array(target) 
            return predictor, target 
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))  
            
    def splitData(self, data, response):
        try:
            
            predictor = data.drop(response, axis=1)
            target = data[response]

            predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2,
                                                                                      random_state=0, shuffle=False)
 
                
            return predictorTrain, predictorTest, targetTrain, targetTest
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))  

    def viewImportantFeatures(self, predictorTrain, model, modelName):
        try:
            # Extract feature importances - https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
            # https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
            sns.set(font_scale = 1.75)
            sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                           "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                           'ytick.color': '0.4'})
            
            # Set figure size and create barplot
            f, ax = plt.subplots(figsize=(12, 9))

            featureImportance = pd.DataFrame({'feature': list(predictorTrain.columns),
                               'importance': model.feature_importances_}).sort_values('importance', ascending=False)

            # Display
            #plt.figure(figsize=(10, 10))
            chart = sns.barplot(x='feature', y='importance', data=featureImportance)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
            
            # Generate a bolded horizontal line at y = 0
            #ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)
            
            # Turn frame off
            ax.set_frame_on(False)
            
            # Tight layout
            plt.tight_layout()
            
            # Save Figure
            #plt.savefig("feature_importance.png", dpi = 1080)
            
            #plt.show()
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def getModelClassifiers(self):
        try:
            SEED = 30
            rf = RandomForestRegressor(
                    n_estimators=100, max_depth=5, max_features='auto', max_leaf_nodes=50, random_state=SEED,
                    min_samples_split=10, bootstrap='True', criterion='mse')
            #bnb = BernoulliNB()
            #nb = GaussianNB(var_smoothing = 0.25)
            svc = SVR()
            knn = KNeighborsRegressor(n_neighbors=3, leaf_size=30, p=2, weights='uniform', 
                                       algorithm='auto', n_jobs=-1, metric='minkowski')
            #lr = LogisticRegression(solver='liblinear', C=100, random_state=SEED)
            nn = MLPRegressor((80, 10), early_stopping=False, random_state=SEED)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
            xgb = XGBRegressor(max_depth = 3, learning_rate=0.1, n_estimators=150, silent=True, objective='reg:squarederror',
                    booster = 'gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                    subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                    base_score=0.5, random_state=SEED, seed=None, missing=None)
            #lda = LinearDiscriminantAnalysis(solver='svd', tol=0.0001)
            #qda = QuadraticDiscriminantAnalysis(reg_param=0.0, store_covariance=False, tol=0.0001)
            ada = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear')
            bagging = BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0, 
                                        max_features=1.0, bootstrap=True, bootstrap_features=False, 
                                        oob_score=False, warm_start=False, n_jobs=None)
            etc = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, 
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                       max_features='auto', max_leaf_nodes=None, 
                                       min_impurity_decrease=0.0, min_impurity_split=None, 
                                       bootstrap=False, oob_score=False, n_jobs=-1,)
            ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, 
                                    max_iter=None, tol=0.001, solver='auto')
            sgd = SGDRegressor(alpha=1.0, penalty='l2')
            #bnb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
            dtc = DecisionTreeRegressor(criterion='mse', splitter='best')

            """
            models = {'bernoulli nb' : BernoulliNB, 'extra trees' : etc, 'ridge' : ridge, 'sgd' : sgd,   
                      'mlp-nn': nn, 'gbm': gb, 'xgb' : xgb, 'lda' : lda, 'qda' : qda, 
                      'random forest': rf, 'bagging' : bagging,  'ada boost' : ada,
                      'knn': knn, 'svm': svc, 'naive bayes': nb, 'DTC' : dtc} """
            
            """ models = { 'bagging' : bagging,  'ada boost' : ada, 'knn': knn, 'DTC' : dtc,
                      'random forest': rf, 'extra trees' : etc,
                      'ridge' : ridge, 'gbm': gb, 'xgb' : xgb, 'mlp-nn': nn, 'svm': SVR
                       }
            """
            
            models = { 'bagging' : bagging,  'ada boost' : ada, 'knn': knn
                       }

            return models
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def getModelParam(self):
        try:
            parameters = {}

            parameters.update({"knn": {"n_neighbors": [3,5,11,19],
                                       "p": [1, 2, 3, 4, 5],
                                       "leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                       "n_jobs": [-1],
                                       "weights" : ['uniform', 'distance'],
                                       "metric" : ['euclidean', 'manhattan']
                                       }})
            parameters.update({"svm": {"kernel": ["linear", "rbf", "poly"],
                                       "gamma": ["auto"], "C": [0.1, 0.5, 1, 5, 10, 50, 100],
                                       "degree": [1, 2, 3, 4, 5, 6]
                                       }})
            #parameters.update({"naive bayes": {"var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
             #                          }})
            parameters.update({"mlp-nn": {"hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                                       "activation": ["identity", "logistic", "tanh", "relu"],
                                       "learning_rate": ["constant", "invscaling", "adaptive"],
                                       "max_iter": [100, 200, 300, 500, 1000, 2000],
                                       "alpha": list(10.0 ** -np.arange(1, 10)),
                                       }})
            parameters.update({"random forest": {"max_features": ["auto", "sqrt", "log2"],
                                                 "max_depth" : [3, 4, 5, 6, 7, 8],
                                                 "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                                 "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                                 "criterion" :["mse", "mae"]
                                                 }})            
            parameters.update({"gbm": {"learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                       "max_depth": [2,3,4,5,6],
                                       "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                       "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                       "max_features": ["auto", "sqrt", "log2"],
                                       "subsample": [0.8, 0.9, 1]
                                       }})
            parameters.update({"xgb": {"max_depth":range(3,10,2), "min_child_weight":range(1,6,2), 
                                       "gamma":[i/10.0 for i in range(0,5)],
                                       "subsample":[i/10.0 for i in range(6,10)], 
                                       "colsample_bytree":[i/10.0 for i in range(6,10)],
                                       "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100], 
                                       "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3]
                                       }})            
            #parameters.update({"lda": {"solver": ["svd"], 
             #                         }})
            #parameters.update({"qda": {"reg_param":[0.01*ii for ii in range(0, 101)], 
             #                         }})
            parameters.update({"ada boost": {"learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
                                             }})
            parameters.update({"bagging": {"n_estimators": [200],
                                           "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                           "n_jobs": [-1]
                                           }})
            parameters.update({"extra trees": {"max_features": ["auto", "sqrt", "log2"],
                                               "max_depth" : [3, 4, 5, 6, 7, 8],
                                               "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                               "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                               "criterion" :["mse", "mae"]     ,
                                               "n_jobs": [-1]
                                               }})
            parameters.update({"ridge": {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                                         }})
            parameters.update({"sgd": {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                                       #"penalty": ["l1", "l2"],
                                       "n_jobs": [-1]
                                       }})
            #parameters.update({"bernoulli nb": {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
             #                                   }})

            parameters.update({"DTC": {"criterion" :["mse", "mae"],
                                       "splitter": ["best", "random"],
                                       "max_features": ["auto", "sqrt", "log2"],
                                       "max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                                       "min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                       "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                       }})

            """            
            parameters.update({"LSVC": { 
                                        "classifier__penalty": ["l2"],
                                        "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
                                         }})
            
            parameters.update({"NuSVC": { 
                                        "classifier__nu": [0.25, 0.50, 0.75],
                                        "classifier__kernel": ["linear", "rbf", "poly"],
                                        "classifier__degree": [1,2,3,4,5,6],
                                         }})
            

            parameters.update({"extra trees": {"classifier__criterion" :["gini", "entropy"],
                                               "classifier__splitter": ["best", "random"],
                                               "classifier__class_weight": [None, "balanced"],
                                               "classifier__max_features": ["auto", "sqrt", "log2"],
                                               "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                                               "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                               "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                               }})  
            
            # Update dict with Decision Tree Classifier

            """            
            return parameters
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def regressEnsemble(self, estimators, predictorTrain, targetTrain, predictorTest, targetTest):
        try:
            # https://www.datacamp.com/community/tutorials/ensemble-learning-python
            votingCLF = VotingRegressor(estimators=estimators, n_jobs=-1, verbose=True)
            votingCLF.fit(predictorTrain, targetTrain)
            predictions = votingCLF.predict(predictorTest)
            self.getMAPE(targetTest, predictions, 'Ensemble')
            
            # metrics.plot_roc_curve(models[m], predictorTest, targetTest)
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))



    def regressMultipleModels(self, predictorTrain, predictorTest, targetTrain, targetTest, forecastPeriod):
        try:
            # https://mlfromscratch.com/gridsearch-keras-sklearn/#/
            # https://www.dataquest.io/blog/introduction-to-ensembles/ 
            kFold = 5
            modelList = self.getModelClassifiers()
            
            modelPredictions = np.zeros((targetTest.shape[0], len(modelList))) 
            modelPredictions = pd.DataFrame(modelPredictions)

            cvPredictions = np.zeros((targetTest.shape[0], len(modelList))) 
            cvPredictions = pd.DataFrame(cvPredictions)
            
            rscvPredictions = np.zeros((targetTest.shape[0], len(modelList))) 
            rscvPredictions = pd.DataFrame(rscvPredictions)
            
            rfecvPredictions = np.zeros((targetTest.shape[0], len(modelList))) # not applicable for 'knn', 'svm', 'naive bayes', 'mlp-nn', 'qda', 'bagging'
            rfecvPredictions = pd.DataFrame(rfecvPredictions)
            
            modelProbabilities = np.zeros((targetTest.shape[0], len(modelList))) # not applicable for ridge and SGD
            modelProbabilities = pd.DataFrame(modelProbabilities)
            
            measurement = pd.DataFrame(columns = ['Model', 'Techniques', 'MapeMean', 'R2', 'R2Score', 'MSE', 'MAE']) 
            
            modelParamGrids = self.getModelParam()
            print("Fitting models.")
            #probCols = list()
            #predCols = list()
            cols = list()
            estimators = list()
            estimatorToDrop = ['bernoulli nb']
            bestRSEstimators = list()
            ensembleEstimators = list() # estimators which support ensemble are copied from bestRSEstimators
            bestRSParams = list()
            
            # bestRSScores = list()
            cvScores = list()
            rfeEstimators = list()
            # train and predict each model with all features
            for i, (name, m) in enumerate(modelList.items()):
                #
                # build model with all features and without CV
                #
                print("%s..." % name, end=" ", flush=False)

                if (name == 'svm'):
                    m = SVR()
                
                m.fit(predictorTrain, targetTrain)
                    
                    
                modelPredictions.iloc[:, i] = m.predict(predictorTest)
                #predCols.append(name)
                
                #cols.append(name)
                modelPredictions = modelPredictions.rename(columns={i: name})
                mapeMean, r2, r2Score, mse, mae = self.getMAPE(targetTest, modelPredictions[name], 'model without CV ' + name) 
                # mapeMean, r2, r2Score, mse, mae = self.getMAPE(targetTest[0:forecastPeriod], modelPredictions[name][0:forecastPeriod]) 
                measurement = measurement.append({'Model' : name, 'Techniques' : 'All Feature without CV', 
                                          'MapeMean' : mapeMean, 'R2' : r2, 'R2Score' : r2Score, 
                                         'MSE' : mse, 'MAE' : mae},
                                         ignore_index = True) 
                
                #estimators.append((name,m)) # do not delete extra bracket. It converts each item in the list into tuple
                print("done")
                #
                # build model with hyper parameter tuning and with CV
                #
                paramGrid = modelParamGrids[name]
                bestRSEstimator, bestRSParam, rsPrediction = self.randomizedSearch(paramGrid, m, predictorTrain, 
                                            targetTrain, predictorTest, targetTest, name)
                rscvPredictions.iloc[:, i] = rsPrediction
                rscvPredictions = rscvPredictions.rename(columns={i: name})
                bestRSEstimators.append((name, bestRSEstimator))
                bestRSParams.append(bestRSParam)
                
                mapeMean, r2, reScore, mse, mae = self.getMAPE(targetTest, rscvPredictions[name], 'Hyper parameter tuned model ' + name) 
                measurement = measurement.append({'Model' : name, 'Techniques' : 'Randomized Search with CV', 
                                          'MapeMean' : mapeMean, 'R2' : r2, 'R2Score' : r2Score, 
                                         'MSE' : mse, 'MAE' : mae},
                                         ignore_index = True) 
                
                #
                # predict with cross_val_predict with CV
                #
                #kFold = StratifiedKFold(shuffle=True, random_state=50)
                cvPredictions.iloc[:, i] = cross_val_predict(bestRSEstimator, predictorTest, targetTest, cv=kFold)
                cvPredictions = cvPredictions.rename(columns={i: name})
                cvScores.append((name, cross_val_score(bestRSEstimator, predictorTest, targetTest, cv=kFold)))
                #self.modelMetrics(targetTest, cvPredictions)
                
                mapeMean, r2, r2Score, mse, mae = self.getMAPE(targetTest, cvPredictions[name], 'cross val predict with CV model ' + name) 
                measurement = measurement.append({'Model' : name, 'Techniques' : 'Cross val with CV', 
                                          'MapeMean' : mapeMean, 'R2' : r2, 'R2Score' : r2Score, 
                                         'MSE' : mse, 'MAE' : mae},
                                         ignore_index = True) 
                

                #
                # predict with recursive feature elimination and CV
                #
                """if (name not in ['knn', 'svm', 'naive bayes', 'mlp-nn', 'qda', 'bagging']):
                    rfeCV = RFECV(estimator=bestRSEstimator, step=1, cv=kFold, scoring='r2')
                    rfeCV.fit(predictorTrain, targetTrain)
    
                    rfeCols = list(predictorTrain.columns[rfeCV.support_])
                    predictorTrainRFE = predictorTrain[rfeCols]
                                        rfeCols = list(predictorTrain.columns[rfeCV.support_])
                    predictorTrainRFE = predictorTrain[rfeCols]
                    
                    rfecvPredictions.iloc[:, i] = rfeCV.predict(predictorTest)
                    rfeCVPredictions = rfecvPredictions.rename(columns={i: name})
                    
                    mapeMean, r2, r2Score, mse, mae = self.getMAPE(targetTest, rfeCVPredictions[name], 'RFECV model ' + name) 
                    measurement = measurement.append({'Model' : name, 'Techniques' : 'RFE with CV', 
                                          'MapeMean' : mapeMean, 'R2' : r2, 'R2Score' : r2Score, 
                                         'MSE' : mse, 'MAE' : mae},
                                         ignore_index = True) 
                    
                    plt.figure(figsize=(16, 9))
                    plt.title('Recursive Feature Elimination with Cross-Validation - ' + name, fontsize=18, fontweight='bold', pad=20)
                    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
                    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
                    plt.plot(range(1, len(rfeCV.grid_scores_) + 1), rfeCV.grid_scores_, color='#303F9F', linewidth=3)
                    
                    plt.show() 

                #
                # Visualize
                #
                if (name not in ['bernoulli nb', 'knn', 'svm', 'naive bayes', 'mlp-nn', 'lda', 'qda', 'bagging', 'ridge', 'sgd']):
                    print('Optimal number of features: {}'.format(rfeCV.n_features_))
                    rfeCV.estimator_.feature_importances_
                    self.viewImportantFeatures(predictorTrain, m, name)
                    self.viewImportantFeatures(predictorTrain, bestRSEstimator, name)
                    self.viewImportantFeatures(predictorTrain.iloc[:,0:rfeCV.n_features_], rfeCV.estimator_, name)
                """

                # Update classifier parameters
                #tuned_params = {item[12:]: best_params[item] for item in best_params}
                #classifier.set_params(**tuned_params)


            
            self.visualizeScoreComparison(measurement)
            
            corrmat(modelPredictions.corr(), inflate=False)
            corrmat(rscvPredictions.corr(), inflate=False)
            corrmat(cvPredictions.corr(), inflate=False)
            #corrmat(rfecvPredictions.corr(), inflate=False)
            
            
            for i in range((len(bestRSEstimators))): # some models are not applicable for ensemble. They need to be dropped from bestRSEstimators
                if bestRSEstimators[i][0] not in estimatorToDrop: #[i][0] returns model name
                    ensembleEstimators.append(bestRSEstimators[i])
                    #del bestRSEstimators[i]
                print (i)     
            self.regressEnsemble(ensembleEstimators, predictorTrain, targetTrain, predictorTest, targetTest)
            
            print("Done.\n")
            #return modelPredictions, modelProbabilities, rscvPredictions, cvPredictions, rfecvPredictions, bestRSEstimators
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def randomizedSearch(self, paramGrid, m, predictorTrain, targetTrain, predictorTest, targetTest, name):
        try:
            # train and predict each model with grid search CV
            kFold = 5
            #kFold = StratifiedKFold(shuffle=True, random_state=50)
            #gscv = GridSearchCV(m, param_grid=paramGrid, cv = kFold,  n_jobs= -1, verbose = 1, scoring = "roc_auc")
            rscv = RandomizedSearchCV(m, param_distributions=paramGrid, n_jobs=-1, scoring='r2', cv=kFold,
                                n_iter=10, verbose=1, random_state=30)  
            # Fit gscv
            print(f"Now tuning {m}.")
            rscv.fit(predictorTrain, np.ravel(targetTrain))  
            prediction = rscv.best_estimator_.predict(predictorTest)
            #rscvPredictions = rscvPredictions.rename(columns={i: name})
            
            #auc = metrics.roc_auc_score(targetTest, prediction)
            #bestRSParams.append(rscv.best_params_)
            bestParams = rscv.best_params_
            
          
            self.plotLearningCurve(rscv.best_estimator_, 'Learning Curves', predictorTrain, targetTrain, 
                        cv = kFold, n_jobs = -1)
            #bestRSScores.append(rscv.best_score_)
            return rscv.best_estimator_, rscv.best_params_, prediction
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def plotLearningCurve(self, estimator, title, train, test, ylim = None, cv = None,
                        n_jobs = -1, trainSizes = np.linspace(0.1, 1.0, 5)):
        try:
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            trainSizes, trainScores, testScores = learning_curve(estimator, train, test, cv = cv,
                                                                 n_jobs = n_jobs, train_sizes = trainSizes)
            trainScoresMean = np.mean(trainScores, axis=1)
            trainScoresSTD = np.std(trainScores, axis=1)
            testScoresMean = np.mean(testScores, axis=1)
            testScoresSTD = np.std(testScores, axis=1)
            plt.grid()
        
            plt.fill_between(trainSizes, trainScoresMean - trainScoresSTD,
                             trainScoresMean + trainScoresSTD, alpha=0.1, color="r")
            plt.fill_between(trainSizes, testScoresMean - testScoresSTD,
                             testScoresMean + testScoresSTD, alpha=0.1, color="g")
            plt.plot(trainSizes, trainScoresMean, 'o-', color="r", label="Training score")
            plt.plot(trainSizes, testScoresMean, 'o-', color="g", label="Cross-validation score")
        
            plt.legend(loc="best")
            plt.show()
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def visualizeScoreComparison(self, measurement):
        try:
            # Set graph style
            sns.set(font_scale = 1.75)
            sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})
    
            # Colors
            training_color = sns.color_palette("RdYlBu", 10)[1]
            test_color = sns.color_palette("RdYlBu", 10)[-2]
            colors = [training_color, test_color]
            
            # Set figure size and create barplot
            f, ax = plt.subplots(figsize=(12, 9))
            
            sns.barplot(x="MapeMean", y="Model", hue="Techniques", palette = colors,
                        data=measurement)
            
            # Generate a bolded horizontal line at y = 0
            ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)
            
            # Turn frame off
            ax.set_frame_on(False)
            
            # Tight layout
            plt.tight_layout()
            plt.show()
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))


    def MLPRegress(self, trainData, testData, forecastPeriod, colName):
        try:
            # https://github.com/neelabhpant/Deep-Learning-in-Python/blob/master/ClimateChange_Prediction.ipynb
            
            
            predictorTrain, targetTrain = self.makePredictorTarget(trainData[colName], forecastPeriod)
            predictorTest, targetTest = self.makePredictorTarget(testData[colName], forecastPeriod)

            def buildNN(neurons1=32, neurons2=16, activation = 'linear', dropoutRate = 0.2, optimizer = 'Adam', initMode='uniform', learnRate=0.0001, momentum = 0.0, weightConstraint=0.0):
                model = Sequential()
                model.add(Dense(neurons1,input_dim = forecastPeriod,kernel_initializer = initMode, activation = activation, kernel_constraint=maxnorm(weightConstraint)))
                #model.add(Dropout(dropoutRate))
                model.add(Dense(neurons2,input_dim = 16,kernel_initializer = initMode, activation = activation))
                model.add(Dropout(dropoutRate))
                model.add(Dense(1, activation = activation))
                
                adam = SGD(lr = learnRate, momentum = momentum)
                model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
                return model

            
            inputLayer = Input(shape=(forecastPeriod,), dtype='float32') 
            dense1 = Dense(32, activation='linear')(inputLayer) 
            dense2 = Dense(16, activation='linear')(dense1) 
            #dense3 = Dense(16, activation='linear')(dense2) 
            dropoutLayer = Dropout(0.6)(dense2) 
            outputLayer = Dense(1, activation='linear')(dropoutLayer) 
            
            model = Model(inputs=inputLayer, outputs=outputLayer) 
            model.compile(loss='mean_squared_error', optimizer='adam') 
            model.summary() 
            
            #model = KerasClassifier(build_fn=buildNN, epochs=100, batch_size=10, verbose=0)
            #kFold = StratifiedKFold(shuffle=True, random_state=50)
            #cross_val_score(model, predictorTrain, targetTrain, cv=5)
            #CVPredictions = cross_val_predict(model, predictorTest, targetTest, cv=3, n_jobs=-1)
            
            checkpointPath = os.path.join('keras_models', 'model.{epoch:02d}-{val_loss:.4f}.hdf5') 
            #save_weights_at = os.path.join('model.h5') 
            earlyStopCallback = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
            saveBest = ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min') 
            
            history = model.fit(x=predictorTrain, y=targetTrain, batch_size=16, epochs=20, callbacks=[earlyStopCallback, saveBest],
                                verbose=1, validation_data=(predictorTest, targetTest), shuffle=True) 
            
            visualizeLoss(history, "Training and Validation Loss")
            #best_model = load_model(os.path.join('keras_models', 'PRSA_data_Air_Pressure_MLP_weights.18-94530.7887.hdf5')) 
            predictions = model.predict(predictorTest) 
            predictions = pd.DataFrame(predictions)
            predictions = predictions.loc[:,0]
            
            #testData[colName + 'Predict'] = predictions
            #predictions = scaler.inverse_transform(predictions)
            #predictions = np.squeeze(predictions)
 
            r2 = r2_score(testData[colName].loc[forecastPeriod:], predictions) 
            print('R-squared for the test data set:', round(r2,4)) 
            
            mae = mean_absolute_error(testData[colName].loc[forecastPeriod:], predictions)
            print('MAE for the test data set:', round(mae, 4))
            
            
            plt.figure(figsize=(5.5, 5.5)) 
            plt.plot(range(50), testData[colName].loc[forecastPeriod:56], linestyle='-', marker='*', color='r') 
            plt.plot(range(50), predictions[:50], linestyle='-', marker='.', color='b') 
            plt.legend(['Actual','Predicted'], loc=2) 
            plt.title('Actual vs Predicted - ' + colName) 
            plt.ylabel(colName) 
            plt.xlabel('Index') 
            plt.show()
            
            return predictions, r2, mae, model
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err)) 

    def visualizeLoss(history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def getMAPE(self, actual, predicted, name):
            try:
                actual.reset_index(drop=True, inplace=True)
                predicted.reset_index(drop=True, inplace=True)
                yActual, yPredicted = np.array(actual), np.array(predicted)
                MAPE = (np.abs((yActual - yPredicted) / yActual)) * 100
                mapeMean = np.mean(np.abs((yActual - yPredicted) / yActual)) * 100
    
                actualDF = pd.DataFrame({'Actual': actual, 'Predicted': predicted, 'MAPE': MAPE})
                df1 = actualDF.head(50)
                print(df1)
                fig, ax = plt.subplots()
                ax = sns.scatterplot(x="Actual", y="Predicted", data=actualDF)
                ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title("Actual vs Predicted - "+name)
    
                plt.show()
    
                df1.plot(kind='bar', figsize=(10, 8), title = 'Actual vs Predicted - '+name)
                #plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
                #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
                plt.show()
    
                from sklearn.metrics import mean_squared_error, r2_score


                mse = mean_squared_error(actual, predicted)

                r2Score = r2_score(actual, predicted)
                
                r2 = r2_score(actual, predicted) 
            
                mae = mean_absolute_error(actual, predicted)

                
                return mapeMean, r2, r2Score, mse, mae
            except Exception as exp:
                self.errObj = ErrorHandler()
                err = self.errObj.handleErr(str(exp))
                print(str(err))

class scaleing:
    def minMaxScaler(self, data):
        try:
            d1 = data.copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaledData = scaler.fit_transform(d1)
            scaledData = pd.DataFrame(scaledData, columns=d1.columns)
            return d1, scaler
        except Exception as exp:
            self.errObj = ErrorHandler()
            err = self.errObj.handleErr(str(exp))
            print(str(err))
            
def main():
    pd.set_option('display.max_columns', 43)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', None)
    pd.set_option('display.expand_frame_repr', True)
    plt.interactive(False)

    response = 'FATALS'
    preProcessorObj = PreProcessor()

    data = preProcessorObj.getAccident()
    # aggregationCols = preProcessorObj.getAggregationCols - not needed
    timeSeriesCols = preProcessorObj.getTimeSeriesCols() # get only time series columns

    ######## 1. missing value
    missingValueObj = MissingValue()
    #missingValueObj.visualizeMissingValue(data)

    # Imputation of missing value by different tecniques
    cleanData = missingValueObj.dropColumn(data, threshold=30.0)
    numericCols = preProcessorObj.geNumericCols(cleanData)

    
    if cleanData.isnull().values.any():
        #imputedDataMean = MissingValueObj.imputeByMean(cleanData, nonNumeric)
        imputedData = missingValueObj.imputeByMice(cleanData, numericCols)
        #imputedDataKNN = missingValueObj.imputeByKNN(cleanData, numericCols)
        #imputedDataMissForest = missingValueObj.imputeByMissForest(cleanData, numericCols)
    else:
        imputedData = cleanData

    #del MissingValueObj
    ######## end missing value

    dataNumeric = preProcessorObj.getNumericData(imputedData)

    ####### 2. visualize data
    visualizeObj = visualize()
    categoricalCols = preProcessorObj.getCategoricalCols(imputedData)
    numericCols = preProcessorObj.geNumericCols(imputedData)
    visualizeObj.visualizeCorrelation(imputedData)
    visualizeObj.snsPlot(imputedData, numericCols, categoricalCols, response)
    #visualizeObj.ggplotPlot(imputedData, numericCols, categoricalCols, response)
    
    ##### end visualization
    modelerObj = modeler()
    scaleingObj = scaleing()
    
    ####### 3. Outlier detection
    scaleingObj = scaleing()
    scaledData, scaler = scaleingObj.minMaxScaler(dataNumeric)
    
    outlierObj = Outlier()
    outlierObj.visualizeOutlier(scaledData, preProcessorObj.importantCatCols)
    
    ##### end of outlier detection
    
    ##### 4. prepare data for aggregation
    # drop date time cols and State. Otherwise they will be used in aggregation. 
    d1 = imputedData.drop(preProcessorObj.getDateTimeCols(), axis=1)
    d1 = d1.drop('STATE', axis=1)
    
    # modelling will be done on aggregated data on date or month or day of week. To use categorical columns 
    # in the modelling we decided to use one hot encoding and sum up occurrence of each
    # categorical value for a given date
    dummyEncodedImputedData = pd.get_dummies(d1)

    aggrCols = preProcessorObj.createAggregationDict(dummyEncodedImputedData, 'sum')
    # aggregate on date
    dummyEncodedDailyData = preProcessorObj.getAggregateData(dummyEncodedImputedData,'AccidentDate', aggrCols)
    
    # delete all columns where unique value of a column is less than 3
    ff=dummyEncodedDailyData.nunique() > 2
    dd=ff[ff==True].index
    dummyEncodedDailyDataFinal = dummyEncodedDailyData[dd]
    
    # aggregate on month year
    #dummyEncodedMonthlyData = dummyEncodedDailyData.groupby(pd.Grouper(freq="M")).agg(aggrCols)
    
    ######## end of preparation of data aggregation  ###################
    #if (len(timeSeriesCols)>1):
     #   fig, axes = plt.subplots(nrows=round(len(timeSeriesCols)/2), ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
        
    ###### 5. viualize time series
    #visualizeObj.plotTimeSeries(dummyEncodedDailyDataFinal, timeSeriesCols)
    #visualizeObj.plotSeasonality(dummyEncodedMonthlyData, timeSeriesCols)
    ######## end of visualization of time series
    
    ####### 6. decompose time series
    #for m in timeSeriesCols:
     #   resultMultiplicative, resultAdditive, MultiValueDF = \
      #  modelerObj.decomposeTimeSeries(dummyEncodedDailyDataFinal[[m]],freq=7)

        #resultMultiplicativeM, resultAdditiveM, MultiValueDFM = \
        #modelerObj.decomposeTimeSeries(dummyEncodedMonthlyData[[m]],freq=12)

    ####### end of decomposition of time series  ###########
    
    ######### 7. test for stationarity of time sries
    """
    for m in timeSeriesCols: 
        # https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-one/
        pd.options.display.float_format = '{:.8f}'.format
            testStationarity(dummyEncodedDailyDataFinal[[m]],'raw data')
    
        print ("ADF Test for " + m + "\n")
        modelerObj.checkADF(dummyEncodedDailyDataFinal[[m]])
        print ("\n")
        
        print ("KPSS Test for " + m + "\n")
        modelerObj.checkKPSS(dummyEncodedDailyDataFinal[[m]])
        print ("\n")
        """
    
    ########### end of test for stationarity of time sries ########
    
    ####### 8. Treat missing value of time series
    #Backward Fill
    #Linear Interpolation
    #Quadratic interpolation
    #Mean of nearest neighbors
    #Mean of seasonal couterparts
    
    ######## end of Treat missing value of time series
    
    ########### 9. Make time series stationary 
    # You can make series stationary by:

    # a. Differencing the Series (once or more)
    # b. Take the log of the series
    # c. Take the nth root of the series
    # d. Combination of the above
    # Why make a non-stationary series stationary before forecasting?
    # Forecasting a stationary series is relatively easy and the forecasts are more reliable.
    """
    stationaryDailyData = dummyEncodedDailyDataFinal.copy()
    for m in timeSeriesCols:
        #d1 = preProcessorObj.convertFloatToNumeric(dataNumeric)
        # make the time series stationary and plot
        diffCnt = modelerObj.getDiffCount(stationaryDailyData[m]) # get number of diff required 
       
        if (diffCnt > 0):
            stationaryDailyData[m] = abs(stationaryDailyData[m] - stationaryDailyData[m].shift(diffCnt))
            stationaryDailyData[m].plot(title='Time Series - ' + m)
            plt.show()
            
    stationaryDailyData = stationaryDailyData.dropna(how='any',axis=0)
    """
    ############ end of Make time series stationary ###########
    
    ######## 10. detrend time series
    """
    for m in timeSeriesCols:                  
        detrendLSFDailyData = modelerObj.detrendLSF(dummyEncodedDailyDataFinal[m].values,title=m + ' detrended by subtracting the least squares fit')
        resultMultiplicative = seasonal_decompose(data[m], model='multi', 
                                            period=freq, extrapolate_trend=freq)
        dtrendTrendDailyData = modelerObj.detrendTrendComponent(data, resultMultiplicative.trend, title=m + ' detrended by subtracting the trend component')
        testStationarity(dtrendTrendDailyData,'de-trended data')
        # ADF_test(y_detrend,'de-trended data')
        modelerObj.checkADF(dtrendTrendDailyData[[m]])

    visualizeObj.plotTimeSeries(detrendLSFDailyData, timeSeriesCols) """
    ####### end of detrending time series #######
    
    ####### 11. test seasonality of time series
    # The common way is to plot the series and check for repeatable patterns in fixed 
    #time intervals. So, the types of seasonality is determined by 
    # Hour of day, Day of month, Weekly, Monthly, Yearly,     
    
    
    
    ######## end of seasonality of time series
    
    ###### 12. deseasonalize time series
    #modelerObj.deseasonalize(dummyEncodedDailyDataFinal, timeSeriesCols, freq=365)
    
    ###### end of deseasonalize time series  #########
    

    #nonTimeSeriesCols = [col for col in data.columns if col not in timeSeriesCols]
    

    
    
    ######## 13. create lag for all time series columns
    #scaledData, scaler = scaleingObj.minMaxScaler(imputedData[timeSeriesCols])
    #scaledData = pd.concat([scaledData, imputedData[nonTimeSeriesCols]], axis=1)
    lagData = modelerObj.createLag(dummyEncodedDailyDataFinal, 7, timeSeriesCols)
    lagData = lagData.dropna(how='any',axis=0)
    ######## end create lag for all time series columns
    
    modelCompare = pd.DataFrame()
    accuracy = pd.DataFrame(columns=['ColumnName', 'R2', 'MAE', 'MAPE'])
    
    ######## 14. autocorrelation and partial autocorrelation ########


    ######## end of autocorrelation and partial autocorrelation ########
    
    
    ###### 15. forecastability of a time series #######
    
    
    ###### end of forecastability of a time series #######
    
    ###### 16. smoothen a time series ######
    # https://www.machinelearningplus.com/time-series/time-series-analysis-python/
    
    ###### end of smoothen a time series ######
    
    ###### 17. Granger Causality test to know if one time series is helpful in forecasting another ######
    
    
    ###### end of Granger Causality test to know if one time series is helpful in forecasting another ######
    
    for m in timeSeriesCols:
        #d1 = preProcessorObj.convertFloatToNumeric(dataNumeric)
        # make the time series stationary and plot
        """diffCnt = modelerObj.getDiffCount(dummyEncodedDailyDataFinal[m]) # get number of diff required 
        if (diffCnt > 0):
            dummyEncodedDailyDataFinal[m] = abs(dummyEncodedDailyDataFinal[m] - dummyEncodedDailyDataFinal[m].shift(diffCnt))
            dummyEncodedDailyDataFinal = dummyEncodedDailyDataFinal.dropna(how='any',axis=0)
            dummyEncodedDailyDataFinal[m].plot()
            plt.show()
        
        diffSeasonalCnt = modelerObj.getSeasonalDiffCount(dummyEncodedDailyDataFinal[m], 7) # get number of seasonal diff required
        if diffSeasonalCnt > 0:
            dummyEncodedDailyDataFinal[m] = dummyEncodedDailyDataFinal[m] - dummyEncodedDailyDataFinal[m].shift(diffSeasonalCnt)
            dummyEncodedDailyDataFinal = dummyEncodedDailyDataFinal.dropna(how='any',axis=0)
            dummyEncodedDailyDataFinal[m].plot()"""
        
        lagData = modelerObj.createLag(dummyEncodedDailyDataFinal, 7, [m])
        lagData = lagData.dropna(how='any',axis=0)
        
        lagScaledData, scaler = scaleingObj.minMaxScaler(lagData)
        
        #dummyEncodedDailyDataFinal[m] = array.diff(dummyEncodedDailyDataFinal, lag=1, differences=diffCnt)
        predictorTrain, predictorTest, targetTrain, targetTest = modelerObj.splitData(lagData, m)
        modelerObj.regressMultipleModels(predictorTrain, predictorTest, targetTrain, targetTest, forecastPeriod=15)
        
        
        
        #predictions, r2, mae, model = modelerObj.MLPRegress(trainData, testData, forecastPeriod=7, colName=m)
        #print(model.get_weights())
        modelCompare[m] = testData[m].loc[7:]
        modelCompare[m + 'Predict'] = predictions
        modelCompare[m + 'MAPE'], mape = modelerObj.getMAPE(testData[m].loc[7:], pd.Series(list(predictions)), name)
        accuracy = accuracy.append({'ColumnName':m, 'R2':r2, 'MAE':mae, 'MAPE':mape}, ignore_index=True)    

        weights = model.get_weights()
        for i in range(len(weights)):
            print(weights[i].shape)
            
        print(weights[4])
    
    accuracy.to_csv('accuracy.csv', index=False)
    modelCompare.to_csv('model Compare.csv', index=False)
    #scaledData = scaleingObj.minMaxScaler(dataNumeric)

    debug = 0


if __name__ == '__main__':
    main()



