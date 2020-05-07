# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:52:00 2020

@author: vskrinjar

This is a simple proof-of-concept of a GUI machine learning application.
"""
# import GUI libraries
#
#import Qt application and widget base class
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
#import visual elements
from PyQt5.QtWidgets import QPushButton, QLineEdit, QComboBox, QLabel, QTableWidget, QTableWidgetItem, QFileDialog #unused: QRadioButton, QStatusBar, QToolBar, 
#import layout managers
from PyQt5.QtWidgets import QGridLayout #not used: ,QHBoxLayout, QVBoxLayout,  QFormLayout
# some more Qt stuff
from PyQt5.QtCore import Qt
#
import sys
from functools import partial #for controller
#
# import scientific and ML libs
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold



#this is the main table - shown in the main GUI
class TableView(QTableWidget):
    def __init__(self, data, *args):
        QTableWidget.__init__(self, *args)
        self._data = data
        self.setData()
 
    def setData(self): 
        #set up header, and prepare size
        horHeaders = ['x1', 'x2', 'X3', 'target']
        self.setHorizontalHeaderLabels(horHeaders)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        #
        for row in range(self._data.shape[0]):
            for col in range(self._data.shape[1]):
                #newitem = QTableWidgetItem(str(self._data[row,col])) #this damn str() cost me about 1hour of life
                newitem = QTableWidgetItem(f"{self._data[row,col]:.2f}")
                self.setItem(row, col, newitem)
        
    def updateData(self, newdata):
        self._data = newdata
        self.setData()


# this is the main GUI
class appGUI(QMainWindow):

    def __init__(self, datasource):
        super().__init__()
        
        #main window properties
        self.setWindowTitle("MiniML GUI")
        #self.setFixedSize(500,400)
        #set up general layout and the central widget
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # set up central table widget
        self.table = TableView(datasource, 100, 4)
        #self.table.setFixedSize(200,300)
        # add controls and the central table widget to the general layout
        self._createControls()
        self._createDataSelection()
        self._createReport()
        self.generalLayout.addWidget(self.table, 1, 0, 2, 1)
        self._createOptiLearn()
        
        
    # this creates the controls GUI    
    def _createControls(self):
        """Create controls """
        #prepare grid layout for: model selection and train-test split 
        self.controlLayout = QGridLayout()
        #create "train model" button and label
        self.train_model_label = QLabel('<h3> Select and train model <\h3>')
        self.train_model_button = QPushButton("Train")
        #self.train_model_button.setFixedSize(150,30)
        # create model selection menu
        self.model_box = QComboBox()
        # add all supported models to the selection list
        for model in models_list:
            self.model_box.addItem(model)
        # add label and selection for train-test split:
        self.train_test_label = QLabel('Select test size for train-test split (%):')
        self.test_ratio_selection = QLineEdit()
        self.test_ratio_selection.setAlignment(Qt.AlignRight)
        self.test_ratio_selection.setText('10%')
        #add to the grid
        self.controlLayout.addWidget(self.train_model_label, 0, 0, 1, 2)
        self.controlLayout.addWidget(self.model_box, 1, 0)
        self.controlLayout.addWidget(self.train_model_button, 1, 1)
        self.controlLayout.addWidget(self.train_test_label, 2,0)
        self.controlLayout.addWidget(self.test_ratio_selection, 2,1)
        #set alignments
        self.controlLayout.setAlignment(self.train_model_label, Qt.AlignTop)
        self.controlLayout.setAlignment(self.model_box,  Qt.AlignTop)
        self.controlLayout.setAlignment(self.train_model_button,Qt.AlignTop)
        #add controls to the general layout
        self.generalLayout.addLayout(self.controlLayout, 0, 1)
        #
        #prepare grid layout for parameter selection for the models.
        self.modelTuningLayout = QGridLayout()
        #add section title
        self.tune_model_label = QLabel('<h3> Tune model parameters <\h3>')
        #Linear models have up to 2 parameters called alpha 1 and 2. They are "hidden" for some models.
        self.alpha1Label = QLabel('Alpha_1:')
        self.alpha1 = QLineEdit()
        self.alpha1.setAlignment(Qt.AlignRight)
        self.alpha1.setText('NA')
        self.alpha1.setEnabled(False)
        self.alpha1.setFixedSize(150,20)
        #
        self.alpha2Label = QLabel('Alpha_2:')
        self.alpha2 = QLineEdit()
        self.alpha2.setAlignment(Qt.AlignRight)
        self.alpha2.setText('NA')
        self.alpha2.setEnabled(False)
        self.alpha2.setFixedSize(150,20)
        #add to the grid
        self.modelTuningLayout.addWidget(self.tune_model_label, 0, 0)
        self.modelTuningLayout.addWidget(self.alpha1Label, 1, 0)
        self.modelTuningLayout.addWidget(self.alpha1, 1, 1)
        self.modelTuningLayout.addWidget(self.alpha2Label, 2, 0)
        self.modelTuningLayout.addWidget(self.alpha2, 2, 1)
        #add model tuning layout to the general layout
        self.modelTuningLayout.setAlignment(Qt.AlignRight)
        self.generalLayout.addLayout(self.modelTuningLayout, 1, 1)
        
    # this creates data selection GUI    
    def _createDataSelection(self):
        """Create data selection panel """
        #prepare grid layout
        self.dataLayout = QGridLayout()
        #create data selection label
        self.dataSelectionLabel = QLabel('<h3> Select data source <\h3>')
        #create data source selection box (drop down menu)
        self.datasource_box = QComboBox()
        self.datasource_box.addItem("Generate data")
        self.datasource_box.addItem("Load CSV")
        self.datasource_box.setToolTip("Load CSV file or generate data in app")
        #add data button. It is used to get new app-generated data (Refresh), or
        #to load data from other sources via Dialogs.
        self.data_button = QPushButton("Refresh")
        self.data_button.setToolTip("Generate new data.\nUse controls below to specify number of predictive features,\
                                    \nnumber of redundant (non-predictive) features\nand number of records.")
        #self.refresh_button.setFixedSize(50,40)
        #add labels and inputs that allow choice of number of records and features of auto-generated data
        self.features_label = QLabel('Number of predictive features:')        
        self.nonpredictive_label = QLabel('Number of redundant features:')    
        self.records_label = QLabel('Number of records:')
        self.features_input = QLineEdit()
        self.features_input.setAlignment(Qt.AlignRight)
        self.nonpredictive_input = QLineEdit()
        self.nonpredictive_input.setAlignment(Qt.AlignRight)
        self.records_input = QLineEdit()
        self.records_input.setAlignment(Qt.AlignRight)
        #add elements to the grid
        self.dataLayout.addWidget(self.dataSelectionLabel,0,0,1,2)
        self.dataLayout.addWidget(self.datasource_box, 1, 0)
        self.dataLayout.addWidget(self.data_button, 2, 0)
        self.dataLayout.addWidget(self.features_label, 3, 0, 1, 2)
        self.dataLayout.addWidget(self.features_input, 3, 2)
        self.dataLayout.addWidget(self.nonpredictive_label, 4, 0, 1, 2)
        self.dataLayout.addWidget(self.nonpredictive_input, 4, 2)
        self.dataLayout.addWidget(self.records_label, 5, 0, 1, 2)
        self.dataLayout.addWidget(self.records_input, 5, 2)
        #add data selection layout to the general layout
        self.generalLayout.addLayout(self.dataLayout, 0, 0)
        
    # this creates model report GUI    
    def _createReport(self):
        """Create model report panel """
        #prepare grid layout
        self.reportLayout = QGridLayout()
        #create dashboard elements
        #add section title
        self.dashboard_title = QLabel('<h3> Model scoring and evaluation <\h3>')
        #various labels
        self.R2_train_label = QLabel('R^2 score on train set:')
        self.R2_test_label = QLabel('R^2 score on test set:')
        self.true_coeffs_label = QLabel('True coefficients are: [0, 0, 0]')
        self.est_coeffs_label = QLabel('Estimated coefficients:')
        #indicators
        self.R2_train_display = QLineEdit()
        self.R2_train_display.setFixedSize(150,20)
        self.R2_train_display.setReadOnly(True)
        self.R2_train_display.setAlignment(Qt.AlignRight)
        self.R2_train_display.setText('NA')
        self.R2_test_display = QLineEdit()
        self.R2_test_display.setFixedSize(150,20)
        self.R2_test_display.setReadOnly(True)
        self.R2_test_display.setAlignment(Qt.AlignRight)
        self.R2_test_display.setText('NA')
        self.est_coeffs_display = QLineEdit()
        #self.est_coeffs_display.setFixedSize(150,20)
        self.est_coeffs_display.setReadOnly(True)
        self.est_coeffs_display.setAlignment(Qt.AlignRight)
        self.est_coeffs_display.setText('NA')
        #add dashboard elements to the remaining rows of the grid
        self.reportLayout.addWidget(self.dashboard_title,0,0)
        self.reportLayout.addWidget(self.R2_train_label, 1, 0)
        self.reportLayout.addWidget(self.R2_train_display, 1, 1)
        self.reportLayout.addWidget(self.R2_test_label, 2, 0)
        self.reportLayout.addWidget(self.R2_test_display, 2, 1)
        self.reportLayout.addWidget(self.true_coeffs_label, 3, 0)
        self.reportLayout.addWidget(self.est_coeffs_label, 4, 0)
        self.reportLayout.addWidget(self.est_coeffs_display, 5, 0, 1, 2)
        self.reportLayout.setAlignment(Qt.AlignRight)
        #add controls to the general layout
        self.generalLayout.addLayout(self.reportLayout, 2, 1)
        
    def _createOptiLearn(self):
        #create OptiLearn layout
        self.optiLayout = QGridLayout()
        self.magicLabel = QLabel("<h3>Make the magic happen:<\h3>")
        self.magicButton = QPushButton("OptiLearn")
        #add stuff to the layout
        self.optiLayout.addWidget(self.magicLabel, 0,0)
        self.optiLayout.addWidget(self.magicButton, 0, 1)
        #add OptiLearn to the general layout
        self.generalLayout.addLayout(self.optiLayout, 3, 1)
        #
        #optiLearn results will be published via QDialog
        self.optiMessage = QDialog()
        self.optiMessage.setWindowTitle("optiLearn results")
        self.optiMessageLayout = QGridLayout() #Dialog will have grid layout
        #prepare widgets to be added to the dialog
        self.optiMessageTitle = QLabel("<b>You have run automated machine learning feature.<\b>\nDetailed results:")
        self.optiMessageR2Label = QLabel("Model R^2 on test set:")
        self.optiMessageR2 = QLineEdit("NA")
        self.optiMessageR2.setReadOnly(True)
        self.optiMessageBICLabel = QLabel("Model BIC on test set:")
        self.optiMessageBIC = QLineEdit("NA")
        self.optiMessageBIC.setReadOnly(True)
        self.optiMessageSelectedFeaturesLabel = QLabel("Selected predictive features (indices):")
        self.optiMessageSelectedFeatures = QLineEdit("NA")
        self.optiMessageSelectedFeatures.setReadOnly(True)
        self.optiMessageParameterAlphaLabel = QLabel("Optimal value of parameter alpha:")
        self.optiMessageParameterAlpha = QLineEdit("NA")
        self.optiMessageParameterAlpha.setReadOnly(True)
        self.optiMessageParameterL1Label = QLabel("Optimal value of parameter L1-ratio:")
        self.optiMessageParameterL1 = QLineEdit("NA")
        self.optiMessageParameterL1.setReadOnly(True)
        self.optiMessageCoeffsLabel = QLabel("Learned coefficients")
        self.optiMessageCoeffs = QLineEdit("NA")
        self.optiMessageCoeffs.setReadOnly(True)
        #OK button to close the dialog
        self.optiMessageExit = QPushButton("OK")
        self.optiMessageExit.clicked.connect(self.optiMessage.reject)
        #add widgets to the dialog
        self.optiMessageLayout.addWidget(self.optiMessageTitle, 0,0)
        self.optiMessageLayout.addWidget(self.optiMessageR2Label, 1,0)
        self.optiMessageLayout.addWidget(self.optiMessageR2, 1,1)
        self.optiMessageLayout.addWidget(self.optiMessageBICLabel, 2,0)
        self.optiMessageLayout.addWidget(self.optiMessageBIC, 2,1)
        self.optiMessageLayout.addWidget(self.optiMessageSelectedFeaturesLabel, 3,0)
        self.optiMessageLayout.addWidget(self.optiMessageSelectedFeatures, 3,1)
        self.optiMessageLayout.addWidget(self.optiMessageParameterAlphaLabel, 4,0)
        self.optiMessageLayout.addWidget(self.optiMessageParameterAlpha, 4,1)
        self.optiMessageLayout.addWidget(self.optiMessageParameterL1Label, 5,0)
        self.optiMessageLayout.addWidget(self.optiMessageParameterL1, 5,1)
        self.optiMessageLayout.addWidget(self.optiMessageCoeffsLabel, 6,0)
        self.optiMessageLayout.addWidget(self.optiMessageCoeffs, 6,1)
        self.optiMessageLayout.addWidget(self.optiMessageExit, 7,1)
        #add everything to the dialog layout
        self.optiMessage.setLayout(self.optiMessageLayout)
        
     
        
#controller code
class Controller:
    """app controller class"""
    def __init__(self,view, data_model, ML_model):
        #initialize
        self._datasrc = data_model #this is the model that creates new data
        self._view = view #this is the main GUI object
        self._MLmodel = ML_model # this is machine learning model object
        #connect signals and slots
        self._connectSignals()
    
    def _updateData(self,datasource):
        #number of requested rows and columns should be an integer
        try:
            num_records = int(self._view.records_input.text())
            num_features = int(self._view.features_input.text())
            num_nonpred = int(self._view.nonpredictive_input.text())
        except:
            num_records = 100
            num_features = 2
            num_nonpred = 0
        #should be positive numbers
        if num_records>0 and num_features>0 and num_nonpred>=0:
            #datasource function returns the new dataset (columns are features followed by target variable), and 
            #the linear coefficients used to combine predictive features into the target
            newdata, linear_coeffs = datasource(num_records,num_features, num_nonpred)
        else:
            newdata, linear_coeffs = datasource(100, 2, 0)
            print("Number of records and features should be positive integers.\nGoing with default choice.")
        #update data displayed in the table
        #prepare display size of the table
        self._view.table = TableView(newdata, newdata.shape[0], newdata.shape[1])
        #add table to the layout
        self._view.generalLayout.addWidget(self._view.table, 1, 0, 2, 1)
        #update data
        self._view.table.updateData(newdata=newdata)
        #update displayed table headers
        try:
            header = ['p'+str(i+1) for i in range(num_features)]
            header.extend( ['r'+str(i+1) for i in range(num_nonpred)] )
            header.extend( ['target'] )
        except:
            header = ['x'+str(i+1) for i in range(len(newdata[0]))]
            header[-1] = 'target'
        self._view.table.setHorizontalHeaderLabels(header)
        #update data available to the ML model
        self._MLmodel.updateData(newdata=newdata)
        #show new coefficients that should be learned:
        coeffs_string = "{"
        for c in linear_coeffs:
            coeffs_string += " "+str(c)+","
        coeffs_string = coeffs_string[:-1] + "}"
        self._view.true_coeffs_label.setText(f'True coefficients are: <b>{coeffs_string}<\b>')
        
    def _loadCSV(self,):
        #open file selection dialog. restrict dialog to csv files only
        file_name, _ = QFileDialog().getOpenFileName(self._view, "Window name", "", "CSV files (*.csv)")
        #read data into a pandas dataframe (easiest approach)
        df = pd.read_csv(file_name, sep= ";", index_col=0)
        #transform data to numpy array and prepare new headers
        newdata = np.array(df)
        headers = list(df.columns)
        #update data and headers
        #prepare display size of the table
        self._view.table = TableView(newdata, newdata.shape[0], newdata.shape[1])
        #add table to the layout
        self._view.generalLayout.addWidget(self._view.table, 1, 0, 2, 1)
        self._view.table.updateData(newdata=newdata)
        self._view.table.setHorizontalHeaderLabels(headers)
        #update data available to the ML model
        self._MLmodel.updateData(newdata=newdata)
        
    def _selectDataSource(self):
        # depending on the datasource_box selection, data_button should do one of the following:
        # it should be a "refresh button" for app-generated data
        # it should call on a select-data dialog
        if self._view.datasource_box.currentText() == "Generate data":
            self._view.data_button.setText("Refresh")
            #remove all old connections
            self._view.data_button.disconnect() 
            #reconnect data_button with updateData function
            self._view.data_button.clicked.connect(partial(self._updateData,provideData))
            #return all graphical elements related to choice of number of records and features
            self._view.features_input.setEnabled(True)
            self._view.nonpredictive_input.setEnabled(True)
            self._view.records_input.setEnabled(True)
            #set tooltip for the data button
            self._view.data_button.setToolTip("Generate new data.\nUse controls below to specify number of predictive features,\
                            \nnumber of redundant (non-predictive) features\nand number of records.")
        elif self._view.datasource_box.currentText() == "Load CSV":
            self._view.data_button.setText("Select data")
            self._view.data_button.disconnect() #remove all old connections
            self._view.data_button.clicked.connect(partial(self._loadCSV))
            #remove graphical elements related to auto-generation of data
            self._view.features_input.setEnabled(False)
            self._view.nonpredictive_input.setEnabled(False)
            self._view.records_input.setEnabled(False)
            #set tooltip for the data button
            self._view.data_button.setToolTip('Load dataset from hard disk.\nOnly ".csv" files are supported.')

    
    def _trainModel(self):
        # call on machine learning class to train selected model. 
        # step 1/3: perform train-test split
        test_ratio = self._view.test_ratio_selection.text()
        self._MLmodel.splitTrainTest(test_ratio)
        # step 2/3: train model (identify model by label in the model selection box, and pass two arguments)
        model = self._view.model_box.currentText()
        alpha1 = self._view.alpha1.text()
        alpha2 = self._view.alpha2.text()
        self._MLmodel.trainModel(model, alpha1, alpha2)
        # step 3/3: get all model quality metrics and estimated parameters
        R2train, R2test, coeffs = self._MLmodel.modelQuality()
        #prepare a string of estimated coefficients to output on est_coeffs_display
        coeff_str = "{"
        for c in coeffs:
            coeff_str += " "+f"{c:.3f}"+","
        coeff_str = coeff_str[:-1] + "}"
        self._view.R2_train_display.setText(f"{R2train:.3f}")
        self._view.R2_test_display.setText(f"{R2test:.3f}")
        self._view.est_coeffs_display.setText(f"{coeff_str}")
        
    def _setTunableParameters(self):
        # selected_model is the model currently selected in the drop-down menu
        selected_model=self._view.model_box.currentText()
        #depending on the chosen model, each of the two alpha parameters may or may not be tuned
        if selected_model == models_list[0]:
            #linear regression
            self._view.alpha1.setText('NA')
            self._view.alpha1.setEnabled(False)
            self._view.alpha2.setText('NA')
            self._view.alpha2.setEnabled(False)
        elif selected_model == models_list[1]:
            #Ridge
            self._view.alpha1.setText('NA')
            self._view.alpha1.setEnabled(False)
            self._view.alpha2.setText('.5')
            self._view.alpha2.setEnabled(True)
        elif selected_model == models_list[2]:
            #Lasso
            self._view.alpha1.setText('1.0')
            self._view.alpha1.setEnabled(True)
            self._view.alpha2.setText('NA')
            self._view.alpha2.setEnabled(False)
        elif selected_model == models_list[3]:
            #ElasticNet
            self._view.alpha1.setText('1.0')
            self._view.alpha1.setEnabled(True)
            self._view.alpha2.setText('.5')
            self._view.alpha2.setEnabled(True)
            
    def _optiLearn(self):    
        try:
            # calculate optimal model and get all model parameters and scores
            R2, BIC, selected_features, alpha, l1_ratio, learned_coeffs = self._MLmodel.optiLearn()
            #prepare display of R^2, BIC, selected features, and model parameters
            self._view.optiMessageR2.setText(f"{R2:.3f}")
            self._view.optiMessageBIC.setText(f"{BIC:.3f}")
            #list of selected features by name
            feat_str = "{"
            for index in selected_features:
                feat_str += " "+self._view.table.horizontalHeaderItem(index).text()+","
            feat_str = feat_str[:-1] + "}"
            self._view.optiMessageSelectedFeatures.setText(feat_str)
            #self._view.optiMessageSelectedFeatures.setText(str(selected_features))
            self._view.optiMessageParameterAlpha.setText(f"{alpha:.4f}")
            self._view.optiMessageParameterL1.setText(f"{l1_ratio:.4f}")
            #prepare display of list of learned coefficients
            coeff_str = "{"
            for c in learned_coeffs:
                coeff_str += " "+f"{c:.3f}"+","
            coeff_str = coeff_str[:-1] + "}"
            self._view.optiMessageCoeffs.setText(coeff_str)
            self._view.optiMessage.exec_()
        except:
            print("something went wrong")
            #self._view.optiMessage.exec_()
        

    def _connectSignals(self):
        """ connect signals and slots """
        #make the main buttons and selection boxes do their magic
        self._view.datasource_box.activated.connect(self._selectDataSource)
        self._view.data_button.clicked.connect(partial(self._updateData,provideData))
        self._view.train_model_button.clicked.connect(self._trainModel)
        self._view.model_box.activated.connect(self._setTunableParameters)
        self._view.magicButton.clicked.connect(self._optiLearn)
        #when focus is on data input QLineEdits make return button click the "Refresh" button
        self._view.features_input.returnPressed.connect(self._view.data_button.click)
        self._view.nonpredictive_input.returnPressed.connect(self._view.data_button.click)
        self._view.records_input.returnPressed.connect(self._view.data_button.click)
        # set up main buttons so that they're clicked if return key is pressed
        self._view.data_button.setAutoDefault(True)
        self._view.train_model_button.setAutoDefault(True)
        self._view.magicButton.setAutoDefault(True)
 


def provideData(records, features, redundant):
    """
    generates random data with "records" number of rows, "features" number of predictive features, and "redundant" number of redundant (non-predictive) variables.
    it outputs: 1) a numpy array of dimensions records x (features + 1), where the last column is target variable, 
    and 2) the list of coefficients in the linear combination
    """
    #generate Gaussian variates with mean zero and fixed variance
    lst = np.random.randn(records,features)*20
    #target will be a linear combination of features with following coefficients
    coeffs = np.random.randint(-10,10,features)
    #generate redundant features (nonpredictive variables)
    lst_redundant = np.random.randn(records,redundant)*20
    # result will have both features and the target in each row
    result = []
    #generate result list
    for i in range(len(lst)):
        #start by putting a list of features in the new row
        newrow = list(lst[i])
        #append non-predictive features to the list of features
        newrow.extend(lst_redundant[i])
        #append the target to the list of features. target is a linear combo of predictive features plus a noise
        tgt = (lst[i]*coeffs).sum() + 20*np.random.randn()
        newrow.append( tgt )
        result.append(newrow)
    return np.array(result), coeffs



# this is the machine learning model
    
class MLmodel:
    def __init__(self,data):
        # save the dataset on which the model is to be trained
        self.dataset = data
        #set classifier, linear reg. coefficients, and R^2 scores to None
        self.model = None
        self.coeffs = None
        self.R2_train = None
        self.R2_test = None
        #set train and test data to None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        
        
    def updateData(self, newdata):
        # update dataset
        self.dataset = newdata
        # reset classifier, scores and coefficients
        self.model = None
        self.coeffs = None
        self.R2_train = None
        self.R2_test = None
        #reset train and test data
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
    
    def splitTrainTest(self, test_size='.1'):
        #clean up test_size:
        #remove "%" if user input it
        if test_size.endswith('%'):
            test_size = test_size[:-1]
        #transform input to number
        test_size = float(test_size)
        #if user input percentages, transform them to values between 0 and 1
        if test_size >=1 and test_size <100:
            test_size = test_size/100.0
        #prepare features and targets
        features = self.dataset[:,:-1]
        targets  = self.dataset[:,-1]
        #perform train-test-split
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(features, targets, test_size=test_size)
    
    def trainModel(self, selected_model, alpha1, alpha2):
        # prepare selected model
        if selected_model == models_list[0]:
            self.model = LinearRegression()
        elif selected_model == models_list[1]:
                self.model = Ridge(alpha=float(alpha2))
        elif selected_model == models_list[2]:
            self.model = Lasso(alpha=float(alpha1))
        elif selected_model == models_list[3]:
            elasticnetFactor = float(alpha1)+float(alpha2)
            self.model = ElasticNet(alpha = elasticnetFactor, l1_ratio = float(alpha1)/elasticnetFactor)
        # train model
        self.model.fit(self.train_X, self.train_y)
        #score model
        self.R2_train = self.model.score(self.train_X, self.train_y)
        self.R2_test = self.model.score(self.test_X, self.test_y)
        self.coeffs = self.model.coef_
        
    def modelQuality(self):
        return self.R2_train, self.R2_test, self.coeffs
    
    def getBIC(self, model, X, y):
        """
        Returns Bayesian information criterion of a linear model. Takes scikit-learn model, list of features and target list.
        """
        # get sum of squared errors (sum of squares of absolute values of differences between prediction and true value)
        SSE = np.sum(abs(model.predict(X)-y)**2)
        # get parameters needed to calculate AIC
        num_params = len(model.coef_) #number of parameters in the model
        nrows = X.shape[0]
        # calculate AIC
        BIC = nrows * (np.log(2*np.pi) + np.log(SSE) - np.log(nrows)) + nrows + np.log(nrows)*num_params
        return BIC
    
    def optiLearn(self):
        #get train, validation, and test set
        #prepare features and targets
        features = self.dataset[:,:-1]
        targets  = self.dataset[:,-1]
        #perform train-test-split to get train and test set
        #train set will be further split into train and validation sets in k-fold cross-validation loop below
        train_X, test_X, train_y, test_y = train_test_split(features, targets, test_size=.1)
        #prepare k-fold cross-validation
        num_splits = 5 #used to get average score across cross-validations
        kfXvalidation = KFold(n_splits=num_splits)
        # list of all features used by the model. for now it is all features, and will be reduced as features are dropped.
        current_columns = [i for i in range(train_X.shape[1])]
        current_X = train_X #this is the original training set, from which we drop columns as necessary
        # keep track of the best score (and the best model) for given set of features
        best_score = 10e6 #used to get the best model on a given set of features
        best_score_improved = True
        #enter feature-selection loop. feature dropout will be done in the end of the loop
        while best_score_improved is True:
            best_score_improved = False
            #enter model-selection loops. in the outer-most two loops we loop over model parameters
            #choose alpha parameter of ElasticNet regression. log-spaced values between 10^-3 and 1 will be attempted
            for alpha in np.logspace(-3,0,30):
                #choose l1_ratio parameter of ElasticNet regression. linearly-spaced values between 0.1  and 1 will be attempted
                for l1_ratio in np.linspace(0.0101, 1, 30): # lower limit has to be >0.01, see doc
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                    #run through k-fold cross-validation, produce train and validation sets, evaluate models
                    cross_score = 0
                    for train_index, valid_index in kfXvalidation.split(current_X):
                        Xtrain, Xvalid = current_X[train_index], current_X[valid_index]
                        ytrain, yvalid = train_y[train_index], train_y[valid_index]
                        model.fit(Xtrain, ytrain) #fit model on new subset
                        cross_score += self.getBIC(model,Xvalid,yvalid) #add up total model score
                    #get average model score on cross-validation
                    cross_score = cross_score/5
                    #if best score is improved by current model, save current model
                    #information criterion (AIC or BIC) should be minimized
                    if cross_score<best_score:
                        best_score = cross_score #update best score
                        best_model = model
                        chosen_features = current_columns
                        best_score_improved = True
            #at this point, the best model is known for the current dataset
            #if it improved on the best model from the previous dataset then continue removing features
            if best_score_improved:
                #now drop the variable with smallest coefficient and repeat on reduced dataset
                smallest_coeff_index = np.argmin(abs(best_model.coef_))
                current_columns = [i for i in range(current_X.shape[1]) if i != smallest_coeff_index]
                current_X = current_X[:, current_columns]
            else:
                break #no sense to keep on removing features, if removing the last one didn't improve results
        #print details about the best model
        #print(f"Best model is:\n\t{best_model}\n")
        #print(f"Model k-fold cross-validation score is: BIC={best_score:.3f}")
        #print(f"Selected features are: {chosen_features}")
        #print(f"Coefficients are: {[f'{c:.3f}' for c in best_model.coef_]}")
        #
        #only reduced set of variables is used for scoring on test set
        test_X = test_X[:, chosen_features]
        #get BIC and R-squared
        BIC = self.getBIC(best_model, test_X, test_y)
        R2 = best_model.score(test_X,test_y)
        #print test set results
        print(f"On test set, model score is: BIC={BIC:.3f} and R^2={R2:.3f}.")
        return R2, BIC, chosen_features, best_model.alpha, best_model.l1_ratio, best_model.coef_



#run the program
if __name__ == "__main__":
    
    #some initial data to show
    init_data = np.array([
          [0,0,0,0],
          [0,0,0,0],
          [0,0,0,0]
        ])
    #models included in the application
    models_list = ["Linear regression", "Ridge regression", "Lasso", "ElasticNet"]
    
    #run the GUI        
    app=QApplication(sys.argv)
    app.setStyle('Fusion')      #options: ['Windows', 'WindowsXP', 'WindowsVista', 'Fusion']
    window=appGUI(init_data)
    window.show()
    #start controller
    data_model = provideData
    ML_model = MLmodel(init_data)
    Controller(view=window, data_model=data_model, ML_model=ML_model)
    #execute
    sys.exit(app.exec_())
    
