#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:06:42 2023

@author: aparnaarun
"""

import pandas as pd #dataframe for handling data
import numpy as np #mathematical operation matrix
from pandas import ExcelWriter #operate on excel file
from pandas import ExcelFile
import re #used for pattern matching
from sklearn.metrics import confusion_matrix 
# from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 
import matplotlib.pyplot as plt
#importing wx files
import wx
import matplotlib.pyplot as plt1
#import the newly created GUI file

from sklearn import svm
import webbrowser
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn import preprocessing



################### FEATURE USER INPUT############################################

def user_input_extract(url):
    
    
    #length of url#lgth of URL flag
    url_length=len(url)
    if(url_length > 35):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if url.startswith("http://") or url.startswith("https://"):
        if (("http://" in url) or ("https://" in url)):
            http_has = 1
        else:
            http_has = 0
    else:
        print("ERROR")

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
       suffix_prefix = 1
    else:
       suffix_prefix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    
    return length_of_url,http_has,suspicious_char,suffix_prefix,dots,slash,phis_term,sub_domain,ip_contain
#################################################################################

########################## FEATURE TEST##########################################
#extract testing feature
def test_url_extract(url,output):
    
    
    #length of url
    url_length=len(url)
    if(url_length > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
       suffix_prefix = 1
    else:
       suffix_prefix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,suffix_prefix,dots,slash,phis_term,sub_domain,ip_contain
##############################################################################################################
################################ FEATURE TRAIN################################################################

#extract training feature
def train_url_extract(url,output):
    
    
    #length of url
    url_length=len(url)
    if(url_length > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
       suffix_prefix = 1
    else:
       suffix_prefix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output

   
        
    return yn,length_of_url,http_has,suspicious_char,suffix_prefix,dots,slash,phis_term,sub_domain,ip_contain

############################## import train data ############################################

def importdata_train(): 
    balance_data = pd.read_csv('train_data.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataswet shape 
    print ("Train Dataset Lenght: ", len(balance_data)) 
  
    return balance_data 
############################################################################################
######################### import test data #################################################

def importdata_test(): 
    balance_data = pd.read_csv('test_data.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
       
    # Printing the dataswet shape 
    print ("Test Dataset Length: ", len(balance_data)) 
 
    return balance_data 

######################### split data into train and test ######################################
def splitdataset(balance_data): 
  
    
    X = balance_data.values[:, 1:10]
    Y = balance_data.values[:, 0] 
  
  
      
    return X, Y

################################################################################################


############################## main funcation ###################################################

def main():
    excel_file= 'trainingurls.xlsx'
    train_data=pd.DataFrame(pd.read_excel(excel_file))
    excel_file_test= 'testurls.xlsx'
    test_data=pd.DataFrame(pd.read_excel(excel_file_test))

    a=[]
    b=[]
    
    a1=[]
    b1=[]
    
    #traing
    for train_url in train_data['url']:
        a.append(train_url)

    for output in train_data['phishing']:
        b.append(output)
        
#testing
    for test_url in test_data['url']:
        a1.append(test_url)
     
    for output in test_data['result']:
        b1.append(output)
  
    c=[]
    d=[]
    
    #to cmbine url with its results
    for url1,output1 in zip(a,b):       
        url=url1
        output=output1
        c.append(train_url_extract(url,output))
      
    for url1,output1 in zip(a1,b1):           
        url=url1
        output=output1
        d.append(test_url_extract(url,output))



    df=pd.DataFrame(c,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df.to_csv('train_data.csv', sep=',', encoding='utf-8')

    df_test=pd.DataFrame(d,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df_test.to_csv('test_data.csv', sep=',', encoding='utf-8')  
    
    data_train=importdata_train()

    data_test=importdata_test()

    X, Y = splitdataset(data_train)
    X1, Y1 = splitdataset(data_test)

    Y=np.where(Y=='yes','1', Y) 
    Y=np.where(Y=='no','0', Y) 
    Y1=np.where(Y1=='yes','1', Y1) 
    Y1=np.where(Y1=='no','0', Y1) 



    
    model=RandomForestClassifier()
    model.fit(X,Y)
    def RF_Model(X,Y,X1,Y1):
     	global acc1
     	print("___________________________Random Forest__________________________________________") 
     	#object 
     	model1=RandomForestClassifier()
     	model1.fit(X,Y)
         #prediction on test data using trainset
     	y_pred1 = model1.predict(X1)
     	
    


    class MainFrame ( wx.Frame ):
	
    	def __init__( self, parent ):
    		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = "Phishing URL Prediction", pos = wx.DefaultPosition, size = wx.Size( 500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
    		panel = wx.Panel(self)
    		
    		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

    		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
    		bSizer3 = wx.BoxSizer( wx.VERTICAL )
    
    		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"Enter Suspiciuos URL", wx.DefaultPosition, wx.DefaultSize, 0 )
    		
    		bSizer3.Add( self.m_staticText2, 0, wx.ALL, 5 )
                
          
		
    		self.text1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_PROCESS_ENTER )
    		self.SetBackgroundColour((238,106,80))
    		bSizer3.Add( self.text1, 0, wx.ALL|wx.EXPAND|wx.BOTTOM, 5)
            
    		
    
    		self.predictButton = wx.Button( self, wx.ID_ANY, u"Predict", wx.DefaultPosition, wx.DefaultSize, 0 )
    		self.predictButton.SetBackgroundColour((51, 105, 255))
    		bSizer3.Add( self.predictButton, 0, wx.ALL|wx.EXPAND, 5 )
                         # Get the box and then set the colour and font
            
            

		
    		self.SetSizer( bSizer3 )
    		self.Layout()
		
    		self.Centre( wx.BOTH )
		
    		# Connect Events
    		self.predictButton.Bind( wx.EVT_BUTTON, self.click )
    		self.text1.Bind(wx.EVT_TEXT_ENTER, self.click)

		
	
    	def __del__( self ):
    		pass
    	def OnCloseMe(self, event):
            RF_Model(X,Y,X1,Y1)
            self.Close(True)
            
    	def OnCloseWindow(self, event):
           
            RF_Model(X,Y,X1,Y1)
            self.Destroy()
	
    	# Virtual event handlers, overide them in your derived class



        #XGBOOST
    	def click( self, event ):
    	    try:
    	        url = self.text1.GetValue()
    	        e=np.array([user_input_extract(url)])
    	       
    	        userpredict1 = model.predict(e.reshape(1,-1)) 
    	        
    	        if(userpredict1[0]=='0'):
    	          
    	            print('Legitimate')
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"POP-UP" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"LEGITIMATE", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.SetBackgroundColour((0,100,0))
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )

    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	                	webbrowser.open(url)

    	            app3 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)

    	            app3.MainLoop()
    	            

    	        else:
    	            print('Phishing')
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 200,150), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Error" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"PHISHING", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.SetBackgroundColour((255,0,0))
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


    	
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )

    	                	def __del__( self ):
    	                		pass
	
	
	# Virtual event handlers, overide them in your derived class
    	                	def click( self, event ):
    	                		event.Skip()
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            app2.MainLoop() 

    	    except Exception:
    	        print ('Error,Invalid Input!')






    app1 = wx.App(False)
    frame = MainFrame(None)
    frame.Show(True)
    app1.MainLoop() 



   
 
    



  
if __name__== "__main__":
  main()





    
