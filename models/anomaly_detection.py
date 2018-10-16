#Anomaly Outliers over Time Series
#@DLegorreta
import pandas as pd 
import numpy as np 
from statsmodels.tsa.stattools import adfuller
from .rPCA import RPCA as rpca
from  .RobustDeepAutoencoder  import * 
import tensorflow as tf
from statsmodels import robust
from sklearn.preprocessing import MinMaxScaler

def check_representation(frequency,X):
	"""
	Validation lenght(X)/frequency

	"""
	if type(frequency)==float:
		raise ValueError(
                    "Expected int")
	if X.ndim>1:
		raise ValueError(
                    "Expected 1D array, got 2D array instead")
	if X.shape[0]%frequency!=0:
		raise ValueError(
                    "Expected 1D array with the length of this array should be divisible by frequency")
	else:
	    return X
	    
def autodiff(X):
	""" Apply Test Dickey-Fuller Test over Time Serie
	"""
	adf=adfuller(X)
	if(adf[1]>0.05):
		flag=True
		X=X.diff().fillna(0)
		return X,flag
	else:
		flag=False
		return X,flag

def scale(Val_scale,X):
	""" Normalize the Time Serie
	"""
	if(Val_scale):
		global_mean=np.mean(X)
		global_sdt=np.std(X)
		X=(X-global_mean)/global_sdt
		return global_mean,global_sdt,X
	else:
		global_mean=0
		global_sdt=1
		return global_mean,global_sdt,X 


def Mad_Outliers_DF(df,column='S_transform'):
    
    L=df.copy()
    mad=robust.mad(L[column].replace(0,np.nan).dropna().apply(np.abs))
    median=L[column].replace(0,np.nan).dropna().apply(np.abs).median(skipna=True)

    # Create an empty column for outlier info
    L['MAD_Outlier'] = None
    # Iterate over rows
    for idx, row in L.iterrows():
        # Update the 'Outlier' column with True if the wind speed is higher than our threshold value
        if (row[column]!=0) and (((np.abs(np.abs(row[column])- median))/mad) > 1.4826) :
            L.loc[idx, 'MAD_Outlier'] = 1
        else:
            L.loc[idx, 'MAD_Outlier'] = 0
    return L        


class AnomalyDetection_RPCA(object):
	""" 
	   Detection of anomalies over Time series: time series anomaly detection 
	                                  using Robust Principal Component Pursuit

        Inspired by Surus Project Netflix:https://github.com/Netflix/Surus

	                                  
	"""

	def __init__(self, frequency=7,autodiff=True,forcediff = False,scale = True,Lpenalty=-1,Spenalty=-1,verbose=False):
		self.frequency=frequency
		self.autodiff=autodiff
		self.forcediff=forcediff
		self.scale=scale
		self.Lpenalty=Lpenalty
		self.Spenalty=Spenalty
		self.verbose=verbose
		self.usediff=False
		
	def fit(self,X):
		
		X=check_representation(self.frequency,X)
		self.X=pd.Series(X)
		X_orig=X.copy()
		#if(self.autodiff==True and self.forcediff==True):
		#	raise ValueError(
        #        "Default apply autodiff, if you set forcediff== True " 
        #        "you will need autodiff=False") 
		
		if self.forcediff==True:
			X=X.diff().fillna(0)
			self.usediff=True
			#self.diff=True

		elif self.autodiff == True: 
			X,self.autodiff=autodiff(X)
			self.usediff=self.autodiff
			#self.diff=True
		
		self.global_mean,self.global_sdt,X=scale(self.scale,X)
		
		M=X.values.reshape((self.frequency,-1),order='F')
		

		if(self.verbose):

			print("..........Start Process..........")
			print("Time Series, frequency=%d and Num Periods= %d." %(M.shape[0],M.shape[1]))
			Xpca,L_matrix,S_matrix,E_matrix=rpca(M,Lpenalty=self.Lpenalty,Spenalty=self.Spenalty,verbose=self.verbose)
                          
		
		else:
			Xpca,L_matrix,S_matrix,E_matrix=rpca(M,Lpenalty=self.Lpenalty,Spenalty=self.Spenalty,verbose=self.verbose)

		
		#if(self.usediff==True):
		self.X_original=X_orig
		self.X_transform=(Xpca.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		self.L_transform=(L_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		self.S_transform=(S_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)
		self.E_transform=(E_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)
		#else:
		#	self.X_original=X_orig
		#	self.X_transform=(Xpca.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		#	self.L_transform=(L_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		#	self.S_transform=(S_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)
		#	self.E_transform=(E_matrix.T.reshape((-1,1)).ravel()*self.global_sdt)

		return self

	def to_frame(self,add_mad=True):
		Output=pd.DataFrame({'X_original':self.X_original,
				                 'X_transform':self.X_transform,
				                 'L_transform':self.L_transform,
				                 'S_transform':self.S_transform,
				                 'E_transform':self.E_transform})

		if add_mad:
			return Output.pipe(Mad_Outliers_DF)
		else:
		    return Output	

	def num_outliers(self):
		return sum(np.abs(self.S_transform)>0)

		
		          
def Function_RDAE(X, layers, lamda=2.2, learning_rate = 0.001, inner = 120, batch_size =12,outer=5,verbose=True):
	X=X.copy()
	with tf.Graph().as_default():
		with tf.Session() as sess:
			rdae = RDAE(sess = sess, lambda_= lamda, layers_sizes=layers,learning_rate=learning_rate)
			L, S= rdae.fit(X = X, sess = sess, inner_iteration = inner, iteration = outer, 
                                          batch_size = batch_size,verbose = verbose)
			M_Transf=rdae.transform(X=X,sess=sess)
			

			M_Recons = rdae.getRecon(X = X, sess = sess)
			
		return L, S,M_Transf,M_Recons,rdae.S		


Prep=MinMaxScaler()

class AnomalyDetection_AUTOENCODER(object):
    """
    Detection of anomalies over Time series : adaptation of the algorithm 
	                                  Anomaly Detection with Robust Deep Autoencoder
    Original Research: http://www.kdd.org/kdd2017/papers/view/anomaly-detection-with-robust-deep-auto-encoders
    """
    def __init__(self,frequency=7,autodiff=True,forcediff = False,scale = True,verbose=True,lamda=2.2,layers=[7,64,64,7],batch_size=12):
    	self.frequency=frequency
    	self.autodiff=autodiff
    	self.forcediff=forcediff
    	self.scale=scale
    	self.layers=layers
    	self.verbose=verbose
    	self.lamda=lamda
    	self.batch_size=batch_size
    	self.usediff=False

    def fit(self,X):

    	X=check_representation(self.frequency,X)
    	self.X=pd.Series(X)
    	X_orig=X.copy()

    	if self.forcediff==True:
    		X=X.diff().fillna(0)
    		self.usediff=True
    	
    	elif self.autodiff == True:
    		X,self.autodiff=autodiff(X)
    		self.usediff=self.autodiff

    	self.global_mean,self.global_sdt,X=scale(self.scale,X)
    	#Last stage
    	X=Prep.fit_transform(X.values.reshape(-1,1))

    	M=X.reshape((-1,self.frequency),order='C').copy()
    	print("Time Series, frequency=%d and Num Periods= %d." %(M.shape[1],M.shape[0]))

    	if(self.verbose):
    		print("..........Start Process..........")
    		L_matrix,S_matrix,M_trans,M_Recons,S_Outliers=Function_RDAE(M,layers=self.layers,lamda=self.lamda,verbose=self.verbose,batch_size=self.batch_size)
    	else:
    		L_matrix,S_matrix,M_trans,M_Recons,S_Outliers=Function_RDAE(M,layers=self.layers,lamda=self.lamda,verbose=self.verbose,batch_size=self.batch_size)

    	if (self.forcediff ==True or self.autodiff ==True):
    		if (self.scale==True):
    			M=Prep.inverse_transform(M.reshape((-1,1)))
    			L=Prep.inverse_transform(L_matrix.reshape((-1,1)))
    			M_trans=Prep.inverse_transform(M_trans.reshape((-1,1)))

    			self.X_original=X_orig
    			self.X_transform=(M.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.L_transform=(L.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.S_transform=(S_matrix.reshape((-1,1)).ravel())
    			self.M_trans=(M_trans.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.M_Recons=M_Recons.reshape((-1,1)).ravel()
    			self.S_Outliers=S_Outliers.reshape((-1,1)).ravel()

    		else:
    			M=Prep.inverse_transform(M.reshape((-1,1)))
    			L=Prep.inverse_transform(L_matrix.reshape((-1,1)))
    			M_trans=Prep.inverse_transform(M_trans.reshape((-1,1)))

    			self.X_original=X_orig
    			self.X_transform=M.reshape((-1,1)).ravel()
    			self.L_transform=L.reshape((-1,1)).ravel()
    			self.S_transform=(S_matrix.reshape((-1,1)).ravel())
    			self.M_trans=M_trans.reshape((-1,1)).ravel()
    			self.M_Recons=M_Recons.reshape((-1,1)).ravel()
    			self.S_Outliers=S_Outliers.reshape((-1,1)).ravel()
    			
    	else:
    		if (self.scale==True):
    			M=Prep.inverse_transform(M.reshape((-1,1)))
    			L=Prep.inverse_transform(L_matrix.reshape((-1,1)))
    			M_trans=Prep.inverse_transform(M_trans.reshape((-1,1)))

    			self.X_original=X_orig
    			self.X_transform=(M.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.L_transform=(L.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.S_transform=(S_matrix.reshape((-1,1)).ravel())
    			self.M_trans=(M_trans.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
    			self.M_Recons=M_Recons.reshape((-1,1)).ravel()
    			self.S_Outliers=S_Outliers.reshape((-1,1)).ravel()

    		else:
    			M=Prep.inverse_transform(M.reshape((-1,1)))
    			L=Prep.inverse_transform(L_matrix.reshape((-1,1)))
    			M_trans=Prep.inverse_transform(M_trans.reshape((-1,1)))

    			self.X_original=X_orig
    			self.X_transform=M.reshape((-1,1)).ravel()
    			self.L_transform=L.reshape((-1,1)).ravel()
    			self.S_transform=(S_matrix.reshape((-1,1)).ravel())
    			self.M_trans=M_trans.reshape((-1,1)).ravel()
    			self.M_Recons=M_Recons.reshape((-1,1)).ravel()
    			self.S_Outliers=S_Outliers.reshape((-1,1)).ravel()
    	return self

    def to_frame(self,add_mad=True):
    	Output=pd.DataFrame({'X_original':self.X_original,
				                 'X_transform':self.X_transform,
				                 'L_transform':self.L_transform,
				                 'S_transform':self.S_transform,
				                 'Trans_X':self.M_trans,
				                 'Recover_X':self.M_Recons,
				                 'S_Outliers':self.S_Outliers})
    	if add_mad:
    		Output['S_Outliers']=Output['S_Outliers'].apply(lambda x:np.log1p(x)).copy()
    		return Output.pipe(Mad_Outliers_DF,'S_Outliers')
    	else:
    		return Output
    def num_outliers(self):
    	return sum(np.abs(self.S_Outliers>0))
	

