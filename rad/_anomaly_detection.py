"""Robust Principal Component Analysis for Time Series"""

# @DLegorreta: d.legorreta.anguiano@gmail.com

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from statsmodels import robust
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted,check_array
from typing import Union

from ._rPCA import RPCA as rpca


def _check_representation(frequency:int,X:np.ndarray):
	"""
	Check if the frequency is integer and the X array is 1D

	Parameters
	----------
	
	frequency : int
	X         : numpy array,np.ndarray

	Returns

	X         : numpy array with float values  

	"""

	X=check_array(X,ensure_2d=False)

	if type(frequency)==float:
		raise TypeError("The frequency must be int, not float")

	if X.ndim>1:
		raise TypeError('A 1D array was expected, instead a 2D array'
	                     ' was obtained')
	return X
	
def _autodiff(X:np.ndarray):
	""" 
	Apply Test Dickey-Fuller Test in Time Serie

	Parameters
	----------
	X  :	numpy array 

	Returns
	-------
	X    : 	numpy array
	flag :  Bool

	"""
	# Dickey-Fuller Test
	adf=adfuller(X)

	if(adf[1]>0.05):
		# No Stationary
		flag=True
		X=np.nan_to_num(np.diff(X,prepend=0),nan=0)
		return X,flag
	else:
		#Stationary
		flag=False
		return X,flag

def _scale(scale:bool,X:Union[pd.Series,np.ndarray]):
	""" 
	Normalization of the Time Serie

	Parameters
	----------
	scale 	: bool
	X 		: pd.Serie or Numpy Array

	Returns
	-------
	global_mean : float, mean of the Time Serie
	global_sdt  : float, sdt of the Time Serie
	X           : numpy array, 1D Array
	"""
	if isinstance(X,pd.Series):
		X=X.values.copy()

	if(scale):
		global_mean=np.mean(X)
		global_sdt=np.std(X)
		X=(X-global_mean)/global_sdt
		return global_mean,global_sdt,X
	else:
		return 0.0,1.1,X 

def _reshape_pad(X:np.ndarray,frequency:int):
	"""
	Transform 1d array to 2d array. Check if the frequency is 
	divisor of the length of the vector, if not, then fill  in 
	the vector  on the left 

	Parameters
	----------

	X 	: 1d numpy array
	frequency : int

	Returns
	-------
	X_out: 2D numpy array 

	"""
	l=len(X)
	r=l%frequency
	if r==0:
		return X.reshape((frequency,-1),order='F')
	else:
		diff=frequency-r
		X_out=np.pad(array=X,pad_width=(diff,0),mode='constant',constant_values=0).reshape((frequency,-1),order='F')
		return X_out

def _decision_mad(x,mu,ma):
	"""
	 Outliers with mad
	"""
	if x!=0 and ((abs(x-ma)/mu)>1.4826):
		return 1
	else: return 0

def _mad_outlier(X:np.ndarray):
	"""
	Outliers with mad
	"""

	L=np.where(X==0,np.NAN,X).copy()
	L=L[~np.isnan(L)]
	L=np.abs(L)
	mad=robust.mad(L)
	median=np.nanmedian(L)
	Output=np.apply_along_axis(_decision_mad,axis=0,arr=X.reshape(1,-1),mu=mad,ma=median)
	return Output

class AnomalyDetection_RPCA(BaseEstimator):
	"""
	Detection of anomalies over Time series: time series anomaly 
	detection using Robust Principal Component Pursuit
    
	Robust Principal Component Pursuit is a matrix decomposition algorithm 
	that seeks to separate a matrix X into the sum of three parts 
	
	         X = L + S + E. 
	L is a low rank matrix representing a smooth X, S is a sparse matrix 
	containing corrupted data, and E is noise. 
	To convert a time series into the matrix X we take advantage of seasonality 
	so that each column represents one full period, for example for weekly 
	seasonality each row is a day of week and one column is one full week.
	
	While computing the low rank matrix L we take an SVD of X and soft threshold 
	the singular values.
	
	This approach allows us to dampen all anomalies across the board simultaneously 
	making the method robust to multiple anomalies. Most techniques such as time series 
	regression and moving averages are not robust when there are two or more anomalies present.
	
	Empirical tests show that identifying anomalies is easier if X is stationary.
	The Augmented Dickey Fuller Test is used to test for stationarity - if X is not 
	stationary then the time series is differenced before calling RPCP. While this 
	test is abstracted away from the user differencing can be forced by setting the 
	forcediff parameter.

	The thresholding values can be tuned for different applications, however we strongly recommend 
	using the defaults which were proposed by Zhou. For more details on the choice of 
	Lpenalty and Spenalty please refer to Zhou's 2010 paper on Stable 
	Principal Component Pursuit.
	
	Inspired by Surus Project Netflix:https://github.com/Netflix/Surus 
	"""
	def __init__(
		self, 
		frequency:int=7,
		autodiff:bool=True,
		forcediff:bool = False,
		scale:bool = True,
		Lpenalty:float=1.0,
		Spenalty:float=-1.0,
		verbose:bool=False):

		self.frequency=frequency
		self.autodiff=autodiff
		self.forcediff=forcediff
		self.scale=scale
		self.Lpenalty=Lpenalty
		self.Spenalty=Spenalty
		self.verbose=verbose
		self.usediff=False
		self.global_mean=0
		self.global_sdt=1

	def fit(self,X):
		"""
		Fit estimador

		Parameters

		X: 1d array-like

		Returns
		-------
		self: object
		      Fitted estimador
		"""
		self._fit(X)
		return self

	def _fit(self,X):
		"""
		TODO:Documetation
		"""


		if self.Spenalty==-1:
			self._Spenalty=1.4/sqrt(max(self.frequency,len(X)/self.frequency))

		if isinstance(X,pd.Series):
			X=X.values.copy()

		X=_check_representation(frequency=self.frequency,X=X)
				
		if self.forcediff==True:
			self.usediff=True
			X=np.nan_to_num(np.diff(X,prepend=0))
			
		elif self.autodiff == True: 
			X,flag=_autodiff(X)
			self.usediff=flag
		
		self.global_mean,self.global_sdt,X=_scale(self.scale,X)
		
		M=_reshape_pad(X=X,frequency=self.frequency)
		
		if(self.verbose):

			print("..........Start Process..........")
			print("Time Series, frequency=%d and Num Periods= %d." %(M.shape[0],M.shape[1]))
			self._L,self._S,self._E=rpca(M,Lpenalty=self.Lpenalty,Spenalty=self._Spenalty,verbose=self.verbose)
		else:
			self._L,self._S,self._E=rpca(M,Lpenalty=self.Lpenalty,Spenalty=self.Spenalty,verbose=self.verbose)

	def fit_transform(self,X):
		"""
		Dispatch to the rPCA Function
		"""
		self.fit(X)
		L_transform=(self._L.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		S_transform=(self._S.T.reshape((-1,1)).ravel()*self.global_sdt)
		E_transform=(self._E.T.reshape((-1,1)).ravel()*self.global_sdt)

		return L_transform,S_transform,E_transform

	def transform(self):
		"""
		Return the Matrix L, S and E as a 1d arrays.
		"""

		check_is_fitted(self,attributes=['_S','_L','_E'])

		L_transform=(self._L.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		S_transform=(self._S.T.reshape((-1,1)).ravel()*self.global_sdt)
		E_transform=(self._E.T.reshape((-1,1)).ravel()*self.global_sdt)

		return L_transform,S_transform,E_transform

	def get_outliers(self):
		"""
		Return the Outliers after the rPCA transformation
		"""
		check_is_fitted(self,attributes=['_S'])
		return np.abs(self._S.T.reshape((-1,1)).ravel()*self.global_sdt)
	
	def decision_function(self):
		"""
		Return the Outliers with label 0 or 1.
		0 : Normal Values
		1 : Outlier

		"""
		check_is_fitted(self,attributes=['_S'])
		S=self._S.T.reshape((-1,1)).copy()
		return _mad_outlier(X=S)

	def to_frame(self,X,add_mad=True):
		"""
		Return DataFrame withe the values of the matrices 
		
		X=L+S+E

		"""

		check_is_fitted(self,attributes=['_S','_L','_E'])

		X_len=len(X)
		L_len=len(self._L.T.reshape((-1,1)).ravel())
		
		length_diff=abs(L_len-X_len)
		if length_diff>0:
			X=np.pad(array=X,pad_width=(length_diff,0),mode='constant',constant_values=0)

		if self.usediff:
			X=np.nan_to_num(np.diff(X,prepend=0))

			
		L_transform=(self._L.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		S_transform=(self._S.T.reshape((-1,1)).ravel()*self.global_sdt)
		E_transform=(self._E.T.reshape((-1,1)).ravel()*self.global_sdt)

		Output=pd.DataFrame({'X_original':X,
				            'L_transform':L_transform,
							'S_transform':S_transform,
							'E_transform':E_transform})

		if add_mad:
			S=self._S.T.reshape((-1,1)).ravel()
			Output['Mad_Outliers']=_mad_outlier(S)
			return Output
		else:
		    return Output	

	def num_outliers(self):
		"""
		Number of Outliers
		"""
		check_is_fitted(self,attributes=['_S'])
		S=self._S.T.reshape((-1,1)).ravel()
		return sum(np.abs(S)>0)
	
	def plot(self,figsize=(10,6)):
		"""
		Plot of the Time Series after rPCA transformation

		Parameters
		----------

		figsize : Size of the plot

		Returns
		-------
		matplotlib plot
		"""
		check_is_fitted(self,attributes=['_S','_L','_E'])

		L_transform=(self._L.T.reshape((-1,1)).ravel()*self.global_sdt)+self.global_mean
		S_transform=(self._S.T.reshape((-1,1)).ravel()*self.global_sdt)
		E_transform=(self._E.T.reshape((-1,1)).ravel()*self.global_sdt)

		fig=plt.figure(facecolor='w',figsize=figsize)
		ax=fig.add_subplot(111)

		ax.plot(range(len(L_transform,)),L_transform+E_transform,label='Time Serie rPCA',ls='--',c='royalblue')
		ax.plot(range(len(L_transform)),np.abs(S_transform), c='red',label='Outliers')
		ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
		ax.legend()
		fig.tight_layout()
		return fig.show()

		
