#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:34:59 2018

@author: florian
"""

import numpy as np
import cPickle as pkl
from iminuit import Minuit, describe, Struct
from mathplotlib import pyplot as plt
from scipy.integrate import quad

###########
#constants
###########

clight=299792.458
H=0.000070
omgM=0.295
alpha=0.141
beta=3.101
Mb=-19.05
delta_M=-0.070
w=-1
omgL=np.nan

class utils(object):
    """
    """
    def __init__(self, X):
        """
        Parameters :
            X : distribution
        """
        self.X = X
        
    def savepkl(self,dic,name='nomame'):
    	'''
    	Function that create a pkl file to export some data
    	inputs:
    		-dic : a dictonary containing the data
    		-name : the name of the pkl file
    	'''
    	File = open('Results/pkl/' + name +'.pkl','w')
    	pkl.dump(dic,File)
    	File.close()
    		
    def Sx(self):
    	'''
    	Function that compute Sx of the distribution given in X
    	For the matematical description of Sx, check the pdf file : 'Data/Equations.pdf'
    	input : the distribution X
    	output : Sx(X)
    	'''
    	return np.sqrt(abs((1/(len(self.X)-1))*(sum((self.X-np.mean(self.X))**2))))
    		
    def RMS(self):
    	'''
    	Funtction that compute the RMS on a distribution
    	inputs :
    		-X the distribution
    	output :
    		-The RMS
    	'''
    	rms = np.sqrt(abs((1/len(self.X))*(sum((self.X-np.mean(self.X))**2))))
    	return rms
    	
    def RMSerr(self, X):
    	'''
    	Funtction that compute the error on the RMS on a distribution
    	inputs :
    		-X the distribution
    	output :
    		-The error of the RMS
    	'''
    	rmserr = self.Sx()/(np.sqrt(2*len(self.X)))
    	return rmserr
    	
    def MEANerr(self):
    	'''
    	Function that compute the error ont he mean of the distribution given in X
    	For the matematical description of Sx, check the pdf file : 'Data/Equations.pdf'
    	inputs :
    		-X the distibution
    	output:
    		-the error on the mean
    	'''
    	meanerr = self.Sx()* (1./np.sqrt(len(self.X)))
    	return meanerr
    	
    def Histo(self,std,stderr,P,orient='horizontal',xlabel='',ylabel=''):
    	'''
    	Function that plot the histogramm of the distribution given in x
    	imputs:
    	-x is the distrib itself (array)
    	-mean is the mean of the distrib (float)
    	-sigma_mean : the error on the average (float)
    	-std : the standard deviation (RMS) of the distribution (float)
    	-stderr : the errot on the RMS (float)
    	-P is the figure where the histogram will be plotted.
    	-xylabel and y label are the name ofthe axis.
    	'''
    	numBins = 20
    	plt.hist(self.X,numBins,color='blue',alpha=0.8,orientation=orient,label='average = ' + str("%.5f" % np.mean(self.X)) + '$\pm$' + str("%.5f" % self.MEANerr()) + '\n' +  'rms =' + str("%.5f" % self.rms()) + '$\pm$' +str("%.5f" % self.RMSerr()))
    	if xlabel == '':
    		plt.set_xlabel('number of SNe')
    	else:
    		plt.set_xlabel(xlabel)
    	if ylabel == '':
    		plt.set_ylabel('number of SNe')    
    
    	else:
    		plt.set_ylabel(ylabel)
    	plt.set_title('Residuals')
    	plt.legend(bbox_to_anchor=(0.95, 1.0),prop={'size':10})

class CosmoTools(object):
    """
    """
    def __init__(self, model):
        self.model = model
        
    def comp_rms(self, residuals, dof, err=True, variance=None):
    	"""                                                                                                                                                                                      
    	Compute the RMS or WRMS of a given distribution.
    	:param 1D-array residuals: the residuals of the fit.
    	:param int dof: the number of degree of freedom of the fit.                                                                                                                              
    	:param bool err: return the error on the RMS (WRMS) if set to True.                                                                                                                      
    	:param 1D-aray variance: variance of each point. If given,                                                                                                                               
            return the weighted RMS (WRMS).                                                                                                                                                                                                                                                                                                                             
    	:return: rms or rms, rms_err                                                                                                                                                             
    	"""
    	if variance is None:                # RMS                                                                                                                                                
    		rms = float(np.sqrt(np.sum(residuals**2)/dof))
    		rms_err = float(rms / np.sqrt(2*dof))
    	else:	  # Weighted RMS                                                                                                                                       
    		assert len(residuals) == len(variance)
    		rms = float(np.sqrt(np.sum((residuals**2)/variance) / np.sum(1./variance)))
    		#rms_err = float(N.sqrt(1./N.sum(1./variance)))                                                                                                                                      
    		rms_err = np.sqrt(2.*len(residuals)) / (2*np.sum(1./variance)*rms)
    	if err:
    		return rms, rms_err
    	else:
    	    return rms
    
    
    def intfun(self, z, omgM,omgL,w):
        """
        Function that build an array that contains theoretical mu (funtion of omgM and luminosity distance)
        imputs:
        -z: represent the redshift
        -omgM:represent the parameter omgM
        -omgL: represent the parameter omgL    
        -w:represent the parameter w
        """
        return 1/np.sqrt(omgM*(1+z)**3 + omgL*(1+z)**(3*(1+w)) + (1-omgM-omgL)*(1+z)**2)

    def dL_z(self, zcmb, zhel, omgM, omgL, w):
        """ 
        Function that compute the integral for the comoving distance.
        imputs:
            -zcmb is the redshift in the cmb framework (array)
            -zhel the redshift in the heliocentric framework (array)
            -omgM = 0.295
        outputs:
            -mu_zz is the array contaning the distance for each SNe
        """
        if self.model==1 or self.model==4:
            omgK=1-omgM-omgL
            w=-1
            if omgK == 0:
                mu_zz = 5*np.log10((1+zcmb)*clight*(quad(self.intfun,0,zcmb,args=(omgM,omgL,w))[0]/(10*H)))        
            elif omgK < 0 :
                mu_zz = 5*np.log10((1+zcmb)*(1/np.sqrt(np.abs(omgK)) *clight*(np.sin(np.sqrt(np.abs(omgK))*quad(self.intfun,0,zcmb,args=(omgM,omgL,w))[0])/(10*H))))
            elif omgK > 0 :
                mu_zz = 5*np.log10((1+zcmb)*(1/np.sqrt(omgK)) *clight*(np.sinh(np.sqrt(omgK)*quad(self.intfun,0,zcmb,args=(omgM,omgL,w))[0])/(10*H)))
                
        else:
            omgL=1-omgM
            mu_zz = 5*np.log10((1+zcmb)*clight*(quad(self.intfun,0,zcmb,args=(omgM,omgL,w))[0]/(10*H)))        
          
        return mu_zz    
        
    def fitfundL(self, zcmb, omgM, omgL, w):
    	"""
    	Function which create the distance modulus for each supernovae
    we replace zhel by zcmb because mb are in restframe (correction did for peculiar velocity)
    	imputs:
    	-zcmb is the redshift of each SNe (array)
    	-omgM = 0.295.
    	outputs:
    	-MU is the array containing the distance modulus of each SNe
    	"""
    	MU=[]
    	for zz in zcmb: 
    		MU.append(self.dL_z(zz,zz,omgM,omgL,w)) 
    	return MU  
    

    
class JLA_Hubble_fit(CosmoTools):

    """
    """
    def __init__(self, path_params, path_cov):
        """
        """
        
        self.path_params = path_params
        self.path_cov = path_cov

    def read_JLA(self):
        """
        read JLA data (cf Marc Betoule 2014)
        """
            #First file is JLA data itself
        jlaData=np.loadtxt(self.path_params, dtype='str')
        
        #Select the chosen data
        self.SnIDjla = np.array(jlaData[:,0])
        self.zcmb = np.array(jlaData[:,1],float)
        self.zhel = np.array(jlaData[:,2],float)
        self.mB = np.array(jlaData[:,4],float)
        self.dmB = np.array(jlaData[:,5],float)
        self.X1 = np.array(jlaData[:,6],float)
        self.dX1 = np.array(jlaData[:,7],float)
        self.C = np.array(jlaData[:,8],float)
        self.dC = np.array(jlaData[:,9],float)
        self.M_stell = np.array(jlaData[:,10],float)
        self.Rajla = np.array(jlaData[:,18],float)
        self.Decjla = np.array(jlaData[:,19],float)
        self.subsample = np.array(jlaData[:,17],int)
        self.cov_ms = np.array(jlaData[:,14],float)
        self.cov_mc = np.array(jlaData[:,15],float)
        self.cov_sc = np.array(jlaData[:,16],float)   
        self.IDJLA = np.arange(740)
    
    def muexp(self, alpha, beta, Mb, delta_M, M_stell):
        	"""
        	#Correction to muexp regarding the stellar mass of the host galaxy (cf Betoule et al 2014)
        	imputs:
        	-alpha: free parameter of the Hubble fit (factor of the stretch)
        	-beta: free parameter of the Hubble fit (factor of the color)
        	-delta_M is a free parameter of the Hubble fit (value of the step for the mass step correction (see Betoule et al 2014))
        	-M_stell: the log_10 host stellar mass in units of solar mass
        	-mB: B band peak magnitude of the SNs (array)
        	-X1:  SALT2 shape parameter (array)
        	-C: colour parameter (array)
        	-M_stell : the stellar ;aass of each host (array)
        	"""
        	mu=[]
        	#As M.Betoule (choose one or the other)
        	for i in range(len(self.mB)):
        		if M_stell[i]<10:
        			mu.append(self.mB[i]-Mb+alpha*self.X1[i]-beta*self.C[i])
        		else :
        			mu.append(self.mB[i]-Mb-delta_M+alpha*self.X1[i]-beta*self.C[i])
        	#With fixed delta_M (choose one or the other)
        	#Without mass step correction
        	'''
        	for i in range(len(mB)):
        		mu.append(mB[i]-Mb+alpha*X1[i]-beta*C[i])
        	'''
        	return mu
    
    def dmuexp(self, alpha,beta):
        	"""
        	Function that build the list of dmuexp (uncertainties propagation) 
        	imputs:
        	-dmB: uncertainty on mB
        	-dX1: uncertainty on X1
        	-dC: uncertainty on C	
        	-alpha: free parameter of the Hubble fit (factor of the stretch)
        	-beta: free parameter of the Hubble fit (factor of the color)
        	"""
        	dmu=[]
        	for i in range(len(self.dmB)):
        	        dmu.append(np.sqrt(self.dmB[i]**2+(alpha*self.dX1[i])**2+(beta*self.dC[i])**2))
        	return dmu    
     
    def Cstat(self):
        """
        function that make a statistical covariance matrix with the covariance term between the fit parameters
        this matrix is diag by bloc
        input:
        -IDJLA: ref of the SN
        -cov-ms:array contains all covariance term between Mb and X1 by SN
        -cov-mc:array contains all covariance term between Mb and C by SN
        -cov-sc:array contains all covariance term between C and X1 by SN
        """
        
        Cstat= np.zeros([len(self.dm)*3,len(self.dm)*3])
        
        for i in range(len(self.dm)):
            Cstat[i*3,i*3]= self.dm[i]**2
            Cstat[i*3,i*3 +1]= self.cov_ms[i]
            Cstat[i*3,i*3 +2]= self.cov_mc[i]
            Cstat[i*3 +1,i*3]= self.cov_ms[i]
            Cstat[i*3 +2,i*3 ]= self.cov_mc[i]
            
            Cstat[i*3 +1,i*3 +1]= self.dX1[i]**2
            Cstat[i*3 +1,i*3 +2]= self.cov_sc[i]
            Cstat[i*3 +2,i*3 +1]= self.cov_sc[i]
            Cstat[i*3+2,i*3 +2]= self.dC[i]**2         
        return Cstat


	def chi2(self,omgM,omgL,w,alpha,beta,Mb,delta_M):
         ''' Funtion that calculate the chi2 '''
         result=0.
         if alpha!=self.cache_alpha or beta!=self.cache_beta:
             self.Mat = inv(mu_cov(alpha,beta,self.IDJLA, self.cov_path, self.sigma_mu_path))
             self.cache_alpha=alpha
             self.cache_beta=beta
         mu_z=muexp(self.mB,self.X1,self.C,alpha,beta,Mb,delta_M,self.M_stell)

         #loop for matrix construction
         for i in range(len(self.zcmb)):
             zz = self.zcmb[i]
             zzz = self.zhel[i]
           
             self.dL[i] = self.dL_z(self.fixe,zz,zzz,omgM,omgL,w)

         #contruction of the chi2 by matrix product
         result =  np.dot( (mu_z-self.dL), np.dot((self.Mat),(mu_z-self.dL)))
         self.chi2tot = result
         return result