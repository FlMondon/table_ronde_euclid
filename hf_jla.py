#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:34:59 2018

@author: florian
"""

import numpy as np
import cPickle as pkl
from iminuit import Minuit, describe, Struct
from matplotlib import pyplot as plt
from scipy.integrate import quad
import glob
import pyfits
import types
import emcee

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

H_zero = 59.4 #a changer
t_plus_BD = 5.2E18

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
    
    def intfun_BD(self, y_BD, t_plus_BD, t_zero_BD, r_BD, s_BD):
        """
        
        """
        a_BD = np.power((y_BD - (t_plus_BD/t_zero_BD))/(1. - (t_plus_BD/t_zero_BD)), r_BD+s_BD) * np.power(y_BD, r_BD-s_BD)
        return 1./a_BD
    
    def get_x_BD_from_zcmb(self, z, r, s, t_plus, t_zero):
        """
        
        """
        def mini_x_BD(x):
            alpha = t_plus/t_zero
            return np.power((x-alpha)/(1.-alpha), r+s) * np.power(x, r-s) - 1./(1.+z)
        m = Minuit(mini_x_BD, x=0.5, limit_x=(0.,1.), print_level=0, pedantic=False)
        m.migrad()
        return m.values['x']

    def dL_z(self, zcmb, zhel, omgM, omgL, w, omg_BD, pm_BD=-1):
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
                
        elif self.model==2018:
            H_zero_BD = H_zero/3.085678E19
            r_BD = (1 + omg_BD)/(4 + 3*omg_BD)
            s_BD = (np.sqrt(1 + (2/3)*omg_BD))/(4 + 3*omg_BD)
            t_zero_BD = ((H_zero_BD*t_plus_BD + 2*r_BD) + pm_BD*np.sqrt((H_zero_BD*t_plus_BD + 2*r_BD)**2 - 4*H_zero_BD*(r_BD-s_BD)*t_plus_BD))/(2*H_zero_BD)
            x_BD = self.get_x_BD_from_zcmb(zcmb, r_BD, s_BD, t_plus_BD, t_zero_BD)
            mu_zz = 5*np.log10((1+zcmb)*clight*t_zero_BD*(quad(self.intfun_BD,x_BD,1,args=(t_plus_BD, t_zero_BD, r_BD, s_BD))[0]))
                
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
    
    def Remove_Matrix(self, Tab,ID):
        	"""
        	function that remove from the 'Tab' matrix all the rows and colomns except those precised in 'ID'
        	"""
        	#create the list with the line to be removed
        	try:
        		tab = np.delete(np.arange(len(Tab[0])),ID,0)
        	except:
        		tab = np.delete(np.arange(len(Tab)),ID,0)
        		
        	#remove these lines to the original matrix
        	Tab = np.delete(Tab,tab,0)
        	try:
        		Tab = np.delete(Tab,tab,1)
        	except:
        		'''nothing else to do'''
        	return Tab
    
class JLA_Hubble_fit(CosmoTools):

    """
    """
    def __init__(self, path_params, cov_path, sigma_mu_path, model=0):
        """
        """
        self.path_params = path_params
        self.cov_path = cov_path
        self.sigma_mu_path = sigma_mu_path
        self.model = model
        self.cache_alpha = np.nan
        self.cache_beta = np.nan
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

    def muexp(self, alpha, beta, Mb, delta_M):
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
        		if self.M_stell[i]<10:
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

    def mu_cov(self, alpha, beta):
        """
        #Function that buil the covariance matrix as Betoule et al (2014) betzeen SNe to get the chi2
        imputs:
        -alpha: free parameter of the Hubble fit (factor of the stretch)
        -beta: free parameter of the Hubble fit (factor of the color)
        -IDJLA: is an array containing the indexing value of each SNe
        outuput:
        -Cmu : the covariance matrix (2 dimension array)
        cov_path: path to covariance matrix or array corespond directly to the covariance matrix itself
        sigma_mu_path : path to sigma_mu.txt
        """
        #Assemble the full covariance matrix of distance modulus, See Betoule et al. (2014), Eq. 11-13 for reference
        #You have to acces the data which are in '/data/software/jla_likelihood_v6/covmat/C*.fits'. The C_hosts.fits has been removed from the analysis  
        #Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('Data/covmat/C*.fits')])
    
        #test if the covariance matrix come from a file or it already a matrix
        if type(self.cov_path) is types.StringType:
            Ceta = sum([pyfits.getdata(mat) for mat in glob.glob(self.cov_path)])
            
            
    #        Ceta += sum([pyfits.getdata(mat) for mat in glob.glob('/data/software/jla_likelihood_v6/covmat/C_cal.fits')])
        else:
            Ceta = self.cov_path
        if type(self.sigma_mu_path) is types.StringType:
            sigma = np.loadtxt(self.sigma_mu_path)
        else:
            sigma= self.sigma_mu_path
    
        Cmu = np.zeros_like(Ceta[::3,::3])
        for i, coef1 in enumerate([1., alpha, -beta]):
            for j, coef2 in enumerate([1., alpha, -beta]):
                Cmu += (coef1 * coef2) * Ceta[i::3,j::3]
        # Add diagonal term from Eq. 13
    #    print(len(Cmu[:,0]))
        sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
        Cmu[np.diag_indices_from(Cmu)] += sigma_pecvel ** 2 + sigma[:, 0] ** 2 + sigma[:, 1] ** 2  
    #    Cmu=N.diag(Cmu[N.diag_indices_from(Cmu)])
        Cmu = self.Remove_Matrix(Cmu,self.IDJLA)
        return Cmu
    
    def chi2(self,omgM,omgL,w, omg_BD,alpha,beta,Mb,delta_M):
         ''' Funtion that calculate the chi2 '''
         result=0.
         dL = np.zeros(shape=(len(self.IDJLA)))
         if alpha!=self.cache_alpha or beta!=self.cache_beta:
             self.Mat = np.linalg.inv(self.mu_cov(alpha,beta))
             self.cache_alpha = alpha
             self.cache_beta = beta
         mu_z = self.muexp(alpha,beta,Mb,delta_M)

         #loop for matrix construction
         for i in range(len(self.zcmb)):
             zz = self.zcmb[i]           
             dL[i] = self.dL_z(zz,zz,omgM,omgL,w, omg_BD)

         #contruction of the chi2 by matrix product
         result =  np.dot( (mu_z-dL), np.dot((self.Mat),(mu_z-dL)))
         return result

    def Hubble_diagram(self):
        """
        Function which make the hubble fit of the given compilation.
        inputs :
        	-omgM: 1st free parameter to be fitted initialized to the 0.295 value if not precised
        	-alpha: 2nd free parameter to be fitted initialized to the 0.141 value if not precised
        	-beta: 3rd free parameter to be fitted initialized to the 3.101 value if not precised
        	-Mb: 4th free parameter to be fitted initialized to the -19.05 value if not precised
        	-delta_M: 5th free parameter to be fitted initialized to the -0.070 value if not precised
          -fixe: when fixe=0 omgL and w are fixe, fixe = 1 w is fixe, fixe =2 omgL is fixe else all parameters are fixe
        	-zcmb: an array which contains the redshifts of the SNs
        	-mB: B band peak magnitude of the SNs
        	-dmB: uncertainty on mB
        	-X1:  SALT2 shape parameter
        	-dX1: uncertainty on X1
        	-dC: uncertainty on C	
        	-C: colour parameter
        	-M_stell: the log_10 host stellar mass in units of solar mass
        	-IDJLA: index of the SNs from the 740 of jla used for the fit
        	-results : a file where some results wiil be written
        	- m : the iminuit object (see doc of iminuit for more information).
        	-ecarts5 : the residuals of the fit
        """
        #check : need to have at least 2 SNe
        '''f2,(test,P2,P3,P4,P5) = P.subplots(5, sharex=True, sharey=False, gridspec_kw=dict(height_ratios=[3,1,1,1,1]))'''
        
        
        #minimisation of the chi2
        '''
        the values of the free parameter can be initialized.
        '''
        self.read_JLA()
        if self.model == 0 :
            m = Minuit( self.chi2, omgM=0.2,omgL=np.nan,w=-1,alpha=0.141,beta=3.101,Mb=-19.05, delta_M=-0.070,limit_omgM=(0.2,0.4),limit_omgL=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=False,fix_omgL=True,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1, omg_BD=np.nan, fix_omg_BD=True)    
        elif self.model ==1 :
            m = Minuit(self.chi2,omgM=0.2,omgL=0.55,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,fix_omgM=False,fix_omgL=False,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1, omg_BD=np.nan, fix_omg_BD=True)    
        elif self.model ==2 :    
            m = Minuit(self.chi2,omgM=0.2,omgL=np.nan,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,fix_omgM=False,fix_omgL=True,fix_w=False, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1, omg_BD=np.nan, fix_omg_BD=True)    
        elif self.model == 2018:
            m = Minuit(self.chi2,omgM=np.nan,omgL=np.nan,w=np.nan,omg_BD=-1.49,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,fix_omgM=True,fix_omgL=True,fix_w=True,fix_omg_BD=False, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1, limit_omg_BD=(-1.49, -1.34)) 
        else :
            m = Minuit(self.chi2,omgM=0,omgL=0.5611,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0,0.4),limit_omgL=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=True,fix_omgL=True,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1, omg_BD=np.nan, fix_omg_BD=True)    
        
        m.migrad()
        return  m
#    def chi2_mcmc(self):
#        
#    def mcmc(self):
        
if __name__=="__main__":
        path_params = '../data_input/jla_lcparams.txt'
        cov_path = '../data_input/covmat/C*.fits'
        sigma_mu_path = '../data_input/covmat/sigma_mu.txt'
        hd_obj = JLA_Hubble_fit(path_params, cov_path, sigma_mu_path)
        hd_obj.model = 2018
        m_obj = hd_obj.Hubble_diagram()
        print m_obj.values
