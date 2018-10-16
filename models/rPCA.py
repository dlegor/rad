# Code Stable Principal Component Pursuit 
#@DLegorreta

import numpy as np 
import scipy as sc
from scipy import linalg
from statsmodels import robust
from numba import double
from numba.decorators import jit

@jit
def Dsoft(M,penalty):
    """ Inverts the singular values
    takes advantage of the fact that singular values are never negative

    Parameters
    ----------

    M : numpy array

    penalty: float number
             penalty scalar to penalize singular values 
    
    Returns
    -------

    M matrix with elements penalized when them viole this condition

    """
    penalty=float(penalty)
    #for x in np.nditer(M, op_flags=['readwrite']):
    #    penalized=x-penalty
    #    if(penalized<0):
    #        x[...]=0.0
    #    else:
    #        x[...]=penalized
    return np.maximum((M-penalty),0.0,out=None)    

@jit
def SVT(M,penalty):
    """
    Singular Value Thresholding on a numeric matrix

    Parameters
    ----------

    M : numpy array

    penalty: float number
             penalty scalar to penalize singular values 
    
    Returns
    -------

    S: numpy array 
       The singular value thresholded matrix  its 

    Ds: numpy array
       Thresholded singular values

    """
    penalty=float(penalty)
    U,s,V=np.linalg.svd(M,full_matrices=False)
    Ds=Dsoft(s,penalty)
    S=np.dot(np.multiply(U,np.diag(Ds)),V)
    #np.dot(U,np.dot(np.diag(Ds),V))
    return S,Ds

@jit
def SoftThresholdScalar(x,penalty):
    """sign(x) * pmax(abs(x) - penalty,0)
    """ 
   # np.sign(x)np.maximum(np.abs(x)-penalty,0)

    x=np.array(x).astype('float64')
    penalty=float(penalty)
    penalized=np.abs(x)-penalty
    if (penalized<0): return 0
    elif (x>0): return penalized
    else:
        return -penalized

@jit
def SoftThresholdVector(X,penalty):
    X=np.array(X).astype('float64')
    penalty=float(penalty)
    for x in np.nditer(X, op_flags=['readwrite']):
        x[...]=SoftThresholdScalar(x,penalty)
    return X

@jit
def SoftThresholdMatrix(X,penalty):
    X=np.array(X).astype('float64')
    penalty=float(penalty)
    return np.apply_along_axis(SoftThresholdVector, 1, X,penalty=penalty)

#@jit
#def SoftThresholdScalar2(X,penalty):
    """sign(x) * pmax(abs(x) - penalty,0)
    """ 
#    return np.multiply(np.sign(X),np.maximum(np.abs(X)-penalty,0.0,out=None))


@jit
def median(X):
    X=np.array(X).astype('float64')# No se usa esta funcion en el code, se encontraba en la fuente original de Netflix
    return np.median(X)

@jit
def mad(X):
    X=np.array(X).astype('float64')
    return robust.mad(X)

@jit
def getDynamicMu(X):
    X=np.array(X).astype('float64')
    m,n=X.shape
    E_sd=X.std()
    #mu=0.0
    #if(m>n):
    #    mu=E_sd*np.sqrt(2*m)
    #else:
    #    mu=E_sd*np.sqrt(2*n)
    mu=E_sd*np.sqrt(2*np.maximum(m,n))
    #if (mu<0.01): return 0.01
    #else:
        #return mu
    return np.maximum(mu,0.001)    

@jit
def getL(X,S,mu,L_penalty):
    mu=float(mu)
    L_penalty=float(L_penalty)
    #X=np.array(X).astype('float64')
    #S=np.array(S).astype('float64')
    L_penalty2 = L_penalty*mu
    C=np.subtract(X,S,out=None)#cambio
    L0,L1=SVT(C,L_penalty)
    L_nuclearnorm =np.sum(L1)
    return L0,L_penalty2*L_nuclearnorm

@jit
def getS(X,L,mu,s_penalty):
    mu=float(mu)
    s_penalty=float(s_penalty)
    #X=np.array(X).astype('float64')
    #L=np.array(L).astype('float64')
    s_penalty2 = s_penalty*mu
    C=np.subtract(X,L,out=None)#Cambio
    S=SoftThresholdMatrix(C,s_penalty2)
    S_l1norm = np.sum(np.abs(S))
    return S,s_penalty2*S_l1norm

@jit
def getE(X,L,S):
    #X=np.array(X).astype('float64')
    #L=np.array(L).astype('float64')
    #S=np.array(S).astype('float64')
    R=X.copy()
    np.subtract(X,L,out=R)
    E=np.subtract(R,S,out=None)
    #E=X-L-S
    return E,np.linalg.norm(E,'fro')

@jit
def objective(L,S,E):
    return (0.5*E)+L+S

#@jit
def RPCA(X,Lpenalty=-1,Spenalty =-1, verbose = True):
    """
    Robust Principal Component Pursuit.

    Parameters
    ----------

    X : numpy array 

    Lpenalty: float, default -1

    Spenalty: float, default -1
              Scalar to penalize remainder matrix to find Anomalous Values or Noise Values
    verbose:  bool, optional (default=False)
              Controls the verbosity of the matrix building process.

    Returns:
    --------
    X : numpy array original
    L_matrix : numpy array, L_matrix is low rank 
    S_matrix : numpy array, S_matrix is sparse 
    E_matrix : numpy array, E_matrix is the remainder matrix of noise

    Reference:
    ----------

    Stable Principal Component Pursuit
    Zihan Zhou, Xiaodong Li, John Wright, Emmanuel Candes, Yi Ma
    https://arxiv.org/pdf/1001.2363.pdf

    """
    X=np.array(X).astype('float64').copy()
    m,n=X.shape
    
    if (Lpenalty == -1):
        Lpenalty = 1
    if (Spenalty == -1):
        Spenalty=(1.6)/np.sqrt(max(n,m))
        #if (m > n):
        #    Spenalty = 1.4 / np.sqrt(m) 
        #else: 
         #   Spenalty = 1.4 / np.sqrt(n)
            
    itere=1
    maxIter=2000
    converged=False 
    obj_prev=0.5*np.linalg.norm(X,'fro')
    tol=(1e-10) * obj_prev
    diff=2*tol
    mu=(X.size)/(4*np.linalg.norm(X,1))
    print("Value obj_prev %2.10f and tol %2.10f"%(obj_prev,tol) )
    L_matrix =np.zeros_like(X,dtype='float')
    S_matrix =np.zeros_like(X,dtype='float')
    E_matrix =np.zeros_like(X,dtype='float')
    while (itere < maxIter and diff > tol):

        S_matrix,S_1 = getS(X, L_matrix, mu, Spenalty)
        #S_matrix = S[0]
        L_matrix,L_1 = getL(X, S_matrix, mu, Lpenalty)
        #L_matrix = L[0]
        E_matrix,E_1 = getE(X, L_matrix, S_matrix)
        #E_matrix = E[0]
        obj = objective(L_1,S_1, E_1)
        if (verbose):
            print("Objective function: %4.8f  on previous iteration %d "%(obj_prev,itere-1))
            print("Objective function: %4.8f  on iteration %d "%(obj,itere))
            diff=np.abs(obj_prev-obj)
            obj_prev=obj
            mu=getDynamicMu(E_matrix)
            itere +=1
            #print( "Diff Value:%2.10f and tol value %2.10f"%(diff,tol))
            if(diff<tol): 
                converged=True    
                break
        else:
            diff=np.abs(obj_prev-obj)
            obj_prev=obj
            mu=getDynamicMu(E_matrix)
            itere +=1
            if(diff<tol): 
                converged=True
                break

    if(converged):
        print("Converged within %d iterations"% itere)
        return X,L_matrix,S_matrix,E_matrix
    else:
        print("Failed to converge within %d maxIter iterations.\n" %maxIter)
        print("ERROR: Matrix Values do not converge!")
    
        return X,L_matrix,S_matrix,E_matrix


