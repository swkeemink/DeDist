''' Functions to estimate decoding distributions

Author: Sander Keemink, swkeemink@scimail.eu
'''

import numpy as np
from scipy.stats import mvn

def sample_E(fun,theta,par,sigma,x,x_,n,full_return=False):
    ''' Samples n errors from a multivariate gaussian distribution.
    
    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    sigma : float
        sigma^2 is the variance of the gaussian noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    n : int
        number of realizations to sample
    full_return : binary
        if False, only returns decoding distribution. If true, also returns
        sampled errors, calculated mean, and covariance
        
    Returns
    -------
    array 
        for each stimulus in x_, what how often this stimulus has the 
        smallest error
    if full_return:
    array
        The sampled error profiles
    array
        The means for the multivariate normal
    array
        The covariance matrix for the multivariate normal
    '''    
    # find dimensionality of multivar Gaussian
    ns = len(x_)    
    
    # find real population response
    f = fun(x,theta,par)
    
    # find possible function values
    Fs = fun(x,x_.reshape(x_.shape+(1,)),par)
    
    # first, find the means
    means = np.sum( (f-Fs)**2,axis=1 )
    
    # now the covariances
    cov = np.zeros((ns,ns))
    cov = 4*sigma**2*np.sum(Fs*Fs[:,None],axis=2)
            
    # now do a set of realizations
    print 'Sampling from distribution'
    Errors = np.random.multivariate_normal(means,cov,size=n)
    sol_th = x_[Errors.argmin(axis=1)]
    print 'Done'
    # return values
    if full_return:
        return sol_th,Errors,means,cov
    else:
        return sol_th
        
def est_p(fun,theta,par,sigma,x,x_,n,full_return=False):
    ''' For each stimulus in fun, estimates the probability that it gives the
    smallest error. It does this by find the multivariate normal for the error
    at each x_, with the error at each other x_' subtracted.
    
    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    sigma : float
        sigma^2 is the variance of the gaussian noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    n : int
        number of realizations to sample
    full_return : binary
        if False, only returns decoding distribution. If true, also returns
        the calculated means and covariance for each stimulus in x_
        
    Returns
    -------
    array 
        for each stimulus in x_, the probability that this has the smallest
        error
    if full_return:
    array
        The full set of means. Dimensions as (len(x_)-1,len(x_)), such that 
        means[:,i] describes the full set of means for the error differences
        with stimulus i. 
    array
        The full set of covariances. Dimensions as (len(x_)-1,len(x_)-1,
        len(x_)). Thus covs[:,:,i] describes the relevant covariance matrix
        for stimulus i. 
        
        
    ''' 
    # find dimensionality of multivar Gaussian
    ns = len(x_)    
    
    # find real population response
    f = fun(x,theta,par)
    
    # make multidimensional version of x_ so less need for for loops
    # a + b.reshape(b.shape+(1,)) gives all possible combinations between 
    # a and b
    x_mult = x_.reshape(x_.shape+(1,))    
    
    # first, find all required function differences
    diffs = (fun(x,x_mult,par)[:,None]-fun(x,x_mult,par))
    diffs_sq = (fun(x,x_mult,par)[:,None]**2-fun(x,x_mult,par)**2)
    
    # then, find the means
    print 'finding means'
    means = np.zeros((ns-1,ns)) # sum((f-f')**2)
    # loop over all to be generated means
    for i in range(ns):
        # loop over all stimuli, except when i=j
        means[:i,i] = np.sum( diffs_sq[i,:i] - 2*f*diffs[i,:i],axis=1 )
        means[i:,i] = np.sum( diffs_sq[i,i+1:] - 2*f*diffs[i,i+1:],axis=1 )
    
    # now for the tough one, the covariances
    print 'finding covariances, ',
    covs = np.zeros((ns-1,ns-1,ns))
    print 'doing set x of ' + str(ns) + ':'
    # loop over coveriances to find
    for i in range(ns):
        print '\r'+str(i), 
        covs[:i,:i,i] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                  *diffs[i,:i],axis=2)
        covs[:i,i:,i] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                  *diffs[i,i+1:],axis=2)
        covs[i:,:i,i] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                  *diffs[i,:i],axis=2)
        covs[i:,i:,i] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                  *diffs[i,i+1:],axis=2)
    print ''
    
    # calculate the cumulative distribution for each of the calculated covarainces
    low = -np.ones(len(x_)-1)*1e50
    upp = np.zeros(len(x_)-1)
    p = np.zeros(ns)
    print 'calculating probability x of ' + str(ns) + ':'
    for i in range(ns):
        print '\r'+str(i), 
        p[i],e = mvn.mvnun(low,upp,means[:,i],covs[:,:,i])
        
    if full_return:
        return p, means,covs   
    else:
        return p