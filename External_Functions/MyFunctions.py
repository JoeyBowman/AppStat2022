# coding: utf-8

#Author: Joey Bowman

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def matplotlib_plotparameters():
    '''This function sets the default parameters for matplotlib plots.'''
    
    #Set style to classic

    plt.style.use('classic')


    # Set the default figure size
    plt.rcParams['figure.figsize'] = (10, 8)
    # Set the default font size
    plt.rcParams['font.size'] = 12

    # Set the default legend font size
    plt.rcParams['legend.fontsize'] = 11

    # Set the default font family
    plt.rcParams['font.family'] = 'serif'
    # Set the default line width
    plt.rcParams['lines.linewidth'] = 2
    # Set the default line color
    plt.rcParams['lines.color'] = 'k'
    # Set the default marker size
    plt.rcParams['lines.markersize'] = 10
    # Set the default marker face color
    plt.rcParams['lines.markerfacecolor'] = 'r'
    # Set the default marker edge color
    plt.rcParams['lines.markeredgecolor'] = 'k'
    # Set the default marker edge width
    plt.rcParams['lines.markeredgewidth'] = 2
    # Set the default axes face color
    plt.rcParams['axes.facecolor'] = 'w'
    # Set the default axes edge color
    plt.rcParams['axes.edgecolor'] = 'k'
    # Set the default axes grid color
    plt.rcParams['grid.color'] = 'k'
    # Set the default axes grid alpha
    plt.rcParams['grid.alpha'] = 0.5
    # Set the default axes grid linestyle
    plt.rcParams['grid.linestyle'] = ':'
    # Set the default axes grid linewidth
    plt.rcParams['grid.linewidth'] = 0.5
    # Set the default axes tick direction
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    # Set minor tick parameters
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.minor.size'] = 4

    # Set major tick parameters
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5

    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 500

   
    # Set the default axes tick label size
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'


def chi2_prob(minuit, dof):
    '''
    Description:
    ------------
    This function calculates the chi2 probability from a minuit fit.
    
    Parameters:
    -----------
    minuit : object
        The minuit object.
    dof : int
        The degrees of freedom.

    Returns:
    --------
    test : float
        The chi2 probability.
    dof : int
        The degrees of freedom.
        '''
    chi2    =   minuit.fval
    test = stats.chi2.sf(chi2, dof)
    return test, dof


# Calculate ROC curves from two datasets (data1 is signal, data2 is background) given a certain bin range
# and number of bins:
def roc_curve(sig,bck, bin_range, n_bins, xtra_data=False) :
    """
    Description:
    ------------
    Calculate ROC curve from two datasets (data1 is signal, data2 is background) given a certain bin range
    and number of bins.

    Parameters:
    -----------
    sig : array_like
        The signal data.
    bck : array_like
        The background data.
    bin_range : tuple
        The range of the bins.
    n_bins : int
        The number of bins.

    Returns:
    --------
    TPR : array_like
        The true positive rate.
    FPR : array_like
        The false positive rate.

    """
    
    
    # Calculate the histogram of the signal and background:
    hist_sig, bin_edges = np.histogram(sig, bins=n_bins, range=bin_range)
    hist_bck, bin_edges= np.histogram(bck, bins=n_bins, range=bin_range)
    
    # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
    x_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    # Calculate the integral (sum) of the signal and background:
    integral_sig = hist_sig.sum()
    integral_bck = hist_bck.sum()
    
    # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
    TPR = np.cumsum(hist_sig[::-1])[::-1] / integral_sig # True positive rate
    FPR = np.cumsum(hist_bck[::-1])[::-1] / integral_bck # False positive rate
    
    #Calculate the area under the ROC curve:
    area = np.trapz(TPR, FPR)


    if xtra_data == True :
        return TPR, FPR, x_centers, hist_sig, hist_bck, area
    else :    
        return FPR, TPR

#Function to calculate Fisher discriminant:
def fisher_disc(species_a, species_b):
    """
    Description:
    ------------
    Calculate the Fisher discriminant between two species.

    Parameters:
    -----------
    species_a : array_like
        The data of species A with the parameters in the 2nd index.
    
    species_b : array_like
        The data of species B with the parameters in the 2nd index.

    Returns:
    --------
    fisher_a : array_like
        The Fisher discriminant for species A.
    fisher_b : array_like
        The Fisher discriminant for species B.
    wf : array_like
        The Fisher weights.
    """


    #Calculate the mean of each species:
    mean_a = np.mean(species_a, axis=0)
    mean_b = np.mean(species_b, axis=0)

    #Calculate the covariance matrix each species:
    cov_a = np.cov(species_a, rowvar=False)
    cov_b = np.cov(species_b, rowvar=False)
    cov_sum = cov_a + cov_b

    #The Fisher weights are the inverse of the sum of the covariance matrices:
    wf = np.dot(np.linalg.inv(cov_sum), (mean_a - mean_b))


    #Calculate the Fisher discriminant:
    fisher_a = np.dot(wf,species_a.T)
    fisher_b = np.dot(wf,species_b.T)


    return fisher_a, fisher_b, wf
 
