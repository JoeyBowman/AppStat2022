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
    '''This function takes in chi2 fit values and the degrees of freedom and returns the probability of the chi2 value.'''
    chi2    =   minuit.fval
    test = stats.chi2.sf(chi2, dof)
    return test, dof


 # Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
def calc_ROC(hist1, hist2) :
    '''This functions takes in 2 histograms with the same binning and returns the ROC curve.'''

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")