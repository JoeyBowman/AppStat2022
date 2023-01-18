# coding: utf-8

#Author: Joey Bowman

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sympy as sp
import numpy.random as r

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
    REMEMBER: If it is binned data use length of bin centers NOT length of data.
    
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
    wf = np.linalg.inv(cov_sum)@(mean_a - mean_b)


    #Calculate the Fisher discriminant:
    fisher_a = np.dot(wf,species_a.T)
    fisher_b = np.dot(wf,species_b.T)


    return fisher_a, fisher_b, wf
 
def normalize(function, range):
    """
    Description:
    ------------
    Normalize a function.

    Parameters:
    -----------
    function : function
        The function to be normalized.
    range : tuple
        The range of the function.
    
    Returns:
    --------
    C : float
        The normalization constant.
    """
    x = sp.symbols('var')
    C = sp.symbols('norm')
    integration = sp.integrate(function(x,C),(x,range[0],range[1]))
    return sp.solve(integration-1, C)


def tranform_method(function,var,Npoints,limits,inv_func = False):
    """
    Description:
    ------------
    Generate random numbers from a function using the transform method.

    IMPORTANT: The function and variable should be defined through sympy.

    Parameters:
    -----------
    function : function
        The function to generate random numbers from.
    var : string
        The variable of the function.
    Npoints : int
        The number of random numbers to generate.
    limits : tuple
        The range of the function.
    inv_func : bool
        If True, the inverse function options are printed.
    
    Returns:
    --------
    Generated data : array_like
        The random numbers distributed according to the function.
    """

    full = sp.integrate(function,(var,limits[0],limits[1]))

    # First find the inverse of the CDF
    F = sp.integrate(function,(var,limits[0],var))/full

    # Now find the inverse of the CDF 
    y = sp.symbols('y', positive = True, real=True) 
    F_inv = sp.solve(F-y,var)

    if len(F_inv) > 1: 
        F_inv_pos = F_inv[1]
    else:
        F_inv_pos = F_inv[0]
    if inv_func == True:
        print(F_inv)
    
    inv_func = sp.lambdify(y,F_inv_pos)
    # Now generate the random numbers
    u = np.random.uniform(0,1,Npoints)
    return inv_func(u)

def acc_rej(function,norm, Npoints, limits, data = False):
    """
    Description:
    ------------
    Generate random numbers from a function using the acceptance-rejection method.

    Parameters:
    -----------
    function : function
        The function to generate random numbers according to.
    norm : float
        The normalization constant of the function.
    Npoints : int
        The number of random numbers to generate.
    limits : tuple
        x_min,x_max,ymax.
    data : bool
        If True, the number of tries during the loop and the efficiency of the algorithm are also retuned.
    
    Returns:
    --------
    Generated data : array_like
        The random numbers distributed according to the function.

    Notes:
    ------
    The function should be normalized.
    
    """

    x_accepted = np.zeros(Npoints)
    xmin, xmax, ymax = limits[0], limits[1], limits[2]
    Ntry = 0

    for i in range(Npoints):
        while True:
            Ntry += 1                    # Count the number of tries, to get efficiency/integral
            x = r.uniform(xmin, xmax)    # Range that f(x) is defined/wanted in
            y = r.uniform(0, ymax)       # Upper bound of the function
            if (y < function(x,norm)) :
                break
        x_accepted[i] = x
    
    efficiency = Npoints/Ntry
    if data == True:
        return x_accepted, Ntry, efficiency
    else:
        return x_accepted

def bins_create(data, Nbins):
    """
    Description:
    ------------
    Create bins for a given data set.
    
    Parameters:
    -----------
    data : array_like
        The data to be binned.
    Nbins : int
        The number of bins.

    Returns:
    --------
    bin_width : float
        The width of the bins.
    bin_edges : tuple
        The edges of the bins.
    """
    dmin, dmax = np.min(data), np.max(data)
    bin_width = (dmax - dmin)/Nbins
    return bin_width, (dmin, dmax)


def hist_create(data, Nbins, bin_width = None, datamin = None, datamax = None,plot=False, type='step',labe='Data',colour='black'):
    """
    Description:
    ------------
    Create a histogram from a given data set with counts in bins, poissonian errors on bins, the bin centers and the bin widths.
    If no bin width or data range is given, the function makes it's own bins from the data.

    Parameters:
    -----------
    data : array_like
        The data to be binned.
    Nbins : int
        The number of bins.
    bin_width : float
        The width of the bins.
    datamin : float
        The minimum of the data.
    datamax : float
        The maximum of the data.
    plot : bool
        If True, the histogram is plotted.
    type : string
        The type of histogram to be plotted.
    labe : string
        The label of the histogram.
    colour : string
        The colour of the histogram.

    Returns:
    --------
    count_int : array_like
        The counts in the bins.
    error_count_int : array_like
        The poissonian errors on the counts in the bins.
    center_bins_int : array_like
        The centers of the bins.
    bin_width : float
        The width of the bins.
    """


    if bin_width == None and datamin == None and datamax == None:    
        datamin, datamax = np.min(data), np.max(data)     # The minimum and maximum of the data
        bin_width = (datamax - datamin) / Nbins           # The width of the bins

    count_int,bins_int,_ = plt.hist(data, bins=Nbins, range=(datamin,datamax), histtype=type, label=labe, color=colour)
    mask_int = count_int > 0


    count_int = count_int[mask_int]
    error_count_int = np.sqrt(count_int)
    center_bins_int = (bins_int[:-1] + bins_int[1:]) / 2
    center_bins_int = center_bins_int[mask_int]
    if plot == True:
        plt.errorbar(center_bins_int, count_int, yerr=error_count_int, fmt='o', label='Data')
    
    return count_int, error_count_int, center_bins_int, bin_width, datamin, datamax


def gauss(x, mean, sigma,N):
    """
    Description:
    ------------
    A gaussian function.

    Parameters:
    -----------
    x : array_like
        The x values.
    mean : float
        The mean of the gaussian.
    sigma : float
        The standard deviation of the gaussian.
    N : float
        The normalization constant of the gaussian.

    Returns:
    --------
    Gaussian : array_like
        The gaussian function.
    """

    return N*stats.norm.pdf(x, mean, sigma)