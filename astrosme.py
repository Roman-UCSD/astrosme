###################################################################
###                          ASTROSME                           ###
#                                                                 #
#  Python library to simulate the effect of physics beyond the    #
#  Standard Model on the polarization state of photons arriving   #
#  from astronomical sources according to the Standard Model      #
#  extension (SME) framework, search for SME effects in observed  #
#  polarimetry and simulate potential bias imposed on the         #
#  polarimetric measurements by the dust in the Milky Way         #
#                                                                 #
#  Developed by Roman Gerasimov and Andrew Friedman               #
#  University of California San Diego                             #
#                                                                 #
###################################################################

# Basic libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import re
from collections import OrderedDict
import operator
import warnings
import time
import sys
import pickle
import os

# Astronomy
from astropy.table import Table
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u

# Maths
import harmonics
import scipy.special as scsp
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ive
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import iv
from scipy.misc import derivative as spd

# Regression
import emcee
import multiprocessing
import copy

# Cosmetics
import tqdm


def mcmc_regression(likelihood, proposal, trials, n_dim, n_walkers, n_threads, show_progress, random_seed):
    """
    Carry out MCMC trials with Gaussian moves to sample a given likelihood space. This function is not designed to be
    called directly, but rather to be dispatched by mcmc_run().

    For MCMC chains, we use the "emcee" package developed by Dan Foreman-Mackey and collaborators. The original
    publication can be located at https://iopscience.iop.org/article/10.1086/670067/pdf. Specifically, we require
    emcee version 3.0

    arguments
        likelihood    :    (Callable) function of form f(x), returning the log-likelihood of x. x may be a list
                           of multiple variables for multidimensional likelihood spaces. The log-likelihood does not
                           need to be absolute and may be defined up to an arbitrary constant
        proposal      :    Width (standard deviation) of the move proposal distribution in all dimensions. The distribution is assumed
                           normal and centered at the origin. The initial positions of the walkers will be drawn from this
                           distribution. Subsequently, the steps of the walkers will be drawn from it as well
        trials:            Number of MCMC trials. The trials will be evenly divided among "n_walkers". Hence, "trials" must be
                           an integer, wholly divisible by "n_walkers"
        n_dim         :    Number of dimensions in the likelihood space. Corresponds to the dimensionality of the argument in the callable
                           passed in "likelihood"
        n_walkers     :    Number of MCMC walkers. It is usually recommended to set the number of walkers as twice the number of
                           dimensions. Whenever possible, different walkers will be spawned in different threads.
        n_threads     :    Number of threads to use. Each thread will be implementing one or more MCMC walkers
        show_progress :    Set to True to display status messages. Set to False to run silently.
        random_seed   :    Random seed to use to produce consistent results for testing purposes. Set to False to generate
                           fully random numbers instead (recommended in production!)

    returns
        storage       :    All accepted positions of all walkers (aka flatchains)
        accepted      :    Total fraction of accepted moves (among all walkers).
    """
    if random_seed != False:
            np.random.seed(random_seed)
    
    if trials % n_walkers != 0:
        raise ValueError('Cannot evenly distribute {} trials among {} walkers'.format(trials, n_walkers))

    # Initialize the walkers
    pos = np.random.normal(0, proposal, [n_walkers, n_dim])

    if show_progress:
        print('\nExecuting trials...')
    with multiprocessing.Pool(processes = n_threads) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood, pool=pool, moves = emcee.moves.GaussianMove(proposal ** 2.0))
        sampler.run_mcmc(pos, trials // n_walkers, progress = show_progress)
    if show_progress:
        print('\n All done!')
    
    return sampler.flatchain, int(np.mean(sampler.acceptance_fraction) * trials)

def __mcmc_likelihood(k):
    """
    Calculate the log-likelihood of compatibility for a given set of MCMC walkers' positions. This function should not be called
    on its own and is, instead, designed to be dispatched by mcmc_run()
    """
    global mcmc_universe, mcmc_catalogue, mcmc_instruments, x
    mcmc_universe.k = k
    return mcmc_universe.compatibility_catalogue(mcmc_catalogue, mcmc_instruments)

def mcmc_run(universe, catalogue, instruments = False, trials = 10000, bins = 100, show_progress = True, n_threads = -1, n_walkers = -1, random_seed = False, proposal = False):
    """
    Perform MCMC regression to establish constraints on the SME parameters that make a given universe compatible with a given
    catalogue of astronomical measurements. See the docstring of mcmc_regression() for technical details of the MCMC chains
    used

    arguments
        universe       :      astrosme.Universe() object, containing the universe to run the regression for. The object will
                              not be altered. The values of the SME coefficients set within the object will not be used
        catalogue      :      astrosme.Catalogue() object, storing the astronomical measurements
        instruments    :      Dictionary of loaded astrosme.Instrument() objects for every band listed in the catalogue. The
                              dictionary must be keyed by the machine name of the band. Alternatively, set to False to generate
                              all instruments from "catalogue" using the default settings. Defaults to False
        trials         :      Total number of MCMC trials to perform across all walkers. Must be wholly divisible by "n_walkers".
                              If "n_walkers" is -1 and the number of trials is not divisible by the final number of MCMC walkers,
                              it may be slightly adjusted automatically. Defaults to 10000
        bins           :      Number of bins in the output MCMC flatchain histograms (defaults to 100)
        show_progress  :      Set to True (default) to display intermediate progress reports. Set to False for a silent run
        n_threads      :      Number of threads to use. Set to -1 (default) to choose the same number of threads as the number
                              of available CPU cores
        n_walkers      :      Number of MCMC walkers to use. Set to -1 (default) to use twice as many walkers as the number
                              of SME coefficients, which is the value typically recommended
        random_seed    :      Random seed to use to produce consistent results for testing purposes. In deployment,
                              "random_seed" must be set to False (default)
        proposal       :      Standard deviation of the proposal distribution (in eV^(4-d)) or False to guess an appropriate value
                              automatically. This Gaussian, origin-centered distribution will be used to determine the initial positions
                              of the walkers as well as their step size during the trials

    returns
        result       :      Dictionary with the following keys
            time                 :    Time taken by the operation in seconds
            RAM                  :    Total RAM usage in bytes
            acceptance           :    Total fraction of accepted moves (among all walkers)
            two_sigma_lower      :    Minimum values of all SME coefficients at 2 sigma confidence (same length as the number of coefficients)
            one_sigma_lower      :    Minimum values of all SME coefficients at 1 sigma confidence
            one_sigma_higher     :    Maximim values of all SME coefficients at 1 sigma confidence
            two_sigma_higher     :    Maximum values of all SME coefficients at 2 sigma confidence
            histogram            :    Histograms of all SME coefficient flatchains
            flatchains           :    Array containing all accepted positions of all MCMC walkers
    """
    # For multiprocessing, store universes, catalogues and instruments as global variables, where every process can access them
    global mcmc_universe, mcmc_catalogue, mcmc_instruments

    histograms = []
    percentiles = [[], [], [], []]
    start_time = time.time()

    # Autoload all instruments if not provided
    if (type(instruments) == bool):
        instruments = {}
        for band in catalogue.bands.keys():
            params = catalogue.bands[band]
            instruments[band] = universe.instrument(params['wl'], params['t'], name = band)

    # Determine the number of threads (processes) if not specified by the user
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()
        if show_progress:
            print('Detected {} CPUs'.format(n_threads))

    # Spawn a universe to carry the calculations in. Note that the multiprocessing will make copies of the universe for
    # each child process automatically
    mcmc_universe = copy.deepcopy(universe)

    # Make the catalogue and the instruments available to all processes
    mcmc_catalogue = catalogue
    mcmc_instruments = instruments

    # If not provided, guess the width of the proposal distribution for the walkers. We do not want it to be too wide or too
    # narrow, as this will make the walkers unable to move. Here, we simply assume all walkers to have the same value and span
    # different orders of magnitude until we find one at which the likelihood transitions between its extreme values
    if type(proposal) == bool:
        # First, disable pretabulation in all instruments, as we will be going off the charts
        original_disable_pretabulation = {}
        for instrument in instruments.keys():
            original_disable_pretabulation[instrument] = instruments[instrument]._disable_pretabulation
            instruments[instrument]._disable_pretabulation = True
        if show_progress:
            print('Estimating a suitable width for the proposal distribution')
        def __get_likelihood(log_guess):
            guess = 10 ** log_guess
            k = np.full([len(universe.k)], guess)
            mcmc_universe.k = k
            try:
                return mcmc_universe.compatibility_catalogue(mcmc_catalogue, mcmc_instruments, log = False)
            except:
                return 0
        max_likelihood = __get_likelihood(-50)
        min_likelihood = __get_likelihood(50)
        log_guess = brentq(lambda x: __get_likelihood(x) - np.mean([min_likelihood, max_likelihood]), -50, 50)
        guess = 10 ** log_guess
        proposal = guess
        if show_progress:
            print('Estimated a suitable width for the proposal distribution as {}'.format(guess))
        # Enable pretabulation back
        for instrument in instruments.keys():
            instruments[instrument]._disable_pretabulation = original_disable_pretabulation[instrument]
    else:
        # Run the likelihood function at least to make sure that all pretabulation is done
        mcmc_universe.compatibility_catalogue(mcmc_catalogue, mcmc_instruments)

    # If the number of walkers is not set by the user, set it to twice the number of coefficients, as recommended
    if n_walkers == -1:
        n_walkers = len(universe.k) * 2
        while trials % n_walkers != 0:
            trials += 1
    if show_progress:
        print('Will employ {} MCMC walkers for {} trials'.format(n_walkers, trials))

    storage, accepted = mcmc_regression(__mcmc_likelihood, proposal, trials, len(universe.k), n_walkers, random_seed = random_seed, n_threads = n_threads, show_progress = show_progress)

    for i, hist in enumerate(np.array(storage).T):
        histograms += [np.histogram(hist, bins = bins)]
        percentiles[0] += [np.percentile(hist, 0.02275 * 100)]
        percentiles[1] += [np.percentile(hist, 0.15865 * 100)]
        percentiles[2] += [np.percentile(hist, 0.84135 * 100)]
        percentiles[3] += [np.percentile(hist, 0.97725 * 100)]
    result = {
        'time': time.time() - start_time,
        'RAM': round(sys.getsizeof(storage) / (1024.0 ** 2.0), 3),
        'acceptance': accepted / float(trials),
        'histograms': histograms,
        'two_sigma_lower': percentiles[0],
        'one_sigma_lower': percentiles[1],
        'one_sigma_higher': percentiles[2],
        'two_sigma_higher': percentiles[3],
        'flatchains': np.array(storage),
    }
        
    if show_progress:
        print('\n')
        print('Execution time: %s seconds' % (time.time() - start_time))
        print('RAM usage: {} MB'.format(round(np.array(storage).size * np.array(storage).itemsize / (1024.0 ** 2.0), 3)))
        print('Acceptance rate:', accepted / float(trials))
    
    return result


class Instrument:
    def __init__(self, d, wl, t, wl_unit = 'nm', name = 'Unnamed', theta_grid = False, theta_excess = 'warning'):
        """
        Initialize an instrument object for an SME universe of given dimension. The instrument object stores the transmission
        profile of the observation bandpass as well as precomputed tables of instrument-dependent functions for fast lookup.
        The instrument object will be required for simulating broadband measurements in the SME universe.

        Note that this constructor is expected to be dispatched by Universe.instrument() rather than called directly.
        Note to maintainer: Universe.instrument() supports most of the arguments of this function. Remember to duplicate any
        documentation presented here into the docstring of Universe.instrument()!

        arguments
            d               :   Mass dimension of the universe, for which this instrument is being initialized
            wl              :   Wavelength grid over which the transmission profile is defined. The units are specified in
                                "wl_unit" (defaults to nm)
            t               :   Vector of the same size as "wl", specifying the transmissivity of the instrument at the
                                wavelengths in "wl". Unitless, must be a fraction between 0 and 1
            wl_unit         :   Unit of "wl". "nm", "A" and "eV" are supported. Defaults to "nm"
            name            :   Name of the instrument. Defaults to "Unnamed"
            theta_grid      :   List of two values. The first value determines the maximum value of theta (see Universe.theta())
                                for which instrument-dependent functions will be pretabulated for fast lookup. The second
                                value sets the number of points. The tabulation will occur on both linear and logarithmic grids
                                with the same number of points in each. Hence, the effective number of tabulation points is
                                twice the value of the second element of "theta_grid". To disable pretabulation, set to False
                                (default). To determine the best range, start with an initial guess of [x, 250x] where x is
                                of the order 100 and perform a test run. Use the warnings thrown by "theta_excess" to adjust.
                                To avoid accuracy loss, it is recommended to maintain the overall ratio of 250 or so.
            theta_excess    :   Reaction when the range of lookup tables is exceeded. Supported values: "warning" to issue a
                                warning and compute the value directly, "silent" to ignore and compute the value directly,
                                "error" to issue a fatal error and halt. Defaults to "warning"
        
        returns
            Initialized astrosme.Instrument() object
        """
        self.__d = d
        # Internal use only. If set to True, all optimization by pretabulation will be force-disabled
        self._disable_pretabulation = False

        if wl_unit == 'nm':
            E = spc.h * spc.c / (wl * 1e-9) / spc.e
        elif wl_unit == 'A':
            E = spc.h * spc.c / (wl * 1e-10) / spc.e
        elif wl_unit == 'eV':
            E = wl * 1.0
        else:
            raise ValueError('Unknown wavelength unit, {}'.format(wl_unit))

        # Sort the transmisison profile in the order of increasing energy. Otherwise, NumPy gets confused
        L = sorted(zip(E, t), key=operator.itemgetter(0))
        E, t = zip(*L)

        self.name = name
        self.E = np.array(E)
        self.t = np.array(t)
        self.N = np.trapz(t, E)

        # F and G integrals deal with rapidly oscillating functions, which means that their accuracy
        # drops at large argument values, where the sampling rate of the integrand becomes comparable
        # to the oscillation rate. Fortunately, we know that both functions asymptote at F=1 and G=0,
        # so instead of using complicated integration techniques and/or upsampling, we can simply
        # find reasonable cutoff values for the argument and set the functions to their asymptotic
        # values afterwards...
        self.F_cutoff = np.inf
        self.G_cutoff = np.inf
        # Look for cutoff values in this range... Will accommodate most filters
        theta_guess = 10 ** np.linspace(0, 7, 1000)
        F_guess = self.F_tabulate(theta_guess)
        G_guess = self.G_tabulate(theta_guess)
        F_dev = np.zeros(np.shape(F_guess))
        G_dev = np.zeros(np.shape(G_guess))
        # Compute mean departures from the asymptotic values (1 for F, 0 for G) in bins of 20 samples
        for i in range(len(F_guess)):
            F_dev[i] = np.mean(np.abs(F_guess[max([0, i - 20]):min([len(F_dev), i + 20])] - 1))
            G_dev[i] = np.mean(np.abs(G_guess[max([0, i - 20]):min([len(G_dev), i + 20])]))
        # The cutoffs are set where the departures are smallest, i.e. after the function has asymptoted,
        # but before numerical integration errors take hold...
        self.F_cutoff = theta_guess[F_dev == np.min(F_dev)][0]
        self.G_cutoff = theta_guess[G_dev == np.min(G_dev)][0]

        # Sample theta on both linear and logarithmic scales
        if (type(theta_grid) == bool) or (theta_grid[0] <= 0) or (theta_grid[1] <= 0):
            self.theta_grid = False
        else:
            self.theta_grid = 10 ** np.linspace(np.log10(theta_grid[0])-20, np.log10(theta_grid[0] / theta_grid[1]), int(theta_grid[1] / 2))
            self.theta_grid = np.concatenate([10 ** np.linspace(np.log10(theta_grid[0] / theta_grid[1]), np.log10(theta_grid[0]), int(theta_grid[1] / 2)), self.theta_grid])
            self.theta_grid = np.concatenate([np.linspace(0, *theta_grid), self.theta_grid])
            self.theta_grid = np.sort(self.theta_grid)
            self.theta_excess = theta_excess
            self.F_grid = self.F_tabulate(self.theta_grid)
            self.G_grid = self.G_tabulate(self.theta_grid)

    def integrate(self, x, y):
        """
        Integrate a function numerically. Alias for np.trapz().

        arguments
            x               :   Independent variable grid
            y               :   Vector of the same size as "x". Values of the dependent variable across the "x" grid
        
        returns
            Integration result
        """
        return np.trapz(y, x)

    def F_tabulate(self, theta_grid):
        """
        Pretabulate the instrument-dependent F-integral for a given grid of theta values (see Universe.theta()).
        The F-integral is a special function such that:
                u'=u_z'(1-F)
        where u' and u_z' are the telescope and source values of the Stokes U parameter in a reference frame where
        xi is 0 (see Universe.xi()).
        Use F() to lookup the table.

        arguments
            theta_grid      :   theta values to tabulate the F-integral at
        
        returns
            Table of values
        """
        result = np.zeros(len(theta_grid))

        for i, value in enumerate(theta_grid):
            if value > self.F_cutoff:
                result[i] = 1.0
            else:
                result[i] = (2.0 / self.N) * self.integrate(self.E, self.t * np.sin(self.E ** (self.__d - 3.0) * value) ** 2.0)
        return result

    def G_tabulate(self, theta_grid):
        """
        Pretabulate the instrument-dependent G-integral for a given grid of theta values (see Universe.theta()).
        The G-integral is a special function such that:
                q=q_z(1-F)-0.5*u_z*G
                u=u_z(1-F)+0.5*q_z*G
        where u, q and u_z, q_z are the telescope and source values of the Stokes U, Q parameters in the standard frame
        of reference and F is the F-integral (see F_tabulate()).
        Use G() to lookup the table.

        arguments
            theta_grid      :   theta values to tabulate the G-integral at
        
        returns
            Table of values
        """
        result = np.zeros(len(theta_grid))
        for i, value in enumerate(theta_grid):
            if value > self.G_cutoff:
                result[i] = 0.0
            else:
                result[i] = (2.0 / self.N) * self.integrate(self.E, self.t * np.sin(2 * self.E ** (self.__d - 3.0) * value))
        return result

    def F(self, theta):
        """
        Lookup the F-integral table at a given value of theta applying interpolation as necessary. Refer to
        Universe.theta() and F_tabulate() for details.

        arguments
            theta      :   The value of theta to interpolate the F-integral table to
        
        returns
            Estimate of the F-integral
        """
        # The F-integral is even
        theta = np.abs(theta)
        if type(self.theta_grid) != bool and theta <= np.max(self.theta_grid) and (not self._disable_pretabulation):
            return np.interp(theta, self.theta_grid, self.F_grid)
        else:
            if type(self.theta_grid) != bool and (not self._disable_pretabulation):
                params = (self.name, theta, np.max(self.theta_grid))
                message = '{}: requested F-integral outside of tabulation limits: {}>{}!'.format(*params)
                if self.theta_excess == 'error':
                    raise ValueError(message)
                elif self.theta_excess == 'warning':
                    warnings.warn(message)
            return self.F_tabulate([theta])[0]

    def G(self, theta):
        """
        Lookup the G-integral table at a given value of theta applying interpolation as necessary. Refer to
        Universe.theta() and G_tabulate() for details.

        arguments
            theta      :   The value of theta to interpolate the G-integral table to
        
        returns
            Estimate of the G-integral
        """
        # The G-integral is odd
        sign = np.sign(theta)
        theta = np.abs(theta)
        if type(self.theta_grid) != bool and theta <= np.max(self.theta_grid) and (not self._disable_pretabulation):
            return sign * np.interp(theta, self.theta_grid, self.G_grid)
        else:
            if type(self.theta_grid) != bool and (not self._disable_pretabulation):
                params = (self.name, theta, np.max(self.theta_grid))
                message = '{}: requested G-integral outside of tabulation limits: {}>{}!'.format(*params)
                if self.theta_excess == 'error':
                    raise ValueError(message)
                elif self.theta_excess == 'warning':
                    warnings.warn(message)
            return sign * self.G_tabulate([theta])[0]

    def get_d(self):
        """
        Retrieve the mass dimension of the SME universe this instrument is compatible with. The value is set at
        initialization
        
        returns
            Mass dimension
        """
        return self.__d

class Universe:
    def __cache_init(self):
        """
        Initialize the caching system. This function must be called when the object is being created by
        __init__() at the very beginning. The purpose of this function is to make sure the cache storage
        exists and to prepare each relevant function for static caching.

        Satic caching is supported for any function within the class as long as its output does not depend
        on the object (self) and all input arguments can be uniquely typecast to strings. Do not attempt to
        use this mechanism to cache functions that depend on "self" or functions whose arguments cannot be reliably
        hashed with str().

        To enable static caching for a particular function, simply add:
            self.<func_name> = self.__cache_enable(self.<func_name>)
        at the end of __cache_init()
        """
        global __astrosme_cache
        try:
            __astrosme_cache
        except:
            __astrosme_cache = {'functions': {}}

        self.__sYjm = self.__cache_enable(self.__sYjm)

    def __cache_enable(self, func):
        """
        Enable static caching for a function within the Universe object passed as a callable in "func".
        Please refer to the docstring of __cache_init() before using this mechanism.
        
        arguments
            func        :   Function (callable) whose output must be statically cached
        
        returns
            Copy of "func" (callable) with output caching enabled.
        """
        def cached_func(*args, **kwargs):
            global __astrosme_cache
            key = '|'.join(list(map(str, args)))
            for kwarg in sorted(kwargs.keys()):
                key += '|' + str(kwarg) + ':' + str(kwargs[kwarg])
            try:
                return __astrosme_cache['functions'][key]
            except:
                __astrosme_cache['functions'][key] = func(*args, **kwargs)
                return __astrosme_cache['functions'][key]
        return cached_func

    def cache_reset(self):
        """
        Flush the entire global cache storage. Note that this function is not private, as it seems to be
        something a user might want to do. Note that this function will flush cache for all universe
        objects existing within the current session, not just the object this function belongs to.
        """
        global __astrosme_cache
        __astrosme_cache = {'functions': {}}
    
    def __init__(self, d, pol_z = 1.0, reset_cache = False):
        """
        A Universe() object defines a given SME model parametrized by its mass dimension (d) and a set of
        SME coefficients (k). The object implements methods to simulate physics under this SME model.
        The mass dimension (d) must be set on creation and cannot be changed later. The SME coefficient vector
        (k) will default to zeros and can be changed at any time either directly or using the methods provided.
        Set reset_cache to True to flush all cache that may have been created by previously existing objects
        of this class.

        The units of all SME coefficients are eV^(-d+4) throughout all of AstroSME

        arguments
            d           :   Desired mass dimension of the universe, int, >3
            pol_z       :   Assumed intrinsic polarization fraction of sources. Defaults to 1, which is usually
                            considered most conservative. If the maximum allowed polarization at the sources is
                            somehow known (e.g. from theory of active galactic nuclei emission), the value can be
                            specified here to tighten the probability of compatibility
            reset_cache :   If True, will flush static cache upon creation for all universes
        """
        self.__cache_init()
        if reset_cache:
            self.cache_reset()

        d = int(d)
        if d < 4:
            raise ValueError('Mass dimension must be 4 or larger!')
        self.__d = d

        # Lists to store labels for all coefficients in complex and real formats. Note that self.k stores
        # coefficients in the real format, while the complex format is converted to and from "on the fly"
        self.__k_complex_labels = []
        self.__k_real_labels = []

        if d % 2 == 1:
            modes = ['V']
            start_j = 0
        else:
            modes = ['E', 'B']
            start_j = 2

        # Generate labels
        for mode in modes:
            for j in range(start_j, d - 1):
                for m in range(-j, j + 1):
                    complex_label = 'k_({}){},{}'.format(mode, j, m)
                    if m == 0:
                        self.__k_real_labels += ['({})'.format(complex_label)]
                    elif m > 0:
                        self.__k_real_labels += ['(Re[{}])'.format(complex_label), '(Im[{}])'.format(complex_label)]
                    self.__k_complex_labels += [complex_label]
        # Initialize the universe with no SME
        self.k = np.zeros(len(self.__k_real_labels))

        # Default cosmological parameters (Planck 2018, VI, Table 2, column 7)
        self.H0  = 67.66
        self.O_r = 9.182e-5       #  for z_eq=3387
        self.O_m = 0.3111
        self.O_l = 0.6889
        self.O_k = False

        # Intrinsic polarization fraction
        self.pol_z = pol_z

    def __convert_to_latex(self, label, latex_mode):
        """
        Convert coefficient labels generated by __init__() into something LaTeX friendly...

        arguments
            label       :   Internal coefficient label (either real or complex) as generated by __init()__
            latex_mode  :   "text" or "math" depending on whether \textrm{} or \mathrm{} is preferred in the output

        returns
            LaTeX'ed label ready to be typeset
        """
        label = label.replace('Re', r'\textrm{Re}').replace('Im', r'\textrm{Im}').replace('text', latex_mode)
        label = re.sub(r'k_(\(.\)[-0-9]+,[-0-9]+)', r'k_{{\1}}', label)
        label = re.sub(r'\[(.+)\]', r'\\left[\1\\right]', label)
        label = re.sub(r'(?<!\{)\((.+)\)', r'\1', label)
        label = label.replace('k_', 'k^{{({})}}_'.format(self.__d))
        return label

    def get_k_real(self, latex = False, latex_mode = 'math'):
        """
        Get the current values of SME coefficients as an ordered dictionary with proper labeling.
        The coefficients are returned in the real format.
        For coefficients in the complex format, see get_k_complex().

        arguments
            latex       :   If False, return internal labels. Otherwise, LaTeX labels. See __convert_to_latex()
            latex_mode  :   If "latex" is True, "latex_mode" can be set to "text" or "math" depending on whether
                            \textrm{} or \mathrm{} is preferred in the output

        returns
            Ordered dictionary of properly labeled SME coefficients in the real format
        """
        labels = list(self.__k_real_labels)
        if latex:
            labels = list(map(lambda x: self.__convert_to_latex(x, latex_mode), labels))
        return OrderedDict(zip(labels, self.k))

    def get_k_complex(self, latex = False, latex_mode = 'math'):
        """
        Get the current values of SME coefficients as an ordered dictionary with proper labeling.
        The coefficients are returned in the complex format.
        For coefficients in the real format, see get_k_real().
        Note that internally, the coefficients are stored in the real format. This function performs an "on the fly"
        conversion. To do so, it uses regular expressions that are computationally heavy. As such, this function
        should only be used for display purposes. Do not use this function to obtain the values of the coefficients
        for calculations. Instead, access self.k directly or use get_k_real().

        arguments
            latex       :   If False, return internal labels. Otherwise, LaTeX labels. See __convert_to_latex()
            latex_mode  :   If "latex" is True, "latex_mode" can be set to "text" or "math" depending on whether
                            \textrm{} or \mathrm{} is preferred in the output
        
        returns
            Ordered dictionary of properly labeled SME coefficients in the complex format
        """
        k_real = self.get_k_real()
        labels = list(self.__k_complex_labels)
        values = []
        for label in labels:
            pattern = r'^(.*k_\(.\)[-0-9]+,)([-0-9]+)(.*)$'
            m = int(re.findall(pattern, label)[0][1])
            label = re.sub(pattern, r'\g<1>{}\g<3>'.format(abs(m)), label)
            if m == 0:
                values += [complex(k_real['({})'.format(label)])]
            elif m > 0:
                values += [k_real['(Re[{}])'.format(label)] + 1j * k_real['(Im[{}])'.format(label)]]
            elif m < 0:
                values += [((-1.0) ** (-m)) * (k_real['(Re[{}])'.format(label)] - 1j * k_real['(Im[{}])'.format(label)])]
        if latex:
            labels = list(map(lambda x: self.__convert_to_latex(x, latex_mode), labels))
        return OrderedDict(zip(labels, values))

    def set_k(self, updates, reset = False, safe_parity = True):
        """
        Update the current values of the SME coefficients. All coefficients to be updated are passed in
        "updates" as a dictionary with keys corresponding to internal coefficient labels generated by
        __init__() and values corresponding to the updated values.
        If "reset" is True, all coefficients will be reset to 0 before applying updates.
        Updates can be specified in both real and complex formats. In the latter case, an "on the fly"
        conversion will be performed.
        By default, the user is not allowed to update m<0 coefficients directly, as their values are parity-linked
        to their positive counterparts (also, we do not even store m<0 coefficients, as the values are stored in
        the real format, which omits all parity-driven redundant data). An attempt to update a m<0 coefficient
        will result in an error if safe_parity is True. Otherwise, the corresponding m>0 coefficient will be updated
        instead according to the parity relationships

        arguments
            updates     :   Dictionary of coefficients to be updated
            reset       :   If True, set all coefficients to 0 before the update (defaults to False)
            safe_parity :   If True, disallow editing m<0 complex coefficients (defaults to True)
        """
        updates = OrderedDict(updates)
        if reset:
            self.k = np.zeros(len(self.k))
        for update in updates.keys():
            if (update not in self.__k_real_labels) and (update not in self.__k_complex_labels):
                keywords = [update, self.__d, '; '.join(self.__k_real_labels + self.__k_complex_labels)]
                raise ValueError('Unknown coefficient {}. Available coefficients for d={}: {}'.format(*keywords))
            if update in self.__k_real_labels:
                if complex(updates[update]).imag != 0.0:
                    raise ValueError('{} must be real!'.format(update))
                self.k[self.__k_real_labels.index(update)] = complex(updates[update]).real
            if update in self.__k_complex_labels:
                pattern = r'^.*k_\((.)\)([-0-9]+),([-0-9]+).*$'
                mode, j, m = re.findall(pattern, update)[0]
                j = int(j); m = int(m)
                if m < 0:
                    if safe_parity:
                        error_msg = 'Cannot set the m={} coefficient directly! Its value is determined by the m={}' \
                                    ' coefficient through parity relationships. You must set the m={} coefficient ' \
                                    'instead or disable safe_parity to do so automatically'
                        raise ValueError(error_msg.format(m, abs(m), abs(m)))
                    else:
                        m = abs(m)
                        updates[update] = (-1.0) ** m * complex(updates[update]).conjugate()
                if m == 0:
                    if complex(updates[update]).imag != 0.0:
                        raise ValueError('All m={} coefficients must be real by parity relationships!'.format(m))
                    self.k[self.__k_real_labels.index('({})'.format(update))] = complex(updates[update]).real
                if m > 0:
                    label = 'k_({}){},{}'.format(mode, j, m)
                    self.k[self.__k_real_labels.index('(Re[{}])'.format(label))] = complex(updates[update]).real
                    self.k[self.__k_real_labels.index('(Im[{}])'.format(label))] = complex(updates[update]).imag

    def __redshift_integrator(self, d, H0, O_r, O_m, O_l, O_k):
        """
        Pretabulate effective comoving distance (Lz) for all redshifts between 0 and 10. Returns a table
        of z and corresponding Lz values. The output is cached for faster execution. Use cache_reset()
        to flush

        arguments
            H0          :   Hubble constant in km/s/Mpc
            O_r         :   Radiation density at present (\Omega_r)
            O_m         :   Matter (dark and baryonic) density at present (\Omega_m)
            O_l         :   Cosmological constant at present (\Omega_\Lambda)
            O_k         :   Spatial curvature density at present (\Omega_k). Set to False to assume flat universe,
                            i.e. O_k = 1 - (O_r + O_m + O_l)
        
        returns
            z           :   Table of redshifts from z=0 to z=10, uniformly sampled
            Lz          :   Table of corresponding effective comoving distances in 1/eV (units of convenience in
                            SME calculations). To convert to metres, use 1 eV = 1.9733e-7 m, where the convention of
                            energy = hbar * c / length is adopted
        """
        global __astrosme_cache
        try:
            __astrosme_cache['z_storage']
        except:
            __astrosme_cache['z_storage'] = {}; __astrosme_cache['Lz_storage'] = {}
        storage_key = '{} {} {} {} {} {}'.format(d, H0, O_r, O_m, O_l, O_k)
        try:
            return __astrosme_cache['z_storage'][storage_key], __astrosme_cache['Lz_storage'][storage_key]
        except:
            if O_k == False:
                O_k = 1 - (O_r + O_m + O_l)
            H0 = spc.hbar * H0 * 1e-3 / spc.parsec / spc.e      # Convert to eV
            __astrosme_cache['z_storage'][storage_key] = np.linspace(0, 10, 10000)
            __astrosme_cache['Lz_storage'][storage_key] = np.zeros(len(__astrosme_cache['z_storage'][storage_key]))
            for i, value in enumerate(__astrosme_cache['z_storage'][storage_key]):
                value = np.linspace(0, value, 10000)
                Hz = H0*np.sqrt(O_r*(1.0+value)**4.0 + O_m*(1.0+value)**3.0 + O_k*(1.0+value)**2.0 + O_l)
                __astrosme_cache['Lz_storage'][storage_key][i] = np.trapz((1 + value) ** (d - 4) / Hz, x = value)
            return __astrosme_cache['z_storage'][storage_key], __astrosme_cache['Lz_storage'][storage_key]

    def z_to_Lz(self, z):
        """
        Estimate the effective comoving distance for an object at a given redshift "z".
        On first run, the dependence will be pretabulated between redshifts of 0 and 10. The table will
        be saved in global cache, shared among all Universe() objects. To force a recalculation,
        call cache_reset() or restart your Python session.

        arguments
            z           :   Redshift to calculate effective comoving distance at
        
        returns
            Effective comoving distance in 1/eV (units of convenience in SME calculations). To convert to metres, use
            1 eV = 1.9733e-7 m, where the convention of energy = hbar * c / length is adopted
        """
        z_storage, Lz_storage = self.__redshift_integrator(self.__d, self.H0, self.O_r, self.O_m, self.O_l, self.O_k)
        return np.interp(z, z_storage, Lz_storage)

    def __sYjm(self, s, j, m, ra, dec):
        """
        Evaluate a spin-weighted spherical harmonic for a given point on the sky. For s=0, the function must reduce
        to regular spherical harmonics. We handle those using a routine from SciPy. For s!=0, a third-party module by
        Christian Reisswig is used instead. We could hypothetically use this module for s=0 cases as well,
        but SciPy tends to be considerably faster.

        In the standard physical convention, right ascension represents the phi (azimuthal) coordinate and declination
        represents the theta (latitudinal) coordinate, where the latter is offset such that the declination of the XY
        plane is 0 (usually, theta is 0 on the Z-axis and 90 on the XY plane)

        Note that the choice of coordinates is arbitrary and our use of equatorial coordinates is merely a convention, as
        this is how most astronomical measurements are reported. In principle, the same formalism can be applied to ecliptic,
        galactic or any other system of spherical coordinates.

        arguments
            s           :   Spin weight (in SME, s=0 for odd d and s=+/-2 for even d)
            j           :   Azimuthal number (j=0,1,2,3,...)
            m           :   Magnetic number (-j <= m <= j)
            ra          :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec         :   Declination on the celestial sphere in degrees, -90<=dec<=90
        
        returns
            Evaluated spin-weighted spherical harmonic (complex)
        """
        ra, dec = np.radians([ra, dec])
        phi = ra
        theta = np.pi / 2.0 - dec
        if s == 0:
            return scsp.sph_harm(m, j, phi, theta)
        else:
            return harmonics.sYlm(s, j, m, theta, phi)

    def S(self, ra, dec):
        """
        Calculate components of the birefringence axis vector for a unit-energy photon arriving from a source at
        given celestial coordinates.
        The aforementioned vector (usually denoted with \varsigma) is expressed in the spin-weighted Stokes basis
        defined as follows:
            s = {(Q - iU), V, (Q + iU)}
        where Q, U and V are the Stokes parameters. The first and the last components have spins of 2 and -2
        respectively, the middle component has a spin of 0.
        For the CPT-odd case (odd mass dimension), this function returns the 0-spin component, as it is the only
        non-zero component.
        For the CPT-even case, the 2-spin component is returned instead. The -2-spin component is simply a complex
        conjugate of the +2 spin component and is not returned.

        Reference: https://arxiv.org/pdf/0905.0031.pdf, eq 155

        arguments
            ra          :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec         :   Declination on the celestial sphere in degrees, -90<=dec<=90
        
        returns
            For odd mass dimensions (d), the 0-spin component of the birefringence axis vector in the spin-weighted Stokes
            basis. For even mass dimensions (d), the +2-spin component of the vector
        """
        result = 0.0
        k = self.get_k_real()

        if self.__d % 2 == 0:
            start_j = 2; s = 2
            modes = ['E', 'B']
            modes_coefficients = [1.0, 1.0j]
        else:
            start_j = 0; s = 0
            modes = ['V']
            modes_coefficients = [1.0]

        for j in range(start_j, self.__d - 1):
            for m in range(-j, j + 1):
                sh = self.__sYjm(s, j, m, ra, dec)
                current_k = []
                for i, mode in enumerate(modes):
                    if m == 0:
                        current_k += [k['(k_({}){},{})'.format(mode, j, m)]]
                    else:
                        pattern = '({}[k_({}){},{}])'
                        current_k += [k[pattern.format('Re', mode, j, abs(m))] + 1j*k[pattern.format('Im', mode, j, abs(m))]]
                        if m < 0:
                            current_k[-1] = (-1) ** abs(m) * current_k[-1].conjugate()
                    current_k[-1] *= modes_coefficients[i]
                result += sh * np.sum(current_k)
        return result
    
    def theta(self, ra, dec, z, current_S = False):
        """
        Calculate the characteristic angle of the SME-induced polarization drift for a unit-energy photon
        arriving from a source at given celestial coordinates and redshift.
        In the CPT-odd case (odd mass dimension), this result is literally the difference between the emitted
        and received polarization angles. See https://arxiv.org/pdf/0905.0031.pdf, eq 158.
        In the CPT-even case, the drift cannot be represented with a single angle as the birefringence axis is not
        fixed and may instead lie in the linear polarization plane (V=0) in any orientation. In such case, the drift
        is usually represented with a complex number with the magnitude of theta() and a phase that can be calculated
        with xi(). See eq 163 of the same reference.

        Note that theta() must be scaled by the energy dependence to obtain \Phi - the actual angle listed in the equation.

        This function relies on the output of S(). By default, S() will run afresh. For optimization purposes,
        a previously calculated value may be provided in "current_S" instead

        arguments
            ra          :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec         :   Declination on the celestial sphere in degrees, -90<=dec<=90
            z           :   Redshift of emission
            current_S   :   If provided, will force-set the birefringence axis to the given value. Otherwise,
                            the axis will be calculated automatically using S()
        
        returns
            Characteristic angle of the SME-induced polarization drift in radians
        """
        if not current_S:
            current_S = self.S(ra, dec)
        if self.__d % 2 == 0:
            # The absolute value is justified by the fact that the phase of the quantity is outsourced to a dedicated function
            current_S = abs(current_S)
        else:
            # In the CPT-odd case, the parity relationships should eliminate the imaginary part of S()
            current_S = current_S.real
        return self.z_to_Lz(z) * current_S

    def xi(self, ra, dec, current_S = False):
        """
        Calculate the phase angle of the propagation eigenmode for a photon arriving from a source given by its
        celestial coordinates.
        This is the second characteristic angle entering the CPT-even Muller matrix. The first angle can be calculated
        with theta() (pending appropriate energy scaling).
        This function assumes that S() returns the component of the birefringence axis of the spin +2. As such,
        this function should not be used in the CPT-odd case!

        Reference: https://arxiv.org/pdf/0905.0031.pdf, eq 160.

        This function relies on the output of S(). By default, S() will run afresh. For optimization purposes,
        a previously calculated value may be provided in "current_S" instead.

        arguments
            ra          :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec         :   Declination on the celestial sphere in degrees, -90<=dec<=90
            current_S   :   If provided, will force-set the birefringence axis to the given value. Otherwise,
                            the axis will be calculated automatically using S()
        
        returns
            Phase angle of the propagation eigenmode in radians
        """
        if self.__d % 2 == 1:
            raise ValueError('xi() is only available in the CPT-even case!')
        if not current_S:
            current_S = self.S(ra, dec)
        return -np.angle(current_S)

    def QU_to_polpsi(self, Q, U):
        """
        Convert Stokes Q and U into linear polarization fraction ("pol") and polarization angle ("psi") in degrees.
        Assume the Stokes parameters are intensity normalized (i.e. both are divided by Stokes I).
        The contribution of circular polarization is not considered

        arguments
            Q          :    Stokes Q, intensity normalized
            U          :    Stokes U, intensity normalized
        
        returns
            pol        :    (Linear) polarization fraction (0<pol<1)
            psi        :    Polarization angle in degrees
        """
        return np.sqrt(Q ** 2.0 + U ** 2.0), np.degrees(0.5 * np.arctan2(U, Q))

    def polpsi_to_QU(self, pol, psi):
        """
        Calculate Stokes Q and U for a source with given linear polarization fraction and polarization angle in degrees.
        The resulting Stokes parameters will be intensity normalized (i.e. as if they were divided by Stokes I).
        The contribution of circular polarization is not considered

        arguments
            pol        :    (Linear) polarization fraction (0<pol<1)
            psi        :    Polarization angle in degrees
        
        returns
            Q          :    Stokes Q, intensity normalized
            U          :    Stokes U, intensity normalized
        """
        psi = np.radians(psi)
        return pol * np.cos(2 * psi), pol * np.sin(2 * psi)

    def muller_apply(self, direction, Q, U, phi, xi, V = False):
        """
        Calculate the change in Stokes parameters due to SME effects (equivalent to constructing and
        applying the Muller matrix from eq 164 of https://arxiv.org/pdf/0905.0031.pdf).
        "Q" and "U" are the initial Stokes parameters before SME, intensity normalized.
        If "direction" is True, "Q", "U" and "V" are assumed to be given at source, while the result will be at
        the telescope. If "direction" is False, the opposite will be assumed (i.e., eq 164 will be inverted).
        SME is parametrized by "phi" and "xi". "xi" is the phase angle of the propagation eigenmode
        (can be computed with xi()). "phi" is the characteristic angle of the SME-induced polarization drift.
        theta() can be used to calculate "phi" for a unit-energy photon, which must be further scaled by the energy
        dependence.

        Under most circumstances, this function should not be called directly. Use send_photon() and
        receive_photon() instead that are much more straightforward.

        If Stokes "V" is given, the calculation will be carried out in full. Otherwise, an optimized algorithm
        is used that should perform much faster

        arguments
            direction   :   If True, apply the Muller matrix. Otherwise, apply its inverse
            Q           :   Pre-SME Stokes Q, intensity normalized (post-SME if "direction" is False)
            U           :   Pre-SME Stokes U, intensity normalized (post-SME if "direction" is False)
            phi         :   Energy-scaled characteristic angle of the SME-induced polarization drift in radians
            xi          :   Phase angle of the propagation eigenmode in radians
            V           :   Pre-SME Stokes V, intensity normalized (post-SME if "direction" is False). If not
                            given, V=0 will be assumed at the source if "direction" is True and V=x will be
                            assumed if "direction" is False, where x is some value that results in V=0 at source
        
        returns
            Q           :   Post-SME Stokes Q, intensity normalized (pre-SME if "direction" is False)
            U           :   Post-SME Stokes U, intensity normalized (pre-SME if "direction" is False)
            V           :   Post-SME Stokes U, intensity normalized (pre-SME if "direction" is False). Only returned
                            if "V" is provided as input
        """
        cos_phi_squared = np.cos(phi) ** 2        # Shorthand for cos(phi)^2 for convenience
        # If no V is given, we trim the Muller matrix to its corner elements and only evaluate the first row, which
        # provides both Q and U in the spin-weighted basis, as the first element of the polarization state vector is
        # Q-iU!
        if type(V) == bool:
            if direction:        # Receiving...
                muller = [cos_phi_squared, (1 - cos_phi_squared) * np.exp(-2j * xi)]
            else:                # Sending...
                muller = [cos_phi_squared, (cos_phi_squared - 1) * np.e ** (-2 * 1j * xi)]
                muller = np.array(muller) / (cos_phi_squared ** 2.0 - (1 - cos_phi_squared) ** 2.0)
            s = [Q - 1j * U, Q + 1j * U]
            s = np.dot(muller, s)
            return s.real, -s.imag
        else:
        # Otherwise, we build the matrix in full and use NumPy's linear algebra routines to invert it if "direction"
        # is False. This is much slower...
            s = np.matrix([Q - 1j * U, V, Q + 1j * U]).T
            muller = [
                        [cos_phi_squared, -1j * np.sin(2 * phi) * np.e ** (-1j * xi), (1 - cos_phi_squared) * np.exp(-2j * xi)],
                        [-0.5j * np.sin(2.0 * phi) * np.e ** (1j * xi), np.cos(2.0 * phi), 0.5j * np.sin(2.0 * phi) * np.e ** (-1j * xi)],
                        [(1 - cos_phi_squared) * np.e ** (2j * xi), 1j * np.sin(2 * phi) * np.e ** (1j * xi), cos_phi_squared],
                     ]
            muller = np.matrix(muller)
            if not direction:
                muller = muller.I
            s = muller * s
            s = np.array(s).T[0]
            return s[0].real, -s[0].imag, s[1].real

    def __transmit_photon(self, direction, ra, dec, E, z, psi, polarization_fraction = False, phi = False, xi = False, V = False):
        """
        Calculate the SME-induced change in polarization angle and (optionally) polarization fraction and
        the circular polarization component of a photon arriving from a source at given celestial coordinates,
        energy and redshift.
        If "direction" is True, "psi", "polarization_fraction" and "V" are interpreted as measured at the source.
        The result will be as measured at the telescope. If "direction" is False, the opposite calculation will
        be done (i.e. estimating the polarization at the source based on the value measured at the telescope).

        "polarization_fraction" can be set to False to use the default value of full polarization (1.0).
        "V" can be set to False to use the default value of 0.0 at the source (if "direction" is False, some value
        of "V" at the telescope will be adopted that ensures V=0 at the source)

        SME is parametrized by "phi" and "xi". "xi" is the phase angle of the propagation eigenmode
        (can be computed with xi()). "phi" is the characteristic angle of the SME-induced polarization drift.
        theta() can be used to calculate "phi" for a unit-energy photon, which must be further scaled by the energy
        dependence. If not provided, the function will estimate both afresh. If the values are already known,
        passing them as arguments will improve the performance.

        arguments
            direction               :   If True, calculate results at the telescope. Otherwise, at the source.
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            E                       :   Energy of the photon in eV
            z                       :   Redshift of emission
            psi                     :   Polarization angle of the photon in degrees (initial)
            polarization_fraction   :   Polarization fraction of the photon (initial, between 0 and 1), defaults to 1
            phi                     :   Energy-scaled characteristic angle of the SME-induced polarization drift
                                        in radians. Will be calculated if not provided.
            xi                      :   Phase angle of the propagation eigenmode in radians.  Will be calculated
                                        if not provided.
            V                       :   Stokes V of the photon (initial), intensity normalized. If not
                                        given, V=0 will be assumed at the source if "direction" is True and V=x will be
                                        assumed if "direction" is False, where x is some value that results in V=0 at source
        
        returns
            psi                     :   Polarization angle of the photon in degrees (final)
            pol                     :   Polarization fraction of the photon (final). Returned only if "polarization_fraction"
                                        or "V" is provided
            V                       :   Stokes V of the photon, intensity normalized (final). Returned only if "V" is provided
        """
        try:
            if V.dtype == bool:
                V = bool(V)
        except:
            pass

        pol = 1.0
        if polarization_fraction:
            pol = polarization_fraction
        if not phi:
            S = self.S(ra, dec)
            theta = self.theta(ra, dec, z, current_S = S)
            phi = E ** (self.__d - 3) * theta
        if self.__d % 2 == 0:
            Q, U = self.polpsi_to_QU(pol, psi)
            if not xi:
                try:
                    xi = self.xi(ra, dec, current_S = S)
                except:
                    xi = self.xi(ra, dec)
            if type(V) == bool:
                Q, U = self.muller_apply(direction, Q, U, phi, xi)
            else:
                Q, U, V = self.muller_apply(direction, Q, U, phi, xi, V)
        else:
            if not direction:
                phi *= -1.0
            Q, U = self.polpsi_to_QU(pol, psi + np.degrees(phi))

        pol, psi = self.QU_to_polpsi(Q, U)
        if type(V) != bool:
            return psi, pol, V
        if polarization_fraction:
            return psi, pol
        else:
            return psi

    def receive_photon(self, ra, dec, E, z, psi, polarization_fraction = False, phi = False, xi = False, V = False):
        """
        Predict the polarization angle of a photon at the telescope that was emitted by a source at given
        celestial coordinates and redshift, assuming a given original polarization angle (as measured at
        the source immediately after emission)

        Optionally, can also perform calculations for the polarization fraction and the circular polarization
        component.

        SME is parametrized by "phi" and "xi". "xi" is the phase angle of the propagation eigenmode
        (can be computed with xi()). "phi" is the characteristic angle of the SME-induced polarization drift.
        theta() can be used to calculate "phi" for a unit-energy photon, which must be further scaled by the energy
        dependence. If not provided, the function will estimate both afresh. If the values are already known,
        passing them as arguments will improve the performance.

        The inverse function is send_photon()

        This function is vectorized, meaning that multiple values for any of the arguments can be provided following
        the standard NumPy vectorization conventions.

        arguments
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            E                       :   Energy of the photon in eV
            z                       :   Redshift of emission
            psi                     :   Polarization angle of the photon in degrees (source)
            polarization_fraction   :   Polarization fraction of the photon (source, between 0 and 1), defaults to 1
            phi                     :   Energy-scaled characteristic angle of the SME-induced polarization drift
                                        in radians. Will be calculated if not provided.
            xi                      :   Phase angle of the propagation eigenmode in radians.  Will be calculated
                                        if not provided
            V                       :   Stokes V of the photon (source), intensity normalized. If not given, assume V=0
        
        returns
            psi                     :   Polarization angle of the photon in degrees (telescope)
            pol                     :   Polarization fraction of the photon (telescope). Returned only if
                                        "polarization_fraction" is provided
            V                       :   Stokes V of the photon, intensity normalized (telescope). Returned only if "V" is provided
        """
        processor = np.vectorize(self.__transmit_photon)
        return processor(True, ra, dec, E, z, psi, polarization_fraction, phi, xi, V)

    def send_photon(self, ra, dec, E, z, psi, polarization_fraction = False, phi = False, xi = False, V = False):
        """
        Reconstruct the original polarization angle of a photon at the source at given celestial coordinates
        and redshift, given a polarization angle measured at the telescope

        Optionally, can also perform calculations for the polarization fraction and the circular polarization
        component.

        SME is parametrized by "phi" and "xi". "xi" is the phase angle of the propagation eigenmode
        (can be computed with xi()). "phi" is the characteristic angle of the SME-induced polarization drift.
        theta() can be used to calculate "phi" for a unit-energy photon, which must be further scaled by the energy
        dependence. If not provided, the function will estimate both afresh. If the values are already known,
        passing them as arguments will improve the performance.

        The inverse function is receive_photon()

        This function is vectorized, meaning that multiple values for any of the arguments can be provided following
        the standard NumPy vectorization conventions.

        arguments
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            E                       :   Energy of the photon in eV
            z                       :   Redshift of emission
            psi                     :   Polarization angle of the photon in degrees (telescope)
            polarization_fraction   :   Polarization fraction of the photon (telescope, between 0 and 1), defaults to 1
            phi                     :   Energy-scaled characteristic angle of the SME-induced polarization drift
                                        in radians. Will be calculated if not provided.
            xi                      :   Phase angle of the propagation eigenmode in radians.  Will be calculated
                                        if not provided
            V                       :   Stokes V of the photon (source), intensity normalized. If not given, assume some
                                        value of V that ensures V=0 at source
        
        returns
            psi                     :   Polarization angle of the photon in degrees (source)
            pol                     :   Polarization fraction of the photon (source). Returned only if "polarization_fraction"
                                        is provided
            V                       :   Stokes V of the photon, intensity normalized (source). Returned only if "V" is provided
        """
        processor = np.vectorize(self.__transmit_photon)
        return processor(False, ra, dec, E, z, psi, polarization_fraction, phi, xi, V)

    def instrument(self, wl, t, **kwargs):
        """
        Initialize an instrument object compatible with this SME universe. The instrument object stores the transmission
        profile of the observation bandpass as well as precomputed tables of instrument-dependent functions for fast lookup.
        The instrument object will be required for simulating broadband measurements in the SME universe.

        arguments
            wl              :   Wavelength grid over which the transmission profile is defined. The units are specified in
                                "wl_unit" (defaults to nm)
            t               :   Vector of the same size as "wl", specifying the transmissivity of the instrument at the
                                wavelengths in "wl". Unitless, must be a fraction between 0 and 1
            wl_unit         :   Unit of "wl". "nm", "A" and "eV" are supported. Defaults to "nm"
            name            :   Name of the instrument. Defaults to "Unnamed"
            theta_excess    :   Reaction when the range of lookup tables is exceeded. Supported values: "warning" to issue a
                                warning and compute the value directly, "silent" to ignore and compute the value directly,
                                "error" to issue a fatal error and halt. Defaults to "silent"
            theta_grid      :   List of two values. The first value determines the maximum value of theta (see theta())
                                for which instrument-dependent functions will be pretabulated for fast lookup. The second
                                value sets the number of points. The tabulation will occur on both linear and logarithmic grids
                                with the same number of points in each. Hence, the effective number of tabulation points is
                                twice the value of the second element of "theta_grid". To disable pretabulation, set to False
                                (default). To determine the best range, start with an initial guess of [x, 250x] where x is
                                of the order 100 and perform a test run. Use the warnings thrown by "theta_excess" to adjust.
                                To avoid accuracy loss, it is recommended to maintain the overall ratio of 250 or so.

        returns
            Initialized astrosme.Instrument() object
        """
        if (np.min(t) < 0.0) or (np.max(t) > 1.0):
            raise ValueError('Invalid transmissivity vector, values outside of 0<=t<=1 found!')

        return Instrument(self.__d, wl, t, **kwargs)
        

    def rho(self, ra, dec, E, z, psi = False, telescope = True, analytic = True, dE = 1e-8):
        """
        Predict SME induced polarization angle rotation across the spectrum for photons arriving from a source
        at given celestial coordinates and redshift. The rotation will be linearized to a single number
        (in deg/eV), evaluated at some given energy. Mathematically, the result is the first derivative of the
        polarization angle spectrum with respect to energy taken at some given energy.
        To carry out this calculation in CPT-even universes, the function needs to know the polarization angle of
        the source, "psi" (in degrees) at the same energy. If "telescope" is set to True (default), "psi" will be
        interpreted as the angle measured at the telescope, which will be automatically converted to the angle at
        the source required in the calculation. Otherwise, it will be interpreted as the angle measured at the source
        right away (which is unlikely to be known in a realistic scenario). In the CPT-odd case, neither "psi" nor
        "telescope" are needed.

        Two algorithms are supported: numerical derivative (set "analytic" to False) and analytic equation
        (set "analytic" to True, default). In performance tests, the latter typically performs a little faster.
        If the former algorithm is chosen, the desired step size in energy (in eV) can be specified.

        arguments
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            E                       :   Energy of linearization and energy at which "psi" is specified (in eV)
            z                       :   Redshift of emission
            psi                     :   Polarization angle of the photon in degrees at energy "E". Unneeded in CPT-odd cases
            telescope               :   If True, "psi" is interpreted as measured polarization at the telescope. Otherwise,
                                        intrinsic polarization at the source pre-SME. Defaults to True. Unneeded in CPT-odd
                                        cases
            analytic                :   If True, the derivative is evaluated analytically. Otherwise, numerical differentiation
                                        is performed using the finite difference method. Defaults to True
            dE                      :   If "analytic" is False, "dE" specifies the finite difference used when evaluating
                                        the derivative in eV. Defaults to 1e-8
        
        returns
            SME induced polarization angle rotation across the spectrum in deg/eV
        """
        S = self.S(ra, dec)
        theta = self.theta(ra, dec, z, current_S = S)
        phi = E ** (self.__d - 3) * theta
        if self.__d % 2 == 0:
            if type(psi) == bool:
                    raise ValueError('Polarization angle is required for CPT-even universes!')
            xi = self.xi(ra, dec, current_S = S)
        else:
            xi = False
        if telescope:
            psi_z = self.send_photon(ra, dec, E, z, psi, phi = phi, xi = xi)
        else:
            psi_z = psi

        if not analytic:
            dpsi = self.receive_photon(ra, dec, E + dE, z, psi_z, phi = phi, xi = xi)
            if not telescope:
                dpsi -= self.receive_photon(ra, dec, E, z, psi_z, phi = phi, xi = xi)
            else:
                dpsi -= psi
            result = dpsi / dE
        else:
            psi_z = np.radians(psi_z)
            if self.__d % 2 == 0:
                result = 2 * (self.__d - 3) * phi * np.sin(2 * phi) * np.sin(2 * (xi - 2 * psi_z))
                result /= (3 + np.cos(4 * phi) + 2 * np.cos(2 * (xi - 2 * psi_z)) * np.sin(2 * phi) ** 2.0) * E
            else:
                result = (self.__d - 3) * phi / E
            result = np.degrees(result)
        return result
        

    def pol_max(self, ra, dec, instrument, z, psi = False, pol_z = 1.0, telescope = True, analytic = True, return_V = False):
        """
        Predict maximum polarization fraction allowed within this universe for a given source as observed through
        a given instrument.
        The source is assumed to emit photons of wavelength-independent polarization fraction given in "pol_z" and some
        wavelength-independent polarization angle that can be either given in "psi" directly by passing the value
        and setting the "telescope" to False, or it can be estimated from the polarization fraction observed by the
        instrument, in which case the value still needs to be passed in "psi", but "telescope" must be set to False.
        In the CPT-odd case, the result will not depend on the polarization angle of the source and, hence, there is no
        need to pass either "psi" or "telescope" to this function.

        Two algorithms are supported: full numerical simulation of photon exchange (set "analytic" to False) and analytic
        equation (set "analytic" to True, default). The former algorithm is considerably slower (typically, multiple orders
        of magnitude); however, it is more robust and can be used in unit tests. The numerical algorithm may also yield
        a slightly higher precision (typically, in the fourth decimal place or so).

        Additionally, the minimum allowed circular polarization (normalized Stokes V) at the telescope may also be calculated
        by setting "return_V" to True. In this case, the source is assumed to have no circular polarization at all wavelengths

        arguments
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            instrument              :   astrosme.Instrument() object specifying the observation bandpass
            z                       :   Redshift of emission
            psi                     :   Polarization angle in degrees as observed by "instrument" (broadband) if "telescope"
                                        is True. Otherwise, polarization angle at the source (assumed constant throughout
                                        the bandpass of the instrument). Unneeded in CPT-odd universes
            pol_z                   :   Polarization fraction at the source (assumed constant throughout the bandpass of
                                        the instrument). Note that this value is measured at the source regardless of
                                        "telescope". Defaults to 1.0 (most conservative assumption)
            telescope               :   See "psi". Unneeded in CPT-odd universes
            analytic                :   If True, use an analytic formula (fast). Otherwise, simulate photon exchange
                                        numerically (slow). Defaults to True
            return_V                :   If True, return the normalized minimum circular polarization (Stokes V) corresponding
                                        to the calculated maximum linear polarization. Defaults to False
        
        returns
            pol                     :   Maximum observable (linear) polarization fraction from the given source through
                                        the given instrument
            V                       :   Only returned if return_V is True. Corresponding normalized Stokes V
        """
        # Quick check to avoid silly mistakes...
        if instrument.get_d() != self.__d:
            raise ValueError('{} was initialized for a universe of a different mass dimension!'.format(instrument.name))

        S = self.S(ra, dec)
        theta = self.theta(ra, dec, z, current_S = S)
        F = instrument.F(theta)

        if analytic:
            if self.__d % 2 == 0:
                if type(psi) == bool:
                    raise ValueError('Polarization angle is required for CPT-even universes!')
                xi = np.degrees(self.xi(ra, dec, current_S = S))
                if not telescope:
                    # We follow Kislat 2018 (mdpi.com/2073-8994/10/11/596/htm) for the analytic case. Note that
                    # this paper treats all Stokes parameters in a rotated basis: psi' = psi - (xi/2).
                    # Q_z, U_z, Q_m and U_m below are expressed in this basis and, hence, do not follow the standard
                    # definition of Stokes parameters
                    Q_z, U_z = self.polpsi_to_QU(pol_z, psi - xi / 2.0)
                else:
                    # In the CPT-even case, we use equation 35 of Kislat 2018 to convert from observer to source.
                    # First, express the value of "psi" given to us in the rotated (primed) basis
                    psi = psi - xi / 2.0
                    # Now apply an inverted form of equation 35. Note that the inversion introduces a sign ambiguity
                    # in both Q_z and U_z, which we ignore here as we only need their absolute values to calculate
                    # the polarization fraction. Therefore, the signs of U_z and Q_z calculated here may be wrong,
                    # but the output should be correct
                    x = np.tan(2 * np.radians(psi))
                    U_z = pol_z * x / np.sqrt(1 - 2 * F + F ** 2.0 + x ** 2.0)
                    # Finally, get Q from U and total polarization
                    if np.abs(U_z) > pol_z:
                        # warnings.warn('Detected unphysical U_z value, {}. Corrected to {}'.format(U_z, np.sign(U_z) * pol_z))
                        U_z = np.sign(U_z) * pol_z
                    Q_z = np.sqrt(pol_z ** 2.0 - U_z ** 2.0)
                # Apply equations 30 and 31 of Kislat 2018 to convert back to observer
                Q_m = Q_z
                U_m = U_z * (1 - F)
                result = np.sqrt(Q_m ** 2.0 + U_m ** 2.0)
                if return_V:
                    # If V is necessary then the sign ambiguity in calculated Q_z and U_z (and hence Q_m and U_m) must be
                    # addressed. The rotated (primed) basis has the axis of birefringence aligned with the Q-axis in the Stokes
                    # space, which means that the sign of Q_z is still irrelevant when determining V at the observer. The sign
                    # of U_z however would change the sign of V, so we need to know it
                    if telescope:
                        # Assume that U_z is positive. Evaluate the sign of equation 35 and compare it to the sign of psi
                        # (after psi is shifted to the standard range between -90 and 90). If they match then the assumption
                        # of U_z being positive is correct...
                        if np.sign(np.arctan2(U_z * (1 - F) , 1.0)) == np.sign((psi + 90) % 180 - 90):
                            sign = +1
                        else:
                        # Otherwise, U_z is negative
                            sign = -1
                    else:
                        # If we are given pzi_z at the source instead of telescope than the calculations that introduced the sign
                        # ambiguity in U_z and Q_z never took place, so we can keep all signs as they are
                        sign = +1
                    # For V_z=0, V_m = -0.5 * G * U_z which may be inferred from the form of the Muller matrix
                    G = instrument.G(theta)
                    V_m = - 0.5 * G * sign * U_z
                    return result, V_m
                return result

            else:
                # The CPT-odd case is drastically easier, as there is no dependence on the polarization angle.
                # We follow the same procedure as Kislat 2018 in the CPT-even case, but adapted to CPT-odd universes.
                # Effectively, below is a modified version of equation 34
                G = instrument.G(theta)
                result = pol_z * np.sqrt((1 - F) ** 2.0 + 0.25 * G ** 2.0)
                if return_V:
                    return result, 0.0
                else:
                    return result

        else:
            # For the numerical method, we simply use receive_photon() to simulate the entire photon exchange.
            # First, define a function that estimates Q and U (in normal, unprimed frame) at the telescope given
            # some polarization angle at the source
            def predict_QU(psi_z, return_V = False):
                response = self.receive_photon(ra = ra, dec = dec, z = z, E = instrument.E, psi = psi_z, polarization_fraction = pol_z, V = 0.0)
                Q_m, U_m = self.polpsi_to_QU(*response[:2][::-1])
                Q_m = instrument.integrate(instrument.E, instrument.t * Q_m) / instrument.N
                U_m = instrument.integrate(instrument.E, instrument.t * U_m) / instrument.N
                if return_V:
                    V_m = instrument.integrate(instrument.E, instrument.t * response[2]) / instrument.N
                    return Q_m, U_m, V_m
                return Q_m, U_m
            # Rotate polarization angle into standard quadrant, such that -90<=psi<=90
            psi = (psi + 90) % 180 - 90
            if not telescope:
                psi_z = psi
            else:
                # Use SciPy's minimize() to invert predict_QU() and work out psi at the source given observed psi
                result = minimize(lambda x: np.abs(self.QU_to_polpsi(*predict_QU(x))[1] - psi), [0], bounds = [(-180, 180)], method = 'Powell')
                # The function runs into instabilities close to zero. If this is the case, just return 0
                psi_z = result.x
            if return_V:
                Q_m, U_m, V_m = predict_QU(psi_z, return_V = True)
                return np.sqrt(Q_m ** 2.0 + U_m ** 2.0), V_m
            else:
                Q_m, U_m = predict_QU(psi_z)
                return np.sqrt(Q_m ** 2.0 + U_m ** 2.0)


    def compatibility(self, measurement_type, **measurement):
        """
        Evaluate the probability of a given astronomical measurement being compatible with this SME universe.
        "measurement_type" is the type of measurement that has been carried out. Supported types are:
            "spectropolarimetry"      :      Polarization angle rotation across the spectrum, rho
            "broadband"               :      Broadband polarization fraction

        The rest of the arguments accepted by the function depend on the type of measurement.

        arguments
            measurement_type        :   "spectropolarimetry" or "broadband"
            ra                      :   Right ascension on the celestial sphere in degrees, 0<=ra<=360
            dec                     :   Declination on the celestial sphere in degrees, -90<=dec<=90
            E                       :   Energy of linearization for "rho". Spectropolarimetry only
            z                       :   Redshift of emission
            psi                     :   Polarization angle of the photon in degrees at energy "E"
                                        for spectropolarimetry. Overall passband polarization for broadband.
                                        Unnecessary in CPT-odd universes (odd mass dimension). Given as the value
                                        at the telescope
            rho                     :   Polarization angle rotation across the spectrum in deg/eV. Spectropolarimetry only
            e_rho                   :   Standard error in "rho" in deg/eV. Spectropolarimetry only
            pol                     :   Measured broadband polarization (0..1). Broadband only
            e_pol                   :   Standard error in "pol". Broadband only
            instrument              :   Initialized Instrument() object, representing the instrument carrying
                                        out the measurement. Broadband only
            pol_z                   :   Assumed polarization at the source. Broadband only. Defaults to the universe default
                                        set at initialization. Unneeded if "e_V" is passed
            e_V                     :   Error in the circular polarization measurement (normalized Stokes V) if available.
                                        When provided, the passed value of "pol_z" is ignored and, instead, it is estimated
                                        my maximizing the compatibility curve with respect to it. Broadband only
            V                       :   Circular polarization measurement corresponding to the error "e_V". If not provided,
                                        0.0 assumed. Broadband only

        returns
            Probability of compatibility
        """
        try:
            measurement['psi']
        except:
            measurement['psi'] = False

        if measurement_type == 'spectropolarimetry':
            # The compatibility of a spectropolarimetric measurement is evaluated as a simple Gaussian
            # distribution given in equation 29 of Kislat 2019...
            rho_sme = self.rho(measurement['ra'], measurement['dec'], measurement['E'], measurement['z'], measurement['psi'])
            return norm.cdf(np.abs(measurement['rho']), loc = np.abs(rho_sme), scale = np.abs(measurement['e_rho']))
        elif measurement_type == 'broadband':
            # ...the broadband case is much harder. First, reset pol_z (polarization at source) to default if not set
            try:
                measurement['pol_z']
            except:
                measurement['pol_z'] = self.pol_z
            # If circular polarization is available, estimate the most conservative value of "pol_z". We do so using the
            # minization routine from SciPy and recursive calls of this function
            if 'e_V' in measurement.keys():
                e_V = measurement['e_V']
                del measurement['e_V']
                del measurement['pol_z']
                if 'V' in measurement.keys():
                    V_val = measurement['V']
                    del measurement['V']
                else:
                    V_val = 0.0
                def func(pol_z):
                    try:
                        pol_z = pol_z[0]
                    except:
                        pass
                    # Somehow the algorithm sometimes attempts to launch itself out of bounds. Prevent that here...
                    if pol_z < 0.0 or pol_z > 1.0:
                        return 99.9
                    linear_prob = self.compatibility(measurement_type, **measurement, pol_z = pol_z)
                    V = self.pol_max(measurement['ra'], measurement['dec'], measurement['instrument_V'], measurement['z'], measurement['psi'], pol_z, return_V = True)[1]
                    circ_prob = np.where(V < V_val, norm.cdf(V, loc = V_val, scale = e_V), 1 - norm.cdf(V, loc = V_val, scale = e_V))
                    return -linear_prob * circ_prob
                # Sample the function at 100 points to determine a good initial guess
                sample_pol_z = np.linspace(0, 1, 100)
                sample_func = np.array(list(map(func, sample_pol_z)))
                result = minimize(func, [sample_pol_z[sample_func == np.min(sample_func)][0]], bounds = [(0, 1.0)])
                return -result.fun
            # Evaluate the maximum polarization fraction predicted by the SME model
            pol_max_sme = self.pol_max(measurement['ra'], measurement['dec'], measurement['instrument'], measurement['z'], measurement['psi'], measurement['pol_z'])

            # If the predicted polarization fraction is 0, no polarization measurement can possibly be compatible
            # with it, so we can return a zero right away
            if pol_max_sme == 0.0:
                return 0.0

            # Remember that SME predicts the MAXIMUM polarization fraction. The measured value is drawn from some
            # underlying distribution and is not necessarily the maximum value. We must estimate the maximum
            # based on the measured value.
            # If the uncertainty in the measured fraction is small, we can pretty much assume that the measured
            # value and the maximum value are identical. In this case, the compatibility equation becomes a simple
            # gaussian:
            if measurement['pol'] / measurement['e_pol'] > 10.0:
                return norm.cdf(pol_max_sme, loc = measurement['pol'], scale = measurement['e_pol'])

            # For less precise data, the shortcut above is unavailable. In this case, we start by defining the
            # probability of measuring some polarization fraction, "pol", given the true value of "pol_true".
            # This is equation 38 of Kislat 2018. ive() is the exponentially scaled modified Bessel functions
            # of the first kind and N is the number of photons received from the source
            pol_prob = lambda N, pol, pol_true: np.e ** (- N * (pol - pol_true) ** 2.0 / 4.0 +
                                                         np.log(N * pol / 2.0) +
                                                         np.log(ive(0, N * pol * pol_true / 2.0)))

            # To use the equation above, we require the number of photons, N. In Kislat 2018, equations 39
            # and 40 give the expected mean value and uncertainty in the measured polarization fraction given
            # the true polarization fraction. We can estimate N from those equations by assuming our measured
            # polarization to be equal to the true polarization
            # First, implement equation 39:
            def pol_expected(pol_true, N):
                Bessel_arg = N * pol_true ** 2.0 / 8.0
                log_result = 0.5 * (np.log(np.pi / 16.0) - np.log(N))
                log_result += (-Bessel_arg)
                Bessel_1 = iv(0, Bessel_arg)
                Bessel_2 = iv(1, Bessel_arg)
                # The Bessel functions evaluate to infinities at very large N. In this limit, the result tends to 
                # pol_true and there is no need to do any further calculations.
                if Bessel_1 == np.inf or Bessel_2 == np.inf:
                    return pol_true
                log_result += np.log(float((4 + N * pol_true ** 2.0)) * float(Bessel_1) + float(N * pol_true ** 2.0) * float(Bessel_2))
                # Same here...
                if not np.isfinite(log_result):
                    return pol_true
                return np.e ** log_result
            # Second, implement equation 40:
            def sigma_expected(pol_true, N):
                return np.sqrt(pol_true ** 2.0 + (4.0 / N) - pol_expected(pol_true, N) ** 2.0)
            # Solve the equations above numerically to get the photon count:
            func = lambda x: sigma_expected(measurement['pol'], x) - measurement['e_pol']
            N = brentq(func, 1.0, 1e12)

            # With the number of photons known, pol_prob() can be integrated to obtain the probability
            func = lambda x: pol_prob(N, x, measurement['pol'])
            return quad(func, 0, pol_max_sme)[0]
        else:
            raise ValueError('Unknown measurement type: {}'.format(measurement_type))

    def compatibility_catalogue(self, catalogue, instruments, log = True):
        """
        Evaluate the probability of a given catalogue of astronomical measurements being compatible with the
        current SME universe

        arguments
            catalogue               :   Object of class astrosme.Catalogue(), storing the catalogue of
                                        astronomical measurements of interest. It is assumed that every
                                        measurement listed in the catalogue is independent
            instruments             :   Dictionary of loaded astrosme.Instrument() objects for every band
                                        listed in the catalogue. The dictionary must be keyed by the machine
                                        name of the band
            log                     :   If True (default), the method will return log likelihood for MCMC
                                        regression routines. Otherwise, the method will return regular
                                        probability

        returns
            Probability of compatibility or log likelihood depending on "log"
        """
        result = 1.0
        for measurement in catalogue.measurements.keys():
            measurement_type = catalogue.measurements[measurement]['type']
            source = catalogue.sources[catalogue.measurements[measurement]['source']]
            measurement = catalogue.measurements[measurement]['measurement']
            if measurement_type == 'broadband':
                if measurement['band'] not in instruments.keys():
                    raise ValueError('Instrument not found for band {}'.format(measurement['band']))
                else:
                    result *= self.compatibility(measurement_type, **measurement, **source, instrument = instruments[measurement['band']])
            else:
                result *= self.compatibility(measurement_type, **measurement, **source)
                self.compatibility(measurement_type, **measurement, **source)
        assert result >= 0
        if not log:
            return result
        else:
            if result != 0:
                result = np.log(result)
            else:
                result = -np.inf
            return result

class Catalogue:
    # Lists of mandatory and optional fields expected in every source
    # z is redshift, ubvri are apparent magnitudes in the corresponding bands. _bib means Simbad bibliographic
    # reference
    __mandatory_fields = ['type', 'ra', 'dec', 'z', 'z_err']
    __optional_fields = ['z_bib', 'u', 'u_err', 'u_bib', 'b', 'b_err', 'b_bib', 'v', 'v_err', 'v_bib', 'r', 'r_err', 'r_bib',
                       'i', 'i_err', 'i_bib']

    # Likewise, lists of mandatory and optional fields for every measurement type. The naming convention is
    # __(mandatory|optional)_fields_[measurement_type]
    __mandatory_fields_spectropolarimetry = ['rho', 'e_rho', 'E']
    __mandatory_fields_broadband = ['pol', 'e_pol', 'band']
    __optional_fields_spectropolarimetry = ['psi', 'e_psi']
    __optional_fields_broadband = ['psi', 'e_psi']

    # Below are functions to validate measurements for every measurement type. The naming convention is
    # __validate_measurement_[measurement_type]. The functions may assume that all mandatory fields are
    # present
    def __validate_measurement_spectropolarimetry(self, measurement):
        """
        Check if a given measurement of type "spectropolarimetry" is valid.

        arguments
            measurement         :   Dictionary containing the measurement with all fields
        
        returns
            True is valid. Error text (string) otherwise
        """
        if measurement['E'] < 0:
            return 'Negative energy of linearization in spectropolarimetry!'
        if ('psi' in measurement.keys()) and ((measurement['psi'] < -90) or (measurement['psi'] > 90)):
            return 'PSI (polarization angle) is outside -90 <= PSI <= 90'
        return True

    def __validate_measurement_broadband(self, measurement):
        """
        Check if a given measurement of type "broadband" is valid.

        arguments
            measurement         :   Dictionary containing the measurement with all fields
        
        returns
            True is valid. Error text (string) otherwise
        """
        if measurement['pol'] < 0 or measurement['pol'] > 1.0:
            return 'POL (polarization fraction) is outside 0<=POL<=1'
        if ('psi' in measurement.keys()) and ((measurement['psi'] < -90) or (measurement['psi'] > 90)):
            return 'PSI (polarization angle) is outside -90 < PSI < 90'
        if measurement['band'] not in self.bands.keys():
            return 'Unknown band {}'.format(measurement['band'])
        return True
    ################## END OF MEASUREMENT VALIDATORS ####################
    
    def __eq__(self, other):
        """
        Check if two objects of type Catalogue() are identical. Most properties can be compared right away
        with the "==" operator. This function must be updated if new properties are added to the class
        (they would be defined in the constructor).

        Bands (self.bands) cannot be compared directly due to NumPy arrays contained within. This function
        typecasts them into lists first.

        arguments
            other             :   Catalogue() object to run the comparison against
        
        returns
            True is identical, False otherwise
        """
        equal = True
        # Compare most properties directly
        equal = equal and (self.name == other.name)
        equal = equal and (self.description == other.description)
        equal = equal and (self.references == other.references)
        equal = equal and (self.sources == other.sources)
        equal = equal and (self.measurements == other.measurements)

        # Compare bands one at a time field-by-field, typecasting all NumPy arrays into lists
        for key in self.bands.keys():
            equal = equal and (key in other.bands.keys()) and (list(self.bands[key]['wl']) == list(other.bands[key]['wl'])) and (list(self.bands[key]['t']) == list(other.bands[key]['t'])) and (self.bands[key]['comment'] == self.bands[key]['comment'])
        return equal

    def __init__(self, filename = False, **args):
        """
        A Catalogue() object stores observational data that may be used to constrain Standard Model Extension (SME)
        realizations. It is merely a storage container with a few additional features to query Simbad for missing
        information and ensure internal consistency. The object stores the following types of data:

            bands        : Transmission profiles of the instruments used in observations. In addition to a grid of
                           wavelengths/transmission fractions, a brief free-text comment can be stored with each
                           band to explain how the profile was obtained
            references   : References to the publications/databases where the data was originally extracted from.
                           Each reference has a machine name that can be associated to each measurement and a brief
                           description, which may be a bibliographic citation. In the case of in-house measurements,
                           this is an opportunity to describe the setup and list observers
            sources      : Sources that have been observed. Specifically, their coordinates (ICRS RA and DEC),
                           redshifts, magnitudes and Simbad-resolvable identifiers. The class is equipped with
                           functions to query all the relevant parameters, although manual input is also possible
            measurements : Measured polarization angles, polarization fractions and polarization drift rates that
                           can be fed directly into a Universe() object to evaluate their compatibility with a
                           given SME model. Each measurement can be linked to a reference, a source and, if
                           necessary, the observation band

        Additionally, the class can save and load the data from files and perform basic data visualization

        arguments
            filename      :   Path to the file storing the catalogue to load in once the object is initialized.
                              Set to False to create a blank catalogue instead (default)
            **args        :   Only relevant when "filename" is not False. Additional arguments that can be passed
                              to the file loader. See load() for details
        """
        self.name = 'Unnamed'
        self.description = 'Add description here'
        self.references = {}
        self.bands = {}
        self.sources = {}
        self.measurements = {}

        if type(filename) != bool:
            self.load(filename, **args)

    def load(self, filename, append = False, override = False, force = False):
        """
        Load a previously saved catalogue into this Catalogue() object. The function will be called automatically
        on initialization if a file path is provided to the constructor (see __init__()). Alternatively, the
        function can be called at any other time to reload the entire dataset or to append the dataset stored in
        a file to the already existing one.
        The files loaded by this function are Python pickles produced by save().

        arguments
            filename      :   Path to the file storing the catalogue to load in
            append        :   True will append the data stored in "filename" to the data already contained within
                              this Catalogue() object. False (default) will erase the current data completely and
                              load the file afresh
            override      :   Only relevant when "append" is True. Determines the behavior of the function if the
                              file contains bands/references/sources with machine names that already exist. If
                              False, the original values (stored in the object) will be kept. If True, the values
                              in the file will be used as replacements. Note that this setting does not affect
                              the name and the description of the catalogue (self.name and self.description).
                              The two will never be overridden as long as "append" is True. Also note that this
                              setting does not affect the measurements. All measurements with clashing machine names
                              will receive newly generated unique machine names. Do not use "append" to combine
                              catalogues with identical measurements!
            force         :   If True (not recommended), load the catalogue as is without validation checks.
                              Otherwise, all incoming data will be validated. Note that the validation is performed
                              during the import and not beforehand. As such, a catalogue that fails validation
                              may be corrupted and must be reloaded
        """
        if not os.path.isfile(filename):
            raise ValueError('Catalogue file {} not found!'.format(filename))
        f = open(filename, 'rb')
        loaded = pickle.load(f)
        f.close()

        # We reload the name and description ONLY when the catalogue is being loaded afresh into an empty
        # object. I.e., only when "append" is False
        if not append:
            self.name = loaded['name']
            self.description = loaded['description']

        # If not appending, erase the current dataset completely
        if not append:
            self.references = {}
            self.bands = {}
            self.sources = {}
            self.measurements = {}

        # We now load all references, bands and sources by feeding the data in the file to the corresponding
        # addition function
        for reference in loaded['references'].keys():
            if reference in self.references.keys():
                if override:
                    del self.references[reference]
                else:
                    continue
            self.add_reference(reference, loaded['references'][reference], force = force)
        for band in loaded['bands'].keys():
            if band in self.bands.keys():
                if override:
                    del self.bands[band]
                else:
                    continue
            self.add_band(band, **loaded['bands'][band], force = force)
        for source in loaded['sources'].keys():
            if source in self.sources.keys():
                if override:
                    del self.sources[source]
                else:
                    continue
            self.add_source(source, **loaded['sources'][source], force = force, query_simbad = False)

        # Measurements are slightly different. When a clash is found, we regenerate the machine name regardless
        # of the value of "override" as described in the docstring
        for measurement in loaded['measurements'].keys():
            if measurement in self.measurements.keys():
                self.add_measurement(loaded['measurements'][measurement]['source'], loaded['measurements'][measurement]['reference'], loaded['measurements'][measurement]['type'], loaded['measurements'][measurement]['measurement'], force = force)
            else:
                self.add_measurement(loaded['measurements'][measurement]['source'], loaded['measurements'][measurement]['reference'], loaded['measurements'][measurement]['type'], loaded['measurements'][measurement]['measurement'], machine_name = measurement, force = force)

    def save(self, filename):
        """
        Dump the data stored in the current state of the catalogue into a Pickle file that can be later loaded
        with load() or by the constructor when creating a new Catalogue() object.

        arguments
            filename      :   Path to the file to store the catalogue in
        """
        to_save = {}
        to_save['name'] = self.name
        to_save['description'] = self.description
        to_save['references'] = self.references
        to_save['bands'] = self.bands
        to_save['sources'] = self.sources
        to_save['measurements'] = self.measurements
        f = open(filename, 'wb')
        pickle.dump(to_save, f)
        f.close()

    def add_reference(self, machine_name, body, force = False):
        """
        Add a new reference to the catalogue. A reference will have a machine name that will be associated with
        measurements and a brief description (body). For external references, it is recommended to use the body
        to store a proper bibliographic citation. For in-house measurements, the body may describe the setup
        used and give credit to observers.

        arguments
            machine_name      :   Machine name of the reference. While we do not impose any restrictions on what
                                  the machine name must be, short and informative combinations of basic characters
                                  are preferred. E.g. "Smith+2009", "GAIA_DR1", "my_backyard_polorimeter"
            body              :   More detailed description of the reference (body)
            force             :   If True (not recommended), the reference will be added without validation
        """
        if machine_name in self.references.keys() and (not force):
            raise ValueError('Reference {} already exists!'.format(machine_name))
        self.references[machine_name] = body

    def remove_reference(self, machine_name, force = False):
        """
        Remove a reference by its machine name. When removing references, be sure to check that no measurements
        are linked to them. No automated checks will run!

        arguments
            machine_name      :   Machine name of the reference to be removed
            force             :   If False, an error will be thrown when a non-existent reference is being removed
        """
        if machine_name not in self.references.keys() and (not force):
            raise ValueError('Reference {} not found!'.format(machine_name))
        if machine_name in self.references.keys():
            del self.references[machine_name]

    def validate_band(self, band):
        """
        Validate a band to make sure that its data format is consistent with the structure of the catalogue

        arguments
            band              :   The band to be validated, as a Python dictionary. The expected fields are "wl"
                                  for the wavelength grid in nm (NumPy array), "t" for the corresponding
                                  transmission fractions between 0 and 1 and "comment" for a brief human-friendly
                                  description of the transmission band

        returns
            True if valid. Error text (string) otherwise
        """
        wl, t = (band['wl'], band['t'])
        if np.max(t) > 1.0:
            return 'Transmission exceeds 1.0'
        if np.min(t) < 0.0:
            return 'Negative transmission'
        if t[wl == np.min(wl)] > 1e-2:
            return '> 1% transmission leakage on the blue end'
        if t[wl == np.max(wl)] > 1e-2:
            return '> 1% transmission leakage on the red end'
        return True

    def add_band(self, machine_name, wl, t, comment = '', force = False):
        """
        Add a new observation band to the catalogue. The band dictionary will contain a grid of wavelengths and
        transmission fractions of the instrument as well as a brief human-friendly comment describing it.
        In the comment, it is recommended to quote the origin of the transmission profile or describe how it was
        sampled for custom filters

        arguments
            machine_name      :   Machine name of the band. While we do not impose any restrictions on what
                                  the machine name must be, short and informative combinations of basic characters
                                  are preferred. E.g. "Johnson V", "SDSS_r"
            wl                :   Grid of wavelengths (1D) over which the transmission profile is defined. All
                                  values must be in nm
            t                 :   Transmission fractions corresponding to the wavelengths in "wl". All values
                                  must be dimensionless between 0 and 1. The profile must definitely vanish
                                  on both blue and red sides (i.e. no "step-up" or "step-down" filters)
            comment           :   Brief description of the instrument
            force             :   If True (not recommended), the band will be added without validation
        """
        if machine_name in self.bands.keys() and (not force):
            raise ValueError('Band {} already exists!'.format(machine_name))
        # Sort the profile in the order of increasing wavelengths
        L = sorted(zip(wl * 1.0, t * 1.0), key=operator.itemgetter(0))
        wl, t = zip(*L)

        # Compile a band dictionary and validate it (if necessary)
        band = {'wl': np.array(wl).astype(float), 't': np.array(t).astype(float), 'comment': comment}
        validated = self.validate_band(band)
        if (not force) and (type(validated) != bool and validated != True):
            raise ValueError('{}: {}'.format(machine_name, validated))

        self.bands[machine_name] = band

    def remove_band(self, machine_name, force = False):
        """
        Remove a band by its machine name. When removing bands, be sure to check that no measurements are linked
        to them. No automated checks will run!

        arguments
            machine_name      :   Machine name of the band to be removed
            force             :   If False, an error will be thrown when a non-existent band is being removed
        """
        if machine_name not in self.bands.keys() and (not force):
            raise ValueError('Band {} not found!'.format(machine_name))
        if machine_name in self.bands.keys():
            del self.bands[machine_name]

    def band_summary(self):
        """
        Produce a summary table for all avaialble observation bands. The summary will include names and
        descriptions of all bands, wavelength ranges, FWHM of the transmission profiles as well as central
        wavelengths


        returns
            AstroPy data table with the following fields:
                name         :        Machine name of the band
                comment      :        Description of the band
                min_wl       :        Blue-most wavelength of the profile without trimming (i.e. no adjustment
                                      is made for profiles that have leading or trailing zeroes) in nm
                max_wl       :        Similarly, the red-most wavelength of the profile in nm
                FWHM         :        Full-Width-at-Half-Maximum of the profile in nm
                centre       :        Central wavelength of the profile (average wavelength weighted by
                                      transmission) in nm
                n            :        Number of samples in the profile
        """
        summary = OrderedDict([('name', []), ('comment', []), ('min_wl', []), ('max_wl', []), ('FWHM', []),
                               ('centre', []), ('n', [])])
        for band in self.bands.keys():
            summary['name'] += [band]
            band = self.bands[band]
            summary['comment'] += [band['comment']]
            wl, t = (band['wl'], band['t'])
            summary['min_wl'] += [np.min(wl)]; summary['max_wl'] += [np.max(wl)]
            HM = np.max(t) / 2.0
            HM_wl = wl[t == np.max(t)][0]
            HM_left = np.max(wl[(t < HM) & (wl < HM_wl)]); HM_right = np.min(wl[(t < HM) & (wl > HM_wl)])
            summary['FWHM'] += [HM_right - HM_left]
            summary['centre'] += [np.sum(wl * t) / np.sum(t)]
            summary['n'] += [len(wl)]

        return Table(list(summary.values()), names = list(summary.keys()), meta = {'name': 'Band summary'})

    def band_plot(self, bands = False, plot_args = {'lw': 2}, fig_args = {'figsize': [12, 9]}):
        """
        Render a plot of avaialble transmission profiles using matplotlib.pyplot

        arguments
            bands             :   List of machine names of the bands to be plotted. Defaults to False, meaning
                                  "all avaialble bands"
            plot_args         :   Dictionary of keyword arguments to be passed to pyplot.plot(). By default,
                                  sets the line width (lw) to "2"
            fig_args          :   Dictionary of keyword arguments to be passed to pyplot.figure(). By default,
                                  sets the figure size (figsize) to [12, 9]
        """
        plt.figure(**fig_args)
        for band_name in self.bands.keys():
            if (type(bands) == bool) or (band_name in bands):
                band = self.bands[band_name]
                plt.plot(band['wl'], band['t'], label = band_name, **plot_args)

    def __is_positive_number(self, s):
        """
        Check if a given entity can be interpreted as a positive number. This function was designed to parse
        Simbad query output.

        arguments
            s                 :   Entity to be checked for being a positive number

        returns
            True if "s" is a positive number, False otherwise
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                float(s)
                if str(s) == '--':
                    return False
                return float(s) >= 0
            except ValueError:
                return False

    def guess_redshift_error(self, z):
        """
        Suggest the approximate upper limit on the error in a given measured redshift value based on the value
        itself. This function allows to make conservative estimates of the precision of redshift values posted
        on Simbad that do not have associated errors. The function relies on an empirical relationship derived
        by fitting polynomials to the upper percentiles of many Simbad redshifts

        At small redshifts (z<0.2) or so, the function is known to overestimate the error by a significant
        factor. As a partial remedy, it is coded to return "z" whenever the empirical relationship suggests an
        error that is larger than "z". This is so that nearby sources could be imported into the catalogue
        automatically, despite having no useful information about the redshift error

        arguments
            z                 :   Simbad redshift requiring an error estimate

        returns
            Estimated redshift error (conservative upper bound)
        """
        log = 0.23 - 3.2 * z + 1.2 * z ** 2.0 - 0.18 * z ** 3.0
        result = (10.0 ** log) * z
        if result > z:
            return z
        else:
            return result

    def query_simbad(self, identifier):
        """
        Query Simbad for a given simbad-resolvable identifier for redshifts, apparent magnitudes and coordinates

        arguments
            identifier      :        Simbad-resolvable identifier

        returns
            Dictionary of successfully retrieved values potentially containing the keys below. All values that
            could not be retrieved will not be present in the dictionary.
                simbad_id       :        Main Simbad identifier
                type            :        Object type
                ra              :        ICRS right ascension in degrees
                dec             :        ICRS declination in degrees
                z               :        Redshift
                z_err           :        Redshift error. If not avaialble on Simbad, will be guessed using
                                         guess_redshift_error()
                z_bib           :        Simbad bibcode of the redshift value. If the error was guessed, the
                                         bibcode will be prefixed with "NE/"
                (u|b|v|r|i)     :        Apparent magnitude in the (u|b|v|r|i) band
                (u|b|v|r|i)_err :        Error in the corresponding apparent magnitude
                (u|b|v|r|i)_bib :        Simbad bibcode of the corresponding apprent magnitude
        """
        def iff(condition, true_out, false_out):
            if condition:
                return true_out
            else:
                return false_out

        # Note that Simbad considers radial velocity and redshift values interchangeable (i.e., only one is
        # actually stored in the database, while the other is generated from it). We can force Simbad to give us
        # redshifts (regardless of whether it is the redshift or the radial velocity that is in the database) by
        # querying Z_VALUE. However, I am not aware of any way to make Simbad return redshift errors, if the
        # value stored in the database is a radial velocity. Hence, we need to know which value is in the database
        # (RVZ_TYPE is either "c" for radial velocity or "z" for redshift), retrieve the raw value from the
        # database (RVZ_RADVEL and RVZ_ERROR) and, if necessary, do the conversion ourselves.
        # By radial velocity ("c"), Simbad means zc. There exists a third data type denoted by "v" that represents
        # the regular radial velocity in km/s
        retrieved = {}
        
        # Speed of light in km/s
        c = spc.c / 1000.0

        # Ignore networking errors from Simbad
        warnings.filterwarnings(action = 'ignore', message = 'unclosed', category = ResourceWarning)

        customSimbad = Simbad()
        customSimbad.add_votable_fields('z_value', 'rvz_radvel', 'rvz_error', 'rvz_type', 'rvz_bibcode', 'otype')
        # We try this twice in case of network/server issues
        try:
            result = customSimbad.query_object(identifier)
        except:
            result = customSimbad.query_object(identifier)
        if result is None:
            warnings.warn('{} not found on SIMBAD'.format(identifier))
            return retrieved
        else:
            result = result[0]
            retrieved['simbad_id'] = result['MAIN_ID']
            retrieved['type'] = result['OTYPE']
            coord = SkyCoord(result['RA'] + " " + result['DEC'], unit=(u.hour, u.deg), frame='icrs')
            retrieved['ra'] = coord.ra.deg
            retrieved['dec'] = coord.dec.deg
        z_bibcode = ''

        if self.__is_positive_number(result['Z_VALUE']) or self.__is_positive_number(result['RVZ_RADVEL']):
            # Based on the format of the redshift value, perform the necessary conversion
            if result['RVZ_TYPE'] == 'z':
                z = result['Z_VALUE']          # Redshift given as is
            elif result['RVZ_TYPE'] == 'c':
                z = result['RVZ_RADVEL'] / c   # Redshift given as cz
            elif result['RVZ_TYPE'] == 'v':
                # Redshift given as radial velocity in km/s. Use the relativistic redshift formula to recover z
                v_to_z = lambda v: np.sqrt((1 + v / c) / (1 - v / c)) - 1
                z = v_to_z(result['RVZ_RADVEL'])
            else:
                raise ValueError('Unknown SIMBAD redshift unit {}'.format(result['RVZ_TYPE']))
        else:
            z = False
            warnings.warn('{}: Redshift not found or negative'.format(identifier))
        if self.__is_positive_number(result['Z_VALUE']) and self.__is_positive_number(result['RVZ_ERROR']) and result['RVZ_ERROR'] != 0:
            # Run the same conversions with the redshift error
            if result['RVZ_TYPE'] == 'z':
                ze = result['RVZ_ERROR']
            elif result['RVZ_TYPE'] == 'c':
                ze = result['RVZ_ERROR'] / c
            elif result['RVZ_TYPE'] == 'v':
                # Use the error propagation formula
                ze = np.abs(result['RVZ_ERROR'] * spd(v_to_z, result['RVZ_RADVEL']))
        else:
            if type(z) == bool:
                ze = False
            else:
                # If no redshift value is avaialble, guess it
                ze = self.guess_redshift_error(z)
                z_bibcode = 'NE/'
        if result['RVZ_BIBCODE'] != '':
            z_bibcode += result['RVZ_BIBCODE']
        else:
            z_bibcode = False
        retrieved['z'] = z; retrieved['z_err'] = ze; retrieved['z_bib'] = z_bibcode

        # Retrieve magnitudes
        customSimbad = Simbad()
        bands = ['U', 'B', 'V', 'R', 'I']
        for band in bands:
            customSimbad.add_votable_fields('flux(' + band + ')', 'flux_unit(' + band + ')',
                                            'flux_system(' + band + ')', 'flux_error(' + band + ')', 'flux_bibcode(' + band + ')')
        try:
            result = customSimbad.query_object(identifier)
        except:
            result = customSimbad.query_object(identifier)
        if result is None:
            warnings.warn('{} not found on SIMBAD'.format(identifier))
            return retrieved
        else:
            result = result[0]
        for i, band in enumerate(bands):
            retrieved[band.lower()] = iff(result['FLUX_' + band], result['FLUX_' + band], False)
            retrieved[band.lower() + '_err'] = iff(result['FLUX_ERROR_' + band], result['FLUX_ERROR_' + band], False)
            retrieved[band.lower() + '_bib'] = iff(result['FLUX_BIBCODE_' + band], result['FLUX_BIBCODE_' + band], False)

        for value in list(retrieved.keys()):
            if (type(retrieved[value]) == bool) and (not retrieved[value]):
                del retrieved[value]
        return retrieved

    def validate_source(self, source):
        """
        Validate a source to make sure that its data format is consistent with the structure of the catalogue

        arguments
            source            :   The source to be validated, as a Python dictionary. The expected fields are
                                  stored in __mandatory_fields and __optional_fields

        returns
            True if valid. Error text (string) otherwise
        """
        mandatory_fields = self.__mandatory_fields
        optional_fields = self.__optional_fields
        for field in mandatory_fields:
            if field not in source.keys():
                return 'Mandatory field {} not found'.format(field)
        for field in source.keys():
            if field not in mandatory_fields and field not in optional_fields:
                return 'Unknown field {}'.format(field)
        if len(source['type']) < 1:
            return 'Object type is empty'
        if source['ra'] < 0.0 or source['ra'] > 360.0:
            return 'RA is outside of 0<=RA<=360'
        if source['dec'] < -90.0 or source['dec'] > 90.0:
            return 'DEC is outside of -90<=DEC<=90'
        if source['z'] <= 0.0:
            return 'The source is blue-shifted; inappropriate for SME'
        if source['z_err'] > source['z']:
            return 'The error in z exceeds the value'
        return True

    def add_source(self, identifier, query_simbad = True, force = False, **source):
        """
        Add a new source to the catalogue. The source dictionary contains basic data, celestial coordinates,
        redshift and apparent magnitudes. The source dictionary must not contain any polarimetric data, as they
        will be stored as measurements instead.
        In order to run, this function only needs a Simbad-resolvable identifier of the source and the rest of the
        necessary data will be queried from Simbad. While manual input is provided as an option, it is recommended
        to avoid using it unless absolutely necessary (e.g. when the object of interest is not listed on Simbad).

        arguments
            identifier        :   Simbad-resolvable identifier, which will be used as the machine name of the
                                  source. If the identifier is not the main Simbad identifier, it will be replaced
                                  with one. For manual input, choose any descriptive machine name
            query_simbad      :   Set to True (default) to query Simbad for necessary values. Alternatively,
                                  input the data manually (not recommended) in **source
            force             :   If True (not recommended), the source will be added without validation
            **source          :   Source parameters for manual input. If "query_simbad" is True, all values that
                                  can be retrieved from Simbad will be replaced. See query_simbad() for a list
                                  of expected parameters and their units

        returns
            Machine name, under which the new source will be stored. If "query_simbad" is False, it will be
            identical to "identifier". Otherwise, it may be replaced with the main Simbad identifier, unless
            "identifier" already happens to be such
        """
        if query_simbad:
            result = self.query_simbad(identifier)
            for value in result.keys():
                if value != 'simbad_id':
                    source[value] = result[value]
            if 'simbad_id' in result.keys():
                identifier = result['simbad_id']
        if identifier in self.sources.keys() and (not force):
            raise ValueError('Source {} already exists!'.format(identifier))
        validated = self.validate_source(source)
        if (not force) and (type(validated) != bool and validated != True):
            raise ValueError('{}: {}'.format(identifier, validated))

        self.sources[identifier] = source
        return identifier

    def remove_source(self, identifier, force = False):
        """
        Remove a source by its machine name (for Simbad sources, it will be their main Simbad identifier). When
        removing sources, be sure to check that no measurements are linked to them. No automated checks will run!

        arguments
            identifier        :   Machine name of the source to be removed
            force             :   If False, an error will be thrown when a non-existent source is being removed
        """
        if identifier not in self.sources.keys() and (not force):
            raise ValueError('Source {} not found!'.format(identifier))
        if identifier in self.sources.keys():
            del self.sources[identifier]

    def source_summary(self, render_table = True):
        """
        Produce a summary of all available sources as a table. All avaialble fields will be displayed, which
        includes all mandatory fields (object types, names, coordinates, redshifts) and all available optional
        fields (magnitudes, Simbad bibcodes etc). All missing values will be replaced with False.
        The default output format is an AstroPy table, which may typecast some of the missing values (False) to
        zeroes. If the original types are preferred instead, Python dictionary output is provided as an option

        arguments
            render_table      :   If True (default), an AstroPy table will be rendered, which may show missing
                                  values as zeroes. If False, a dictionary will be provided instead keyed by
                                  column names. In such case, all missing values will be set to False

        returns
            AstroPy table or dictionary (depending on "render_table") with a tabular summary of all
            avaialble sources
        """
        mandatory_fields = self.__mandatory_fields
        optional_fields = self.__optional_fields
        summary = OrderedDict({'identifier': []})
        for field in mandatory_fields + optional_fields:
            summary[field] = []

        for source in self.sources.keys():
            summary['identifier'] += [source]
            for field in mandatory_fields + optional_fields:
                if field in self.sources[source].keys():
                    summary[field] += [self.sources[source][field]]
                else:
                    summary[field] += [False]

        for field in mandatory_fields + optional_fields:
            if np.all(list(map(lambda x: type(x) == bool and x == False, summary[field]))):
                del summary[field]
        if render_table:
            return Table(list(summary.values()), names = list(summary.keys()), meta = {'name': 'Source summary'})
        else:
            return summary

    def source_plot(self, galactic = False, projection = 'mollweide', mask = False, marker = False, size = False, color = False, plot_args = {}, fig_args = {'figsize': [12, 9]}):
        """
        Plot a starchart, showing all available sources. The function provides conditional formating, filtering
        and various sky projections as options

        arguments
            galactic          :   If False (default), the sources will be plotted in their original ICRS equatorial
                                  coordinates (RA and Dec). Otherwise, all coordinates will be converted to
                                  galactic (galactic latitude (b) and longitude(l))
            projection        :   MatPlotLib projection to be rendered. Defaults to "mollweide"
            mask              :   Callable object accepting a single parameter that is a dictionary containing
                                  all parameters of a given source (see query_simbad() for a list with
                                  descriptions). If provided, only those sources will be plotted for which mask()
                                  returns True. Defaults to False, meaning "no filtering"
            marker            :   Similar to "mask", returns the marker style for each source. If False, all
                                  markers are circles
            size              :   Similar to "mask", returns the marker size for each source. If False, all markers
                                  are set to 50
            color             :   Similar to "mask", returns the marker colour for each source. If False, all
                                  markers are black
            plot_args         :   Dictionary of keyword arguments to be passed to pyplot.plot(). By default,
                                  no additional arguments are passed
            fig_args          :   Dictionary of keyword arguments to be passed to pyplot.figure(). By default,
                                  sets the figure size (figsize) to [12, 9]

        returns
            fig               :   MatPlotLib figure object associated with the plot
            ax                :   MatPlotLib axes object associated with the plot
        """
        summary = self.source_summary(render_table = False)
        # Convert all coordinates to radians
        ra = Angle(summary['ra'], unit=u.degree).wrap_at(180 * u.degree).radian
        dec = Angle(summary['dec'], unit=u.degree).radian

        if galactic:
            gc = SkyCoord(ra=Angle(ra, unit=u.radian), dec=Angle(dec, unit=u.radian), frame='icrs')
            ra = gc.galactic.l.wrap_at(180 * u.degree).radian
            dec = gc.galactic.b.radian

        # Cartesian projections require degrees
        if projection is None:
            ra = (np.degrees(ra) + 360) % 360
            dec = np.degrees(dec)

        fig = plt.figure(**fig_args)
        ax = fig.add_subplot(111, projection = projection)
        if (type(mask) == bool) and (mask == False):
            mask = np.full([len(ra)], True)
        else:
            mask = np.array(list(map(mask, self.sources.values())))
        if (type(marker) == bool) and (marker == False):
            marker = np.full([len(ra[mask])], 'o')
        else:
            marker = np.array(list(map(marker, np.array(list(self.sources.values()))[mask])))
        if (type(size) == bool) and (size == False):
            size = np.full([len(ra[mask])], 50)
        else:
            size = np.array(list(map(size, np.array(list(self.sources.values()))[mask])))
        if (type(color) == bool) and (color == False):
            color = np.full([len(ra[mask])], 'k')
        else:
            color = np.array(list(map(color, np.array(list(self.sources.values()))[mask])))
        for current_marker in list(set(marker)):
            c = current_marker == np.array(marker)
            ax.scatter(ra[mask][c], dec[mask][c], s = size[c], marker = current_marker, c = color[c], **plot_args)
        if projection == 'polar':
            plt.thetagrids()
        else:
            plt.grid()
        return fig, ax

    def validate_measurement(self, measurement_type, measurement):
        """
        Validate a measurement to make sure that its data format is consistent with the structure of the catalogue

        arguments
            measurement_type  :   "spectropolarimetry" or "broadband"
            measurement       :   Dictionary containing the measurement

        returns
            True if valid. Error text (string) otherwise
        """
        mandatory_fields = self.__getattribute__('_Catalogue__mandatory_fields_' + measurement_type)
        optional_fields = self.__getattribute__('_Catalogue__optional_fields_' + measurement_type)
        validator = self.__getattribute__('_Catalogue__validate_measurement_' + measurement_type)

        for field in mandatory_fields:
            if field not in measurement.keys():
                return 'Mandatory field {} not found!'.format(field)
        for field in measurement.keys():
            if field not in optional_fields + mandatory_fields:
                return 'Unknown field {}!'.format(field)
        return validator(measurement)

    def add_measurement(self, source, reference, measurement_type, measurement, machine_name = False, force = False):
        """
        Add a new measurement to the catalogue. Machine names can either be passed manually or generated in a
        unique way. Each measurement is associated with a source and a literature reference. See add_source()
        and add_reference() for details.

        arguments
            source            :   Machine name of the source this measurement belongs to. For Simbad sources,
                                  this will be the main Simbad identifier. Use source_summary() to see all
                                  avaialble sources
            reference         :   Machine name of the reference, containing the origin of this measurement
            measurement_type  :   "spectropolarimetry" or "broadband"
            measurement       :   Dictionary, containing the parameters of the measurement. The expected fields
                                  depend on the measurement type. For guidance, see Universe.compatibility()
            machine_name      :   Machine name of the measurement. Defaults to "False", which triggers
                                  autogeneration
            force             :   If True (not recommended), the measurement will be added without validation

        returns
            Machine name, under which the new measurement will be stored. If provided in "machine_name", it will
            be identical to the given value. Otherwise, it will be an autogenerated value
        """
        validated = self.validate_measurement(measurement_type, measurement)
        if (not force) and (type(validated) != bool and validated != True):
            raise ValueError('{}, {}: {}'.format(source, measurement_type, validated))
        if (not force) and (source not in self.sources.keys()):
            raise ValueError('{}, {}: Unknown source!'.format(source, measurement_type))
        if (not force) and (reference not in self.references.keys()):
            raise ValueError('{}, {}: Unknown reference {}!'.format(source, measurement_type, reference))

        # Generating a unique machine name here. This will be a sequence of B, C, D, E, ... BA, BB etc
        # I.e. a 26-base serial number where A represents 0 and Z represents 26. All leading A's are trimmed
        # which is why we start with a "B"
        if not machine_name:
            index = 0
            while True:
                index += 1
                machine_name = ''
                pos = 0
                num = index
                while num != 0:
                    pos += 1
                    digit = num % 26
                    num -= digit
                    num = int(num / 26)
                    machine_name = chr(digit + 65) + machine_name
                if machine_name not in self.measurements.keys():
                    break

        # Generate a measurement dictionary
        measurement = {
            'type': measurement_type,
            'source': source,
            'reference': reference,
            'measurement': measurement,
        }

        if machine_name in self.measurements.keys() and (not force):
            raise ValueError('Measurement {} already exists!'.format(machine_name))
        self.measurements[machine_name] = measurement
        return machine_name

    def remove_measurement(self, machine_name = False, source = False, reference = False, force = False):
        """
        Remove a measurement by its machine name, source name or reference. Multiple references can be removed
        in a single call. Note that if multiple removal conditions are provided (e.g. both source and
        reference), they will be stacked using AND. To stack using OR, call the function multiple times

        arguments
            machine_name      :   Machine name of the source to be removed. May be a list of multiple values.
                                  Defaults to False, meaning "do not remove by machine names"
            source            :   Source machine name to remove all linked measurements. May be a list of multiple
                                  values. Defaults to False, meaning "do not remove by sources"
            reference         :   Reference machine name to remove all linked measurements. May be a list of
                                  multiple values. Defaults to False, meaning "do not remove by references"
            force             :   If True (not recommended), no errors will be thrown when removing non-existent
                                  measurements or when removing by non-existent sources and references

        returns
            Total number of measurements removed
        """
        if type(machine_name) == str:
            machine_name = [machine_name]
        if type(source) == str:
            source = [source]
        if type(reference) == str:
            reference = [reference]

        if not force:
            if type(machine_name) != bool:
                test = list(map(lambda x: x not in self.measurements.keys(), machine_name))
                if sum(test):
                    raise ValueError('Uknown measurement(s): {}'.format(np.array(machine_name)[test]))
            if type(source) != bool:
                test = list(map(lambda x: x not in self.sources.keys(), source))
                if sum(test):
                    raise ValueError('Uknown source(s): {}'.format(np.array(source)[test]))
            if type(reference) != bool:
                test = list(map(lambda x: x not in self.references.keys(), reference))
                if sum(test):
                    raise ValueError('Uknown reference(s): {}'.format(np.array(reference)[test]))

        deletion_count = 0
        def measurement_filter(measurement):
            condition_1 = (type(machine_name) == bool) or (measurement[0] in machine_name)
            condition_2 = (type(source) == bool) or (measurement[1]['source'] in source)
            condition_3 = (type(reference) == bool) or (measurement[1]['reference'] in reference)
            return condition_1 and condition_2 and condition_3
        for current_machine_name in list(self.measurements.keys()):
            if measurement_filter((current_machine_name, self.measurements[current_machine_name])):
                deletion_count += 1
                del self.measurements[current_machine_name]
        return deletion_count

    def measurement_table(self, fields = [], query = False, force = False):
        """
        Retrieve all measurements as a table. The measurements will be returned as a dictionary keyed by
        measurement properties (such as psi, rho etc). Each element is a list, such as identical indices refer
        to the same measurement

        arguments
            fields            :   List of columns to generate. Could be any valid measurement fields. The full
                                  list of available fields depends on the measurement type. For details, see
                                  Universe.compatibility(). In addition to those, one may also request "source",
                                  "reference", "type" or "machine_name" as well as all the fields associated with
                                  the source (e.g. "ra", "dec" etc, see query_simbad())
            query             :   Callable object of the form func(measurement, type, source, reference), where
                                  "measurement" is a dictionary storing a measurement, "type" is the measurement
                                  type and "source" and "reference" are the machine names of the linked source
                                  and reference respectively. Only those measurements, for which this callable
                                  returns True will be provided in the output table. Defaults to False, meaning
                                  "no filtering"
            force             :   Determines the reaction when a field listed in "fields" is not available for
                                  a given measurement. If True, the measurement will be excluded from the table.
                                  Otherwise, an exception will be thrown

        returns
            Table as a dictionary of queried fields, containing selected measurements
        """
        output = {}
        for field in fields:
            output[field] = []
        index = 0
        for measurement in self.measurements.keys():
            if type(query) != bool:
                if not query(**self.measurements[measurement]):
                    continue
            bad_measurement = False
            for field in fields:
                if field in self.measurements[measurement]['measurement']:
                    output[field] += [self.measurements[measurement]['measurement'][field]]
                elif field in self.sources[self.measurements[measurement]['source']]:
                    output[field] += [self.sources[self.measurements[measurement]['source']][field]]
                elif field in self.measurements[measurement]:
                    output[field] += [self.measurements[measurement][field]]
                elif field == 'machine_name':
                    output[field] += [measurement]
                else:
                    if not force:
                        raise ValueError('Measurement {} does not have field {}'.format(measurement, field))
                    else:
                        bad_measurement = True
                        break
            if bad_measurement:
                for field in fields:
                    try:
                        del output[field][index]
                    except:
                        pass
                continue
            index += 1
        for field in fields:
            output[field] = np.array(output[field])

        return output


class Dust:
    def __init__(self, x_axis, y_axis, z_axis, B_x, B_y, B_z, n, telescope, source, north = False, flux_error = 0.001):
        """
        Dust() simulates propagation of starlight through dust clouds immersed in the galactic magnetic field.
        If the field is sufficiently strong, asymetric dust grains will align themselves with it and proceed
        to preferentially scatter starlight of certain polarizations. Hence, polarized starlight can be used
        as a probe of the galactic magnetic field as well as interstellar dust distribution.

        This code is primarily based on a theoretical paper by Davis and Greenstein in 1951 (access online at
        https://ui.adsabs.harvard.edu/abs/1951ApJ...114..206D/abstract). We will refer to the paper as DG51.
        The code requires a data cube, containing magnetic field components and dust number density at all
        points. The code relies on the following assumptions:

        1) All dust grains are much smaller than the wavelength of starlight. For the Milky Way, the most dominant
           grain size is believed to be around 300 nm, which is slightly smaller than visible light and much
           smaller than near infrared. Unfortunately, dust grains of larger sizes are also known to exist.
        2) All dust grains are spheroids, parametrized by their aspect ratio that is fixed to a constant value
           for each simulation.
        3) All dust grains are dielectric and have a fixed index of refraction. Physically, this means that
           they only scatter and do not absorb radiation. 
        4) The magnetic field is strong enough to fully align all grains everywhere with no precession.
        5) Stars carry no intrinsic polarization (i.e. Q=U=0 at emission)
        6) Neither stars no the interstellar medium are compatible with circular polarization (i.e. V=0 always)
        7) Interstellar dust is optically thin along the line of sight
        9) All dust grains are paramagnetic

        arguments
            x_axis        :   Vector of all x-coordinate values, defining the first dimension of the data cube.
                              The unit is pc. The origin can be placed anywhere and negative values are accepted
            y_axis        :   Similar to "x_axis" for the second dimension of the data cube
            z_axis        :   Similar to "x_axis" for the third dimension of the data cube
            B_x           :   x-component of the magnetic field at all points of the data cube. Must be a 3D
                              array of shape (len(x_axis), len(y_axis), len(z_axis)). The units are arbitrary,
                              since the strong magnetic field limit is assumed
            B_y           :   Similar to "B_x" for the y-component of the magnetic field
            B_z           :   Similar to "B_x" for the z-component of the magnetic field
            n             :   Dust number density at all points of the data cube. Must be a 3D array of the same
                              shape as "B_x". The unit is cm^-3
            telescope     :   1D array of three numbers: position of the observer in the data cube in pc
            source        :   1D array of three numbers: position of the star, whose starlight polarization is
                              being simulated, in pc
            north         :   Direction vector towards celestial north as observed from the origin of the data
                              cube. The vector will be normalized automatically. The y-axis (self.e3) of the
                              observer coordinates will be aligned with the projection of this vector onto the
                              celestial plane. If set to False (default) or if the vector is parallel to the line
                              of sight, the observer basis will be chosen arbitrarily. In the latter case, a warning
                              will be issued
            flux_error    :   Target fractional error in flux when solving radiative transfer. The error is
                              calculated at every step. When errors larger than this threshold are found, the
                              step is repeated with a smaller step size. Likewise, smaller errors lead to an 
                              increase in step size. Defaults to 0.001, i.e. 1% error. Reduce for accuracy, 
                              increase for performance
        """
        # Check that the data cube is of correct shape
        for to_verify in [B_x, B_y, B_z, n]:
            if np.shape(to_verify) != (len(x_axis), len(y_axis), len(z_axis)):
                raise ValueError('The B-field and grain density (B_x, B_y, B_z, n) must have the same dimensions'/
                                 ' as the provided coordinate axes!')

        # Save all arguments locally
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.B_x = B_x
        self.B_y = B_y
        self.B_z = B_z
        self.n = n
        self.telescope = np.array(telescope).astype(float)
        self.source = np.array(source).astype(float)
        self.flux_error = flux_error

        # We now calculate a set of unit vectors for the observer basis. The z-vector (e3) is obvious, as it must be
        # pointed along the line of sight from the telescope to the source
        self.e3 = self.norm(self.source - self.telescope)
        # We use cross-products to get some two other arbitrary unit vectors that are perpendicular to each other
        # and e3. This will give us a valid basis for the observer
        if list(np.abs(np.round(self.e3, 5))) == [0, 0, 1.0]:
            self.e1 = self.norm(np.cross([0, 1, 0], self.e3))
        else:
            self.e1 = self.norm(np.cross([0, 0, 1], self.e3))
        self.e2 = self.norm(np.cross(self.e3, self.e1))
        # Now we have an arbitrary observer basis. If however the user provided a direction of the celestial North,
        # we can rotate the said basis so that the y-axis (e2) points towards said North and the basis is no longer
        # arbitrary
        if type(north) != bool:
            if np.allclose(np.cross(north, self.e3), [0, 0, 0]):
                warnings.warn('Line of sight coincides with North; arbitrary observer basis selected!')
            else:
                due_north = self.norm(np.array(north)) - self.e3 # Direction due North in global 3D coordinates
                # Direction due North projected onto the celestial plane and expressed in terms of e1 and e2
                due_north_projected = self.project([self.e1, self.e2], *due_north)
                self.e2 = self.e1 * due_north_projected[0][0] + self.e2 * due_north_projected[1][0]
                self.e1 = np.cross(self.e2, self.e3)
                self.e1 = self.norm(self.e1)
                self.e2 = self.norm(self.e2)
            
        self.distance = self.mag(self.source - self.telescope) # Distance to the star in pc

        # Prepare the field and density for interpolation in between the values provided in the data cube
        self.B_x_interp = RegularGridInterpolator((x_axis, y_axis, z_axis), B_x)
        self.B_y_interp = RegularGridInterpolator((x_axis, y_axis, z_axis), B_y)
        self.B_z_interp = RegularGridInterpolator((x_axis, y_axis, z_axis), B_z)
        self.n_interp   = RegularGridInterpolator((x_axis, y_axis, z_axis), self.n)

        self.sigma_A = np.vectorize(self.sigma_A)
        self.sigma_T = np.vectorize(self.sigma_T)

    def eval(self, x, y, z):
        """
        Interpolate the data cube to a given point(s)

        arguments
            x                 :   x-coordinate of the point to interpolate in pc or array of such
            y                 :   y-coordinate of the point to interpolate in pc or array of such
            z                 :   z-coordinate of the point to interpolate in pc or array of such

        returns
            B_x               :   x-component of the magnetic field in arbitrary units
            B_y               :   y-component of the magnetic field in arbitrary units
            B_z               :   z-component of the magnetic field in arbitrary units
            nu                :   Angle of deviation from orthogonality of the Poynting vector of starlight
                                  with the (galactic) magnetic field vector, in radians. E.g., 0 is returned
                                  when the two vectors are orthogonal. See figure 3 of DG51
            n                 :   Dust number density at the interpolation point in cm^-3
            E_sigma_x         :   x-component of the unit-vector, specifying the direction of the component of
                                  the electric field of starlight, parallel to the plane of the (galactic) 
                                  magnetic field and the Poynting vector. This vector is required, as DG51 define
                                  attenuation by dust in terms of it, as well as its perpendicular counterpart,
                                  "E_pi". Both "E_pi" and "E_sigma" will be perpendicular to the Poynting vector.
                                  If the galactic field is coaligned with the Poynting vector, both "E_pi" and
                                  "E_sigma" will be set to zero, as no polarization will occur
            E_sigma_y         :   y-component of "E_sigma". See "E_sigma_x"
            E_sigma_z         :   z-component of "E_sigma". See "E_sigma_x"
            E_pi_x            :   x-component of "E_pi". See "E_sigma_x"
            E_pi_y            :   y-component of "E_pi". See "E_sigma_x"
            E_pi_z            :   z-component of "E_pi". See "E_sigma_x"
        """
        if np.shape(x) != np.shape(y) or np.shape(x) != np.shape(z):
            raise ValueError('The shapes of x, y and z do not match each other!')
        scalar_input = False
        if np.asarray(x).ndim == 0:
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])
            scalar_input = True
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        B_x = self.B_x_interp(np.array([x, y, z]).T)
        B_y = self.B_y_interp(np.array([x, y, z]).T)
        B_z = self.B_z_interp(np.array([x, y, z]).T)
        n = self.n_interp(np.array([x, y, z]).T)
        field = self.norm(np.array([B_x, B_y, B_z]).T)
        field_mag = self.mag(field)

        nu = np.zeros(len(field_mag))
        E_sigma = np.zeros([len(field_mag), 3])
        E_pi = np.zeros([len(field_mag), 3])

        nu[field_mag != 0] = np.arccos(np.sum(field[field_mag != 0] * self.e3, axis = 1) / field_mag[field_mag != 0]) - np.pi/2
        E_sigma[field_mag != 0] = self.norm(np.cross(field[field_mag != 0], self.e3))
        E_pi[field_mag != 0] = self.norm(np.cross(E_sigma[field_mag != 0], self.e3))

        if scalar_input:
            return B_x[0], B_y[0], B_z[0], nu[0], n[0], E_sigma.T[0][0], E_sigma.T[1][0], E_sigma.T[2][0], E_pi.T[0][0], E_pi.T[1][0], E_pi.T[2][0]
        return B_x, B_y, B_z, nu, n, E_sigma.T[0], E_sigma.T[1], E_sigma.T[2], E_pi.T[0], E_pi.T[1], E_pi.T[2]


    def mag(self, vector):
        """
        Get the magnitude of a vector or multiple arbitrary-rank tensors

        arguments
            vector            :   1D NumPy array, specifying the vector. Alternatively, (x+1)D array
                                  of multiple x-rank tensors

        returns
            Magnitude of "vector" (scalar) or 1D array of magnitudes if multiple tensors are provided
        """
        vector = np.array(vector)
        scalar_input = False
        if vector.ndim == 1:
            vector = np.array([vector])
            scalar_input = True
        result = np.sqrt(np.sum(vector * vector, axis = tuple(range(1, vector.ndim))))
        if scalar_input:
            return result[0]
        else:
            return result

    def norm(self, vector):
        """
        Normalize a vector or multiple arbitrary-rank tensors

        arguments
            vector            :   1D NumPy array, specifying the vector. Alternatively, (x+1)D array
                                  of multiple x-rank tensors

        returns
            Unit-vector, co-aligned with "vector" as a 1D array or (x+1)D array of normalized x-rank
            tensors
        """
        scalar_input = False
        vector = np.array(vector)
        if len(np.shape(vector)) == 1:
            scalar_input = True
            vector = np.array([vector])
        mag = self.mag(vector)
        result = np.zeros(np.shape(vector))
        result[mag != 0] = vector[mag != 0] / np.multiply.outer(mag[mag != 0], np.ones(np.shape(vector[mag != 0])[1:]))
        if scalar_input:
            return result[0]
        return result

    def angle_2d(self, vec1, vec2, norm):
        """
        Calculate the angle between two vectors in the same plane, preserving the appropriate sign
        (i.e. clockwise VS anticlockwise)

        arguments
            vec1              :   The first vector
            vec2              :   The second vector
            norm              :   Unit-vector, normal to the plane

        returns
            Signed angle between "vec1" and "vec2" in the plane defined by "norm" in radians between 0 and 2pi
        """
        vec1 = self.norm(vec1)
        vec2 = self.norm(vec2)
        sign = np.sign(np.dot(norm, np.cross(vec1, vec2)))
        if sign == 0:
            sign = 1.0
        dot = np.dot(vec1, vec2)
        if np.abs(dot) > 1.0:
            dot = 1.0 * np.sign(dot)
        return np.arccos(dot) * sign % (2 * np.pi)

    def project(self, unit_vectors, x, y, z):
        """
        Project a collection of points given by their xyz coordinates onto a different coordinates grid defined
        by a collection of unit vectors

        arguments
            unit_vectors      :   Unit vectors defining the projected coordinates grid. Must be a list of
                                  NumPy 1D arrays, each of length 3. The number of unit vectors does not have
                                  to equal three
            x                 :   Array of x-coordinates in pc. Alternatively, a single value
            y                 :   Array of y-coordinates in pc. Alternatively, a single value
            z                 :   Array of z-coordinates in pc. Alternatively, a single value

        returns
            2D NumPy array of shape (len(unit_vectors), len(x|y|z)), containing the projected coordinates in pc
        """
        if np.shape(x) == ():
            x = np.array([x])
        if np.shape(y) == ():
            y = np.array([y])
        if np.shape(z) == ():
            z = np.array([z])
        original = np.matrix(np.vstack([x.flatten(), y.flatten(), z.flatten()])).T
        return np.array((original * np.matrix(np.vstack(unit_vectors)).T).T)

    def quiver(self, sky = False, normalize = True, resample = False):
        """
        Express the magnetic field in a format that can be directly plotted with MatPlotLib's quiver() in
        2D (plane of the sky projection) or 3D (requires mpl_toolkits.mplot3d.Axes3D)

        arguments
            sky               :   Set to True to prepare a 2D projection of the field onto the plane of the sky.
                                  Flat sky approximation will be used, centered at the origin of the data cube.
                                  The plane of the sky is defined by self.e3. Note that all magnetic field data
                                  points will be projected, regardless of whether they are behind or in front of
                                  the telescope. Note that any vectors that project into (0, 0) will be excluded.
                                  If False (default), no projection will be carried out and the data
                                  cube will be returned in its native 3D format
            normalize         :   If True (default), all magnetic field vectors will be scaled by the dust number
                                  density in their immediate vicinity
            resample          :   Resample the grid to a given number of points. The output will be down- or up-
                                  sampled to the number of grid points along each axis specified in this argument.
                                  For example, if "resample" is set to 10, the output will contain 10x10x10 quiver
                                  vectors in 3D ("sky" is False) or 10x10 quiver vectors in 2D ("sky" is True).
                                  This argument is necessary when visualizing extensive data cubes that will not
                                  be supported by MatPlotLib without downsampling due to memory limitations. The 
                                  argument can also be used to make the resulting quiver plot less crowded. Set to
                                  False (default) to disable the feature

        returns
            x                 :   "X" argument of MatPlotLib's quiver()
            y                 :   "Y" argument of MatPlotLib's quiver()
            z                 :   "Z" argument of MatPlotLib's quiver() (only if "sky" is False)
            u                 :   "U" argument of MatPlotLib's quiver()
            v                 :   "V" argument of MatPlotLib's quiver()
            w                 :   "W" argument of MatPlotLib's quiver() (only if "sky" is False)
        """
        if type(resample) == bool and resample == False:
            x_axis, y_axis, z_axis = np.meshgrid(self.x_axis, self.y_axis, self.z_axis, indexing = 'ij')
            B_x = self.B_x; B_y = self.B_y; B_z = self.B_z; n = self.n
        else:
            x_axis, y_axis, z_axis = np.meshgrid(np.linspace(min(self.x_axis), max(self.x_axis), resample), np.linspace(min(self.x_axis), max(self.x_axis), resample), np.linspace(min(self.x_axis), max(self.x_axis), resample), indexing = 'ij')
            B_x = self.B_x_interp(np.array([x_axis, y_axis, z_axis]).T)
            B_y = self.B_y_interp(np.array([x_axis, y_axis, z_axis]).T)
            B_z = self.B_z_interp(np.array([x_axis, y_axis, z_axis]).T)
            n   = self.n_interp(np.array([x_axis, y_axis, z_axis]).T)
        if normalize:
            prefactor = n
        else:
            prefactor = 1.0
        if not sky:
            return x_axis, y_axis, z_axis, prefactor* B_x, prefactor * B_y, prefactor* B_z
        else:
            e1_axis, e2_axis = self.project([self.e1, self.e2], x_axis, y_axis, z_axis)
            B_e1, B_e2 = self.project([self.e1, self.e2], B_x, B_y, B_z)
            prefactor = prefactor.flatten()
            result = e1_axis, e2_axis, prefactor * B_e1, prefactor * B_e2
            # Remove all null-vectors
            mask = (result[2] != 0) | (result[3] != 0)
            return result[0][mask], result[1][mask], result[2][mask], result[3][mask]

    def P(self, x):
        """
        Calculate the axial depolarization factor for a dust grain of the aspect ratio "x".
        Based on equations 26 and 27 of DG51

        arguments
            x                 :   Aspect ratio of the grain, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis

        returns
            Axial depolarization factor, as defined in equations 26 and 27 of DG51
        """
        scalar_input = False
        x = np.array(x)
        if len(np.shape(x)) == 1:
            scalar_input = True
            x = np.array([x])

        if np.min(x) <= 0.0:
            raise ValueError('The aspect ratio of the grains must be a positive number')

        result = np.zeros(np.shape(x))
        # Prolate
        result[x > 1.0] = (4 * np.pi) / (x[x > 1.0] ** 2.0 - 1) * (x[x > 1.0] / np.sqrt(x[x > 1.0]**2 - 1) * np.arccosh(x[x > 1.0]) - 1)
        # Oblate
        result[x < 1.0] = -(4 * np.pi) / (x[x < 1.0] ** 2.0 - 1) * (1 - x[x < 1.0] / np.sqrt(1 - x[x < 1.0]**2) * np.arccos(x[x < 1.0]))
        # Spherical (limit as x->1 from either of two formulae above)
        result[x == 1.0] = (4.0 / 3.0) * np.pi

        if scalar_input:
            return result[0]
        return result

    def P_prime(self, x):
        """
        Calculate the transverse depolarization factor for a dust grain of the aspect ratio "x".
        Based on equation28 of DG51

        arguments
            x                 :   Aspect ratio of the grain, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis

        returns
            Axial depolarization factor, as defined in equation 28 of DG51
        """
        return 2 * np.pi - 0.5 * self.P(x)

    def sigma(self, a, wl, m, P):
        """
        Calculate the scattering cross-section for a dust grain with a given depolarization factor.
        Based on equation 29 of DG51. Note that the metallic case is not currently supported!

        arguments
            a                 :   Effective radius of the dust grain in cm
            wl                :   Wavelength of starlight interacting with the grain (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            P                 :   Depolarization factor that can be computed for axial scattering with P() and
                                  transverse scattering with P_prime()

        returns
            Scattering cross-section in cm^2
        """
        result = (128 * np.pi**5 * a**6) / (3 * wl**4)
        result *= ((m.real**2 - 1) / (3 + (m.real**2 - 1) * (3 * P / (4 * np.pi)))) ** 2.0
        return result

    def sigma_A(self, a, wl, m, x):
        """
        Calculate the axial scattering cross-section for a dust grain. This method dispatches P() and sigma()
        and does not carry out any calculations on its own. By "axial scattering", one means scattering of a
        photon, polarized in parallel with the axis of symmetry of the dust grain

        arguments
            a                 :   Effective radius of the dust grain in cm
            wl                :   Wavelength of starlight interacting with the grain (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The
                                  metallic case is not currently supported
            x                 :   Aspect ratio of the grain, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis

        returns
            Scattering cross-section in cm^2
        """
        return self.sigma(a, wl, m, self.P(x))
    

    def sigma_T(self, a, wl, m, x):
        """
        Calculate the transverse scattering cross-section for a dust grain. This method dispatches P_prime() and
        sigma() and does not carry out any calculations on its own. By "transverse scattering", one means
        scattering of a photon, polarized in parallel with the axis, transverse to the axis of symmetry of the
        grain

        arguments
            a                 :   Effective radius of the dust grain in cm
            wl                :   Wavelength of starlight interacting with the grain (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            x                 :   Aspect ratio of the grain, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis

        returns
            Scattering cross-section in cm^2
        """
        return self.sigma(a, wl, m, self.P_prime(x))

    def S_t(self, wl, m, x, a, rho):
        """
        Absorption coefficient for a beam of starlight propagating through a cloud of interstellar dust, where
        each grain has a fixed aspect ratio and lies on a given size distribution. Based on equation 45
        of DG51. Note the distribution of grain sizes provided in "a" and "rho" must integrate to unity!

        arguments
            wl                :   Wavelength of starlight (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            a                 :   Vector, defining the grain sizes, in terms of which the grain size distribution
                                  is defined. All values are in cm
            rho               :   Vector of the same length as "a", characterizing the abundance of each grain
                                  size in "a". The units of each value are cm^-1. Note that "rho" must integrate
                                  over "a" to unity to define a valid distribution
            

        returns
            Absorption coefficient (unitless)
        """
        return (1.0 / 3.0) * np.trapz((2 * self.sigma_T(a, wl, m, x) + self.sigma_A(a, wl, m, x)) * rho, a)

    def S_p(self, wl, m, x, a, rho, F = False):
        """
        Absorption coefficient for the polarized component of the beam of starlight propagating through a cloud
        of interstellar dust, where each grain has a fixed aspect ratio and lies on a given size distribution.
        Based on equation 46 of DG51. Note the distribution of grain sizes provided in "a" and "rho" must
        integrate to unity!

        arguments
            wl                :   Wavelength of starlight (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            a                 :   Vector, defining the grain sizes, in terms of which the grain size distribution
                                  is defined. All values are in cm
            rho               :   Vector of the same length as "a", characterizing the abundance of each grain
                                  size in "a". The units of each value are cm^-1. Note that "rho" must integrate
                                  over "a" to unity to define a valid distribution
            F                 :   The value of the F-integral defined in equation 44 of DG51. The value determines
                                  the degree, to which the dust grains are aligned with the galactic magnetic field.
                                  For oblate ("x"<1) grains, in the strong field limit (where all grains are perfectly
                                  aligned), this value tends to -(2/3). Use (0) for random alignment and (1/3) for
                                  perfect antialignment. The sign is inverted for prolate grains. Set to False (default)
                                  for perfectly aligned grains

        returns
            Absorption coefficient (unitless)
        """
        if type(F) == bool and (not F):
            F = np.sign(x - 1) * (2.0/3.0)
        return (3.0 / 2.0) * np.trapz((self.sigma_A(a, wl, m, x) - self.sigma_T(a, wl, m, x)) * rho * F, a)

    def S_pi(self, wl, m, x, a, rho, F = False, nu = 0, S_t = False, S_p = False):
        """
        Absorption coefficient for the component the beam of starlight, polarized along the "E_pi" unit vector
        as calculated by eval(), propagating through a cloud of interstellar dust, where each grain has a fixed
        aspect ratio and lies on a given size distribution. Based on equation 47 of DG51. Note the distribution
        of grain sizes provided in "a" and "rho" must integrate to unity!

        arguments
            wl                :   Wavelength of starlight (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            a                 :   Vector, defining the grain sizes, in terms of which the grain size distribution
                                  is defined. All values are in cm
            rho               :   Vector of the same length as "a", characterizing the abundance of each grain
                                  size in "a". The units of each value are cm^-1. Note that "rho" must integrate
                                  over "a" to unity to define a valid distribution
            F                 :   The value of the F-integral defined in equation 44 of DG51. The value determines
                                  the degree, to which the dust grains are aligned with the galactic magnetic field.
                                  For oblate ("x"<1) grains, in the strong field limit (where all grains are perfectly
                                  aligned), this value tends to -(2/3). Use (0) for random alignment and (1/3) for
                                  perfect antialignment. The sign is inverted for prolate grains. Set to False (default)
                                  for perfectly aligned grains
            nu                :   Angle of deviation from orthogonality of the Poynting vector of starlight
                                  with the (galactic) magnetic field vector, in radians. E.g., 0 is returned
                                  when the two vectors are orthogonal. See figure 3 of DG51. "nu" can be computed
                                  with eval(). Defaults to 0
            S_t               :   Precomputed output of self.S_t() if avaialble (see the docstring for details)
                                  Defaults to False, which forces a fresh calculation (slow)
            S_p               :   Precomputed output of self.S_p() if avaialble (see the docstring for details)
                                  Defaults to False, which forces a fresh calculation (slow)

        returns
            Absorption coefficient (unitless)
        """
        if type(F) == bool and (not F):
            F = np.sign(x - 1) * (2.0/3.0)
        if type(S_t) == bool and (not S_t):
            S_t = self.S_t(wl, m, x, a, rho)
        if type(S_p) == bool and (not S_p):
            S_p = self.S_p(wl, m, x, a, rho, F)
        return S_t - S_p * (np.cos(nu)**2.0 - (1/3.0))

    def S_sigma(self, wl, m, x, a, rho, F = False, nu = 0, S_t = False, S_p = False):
        """
        Absorption coefficient for the component the beam of starlight, polarized along the "E_sigma" unit vector
        as calculated by eval(), propagating through a cloud of interstellar dust, where each grain has a fixed
        aspect ratio and lies on a given size distribution. Based on equation 47 of DG51. Note the distribution
        of grain sizes provided in "a" and "rho" must integrate to unity!

        arguments
            wl                :   Wavelength of starlight (in cm)
            m                 :   Index of refraction. Must be a real number for dielectric particles. The metallic
                                  case is not currently supported
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            a                 :   Vector, defining the grain sizes, in terms of which the grain size distribution
                                  is defined. All values are in cm
            rho               :   Vector of the same length as "a", characterizing the abundance of each grain
                                  size in "a". The units of each value are cm^-1. Note that "rho" must integrate
                                  over "a" to unity to define a valid distribution
            F                 :   The value of the F-integral defined in equation 44 of DG51. The value determines
                                  the degree, to which the dust grains are aligned with the galactic magnetic field.
                                  For oblate ("x"<1) grains, in the strong field limit (where all grains are perfectly
                                  aligned), this value tends to -(2/3). Use (0) for random alignment and (1/3) for
                                  perfect antialignment. The sign is inverted for prolate grains. Set to False (default)
                                  for perfectly aligned grains
            nu                :   Angle of deviation from orthogonality of the Poynting vector of starlight
                                  with the (galactic) magnetic field vector, in radians. E.g., 0 is returned
                                  when the two vectors are orthogonal. See figure 3 of DG51. "nu" can be computed
                                  with eval(). Defaults to 0
            S_t               :   Precomputed output of self.S_t() if avaialble (see the docstring for details)
                                  Defaults to False, which forces a fresh calculation (slow)
            S_p               :   Precomputed output of self.S_p() if avaialble (see the docstring for details)
                                  Defaults to False, which forces a fresh calculation (slow)

        returns
            Absorption coefficient (unitless)
        """
        if type(F) == bool and (not F):
            F = np.sign(x - 1) * (2.0/3.0)
        if type(S_t) == bool and (not S_t):
            S_t = self.S_t(wl, m, x, a, rho)
        if type(S_p) == bool and (not S_p):
            S_p = self.S_p(wl, m, x, a, rho, F)
        return S_t + (1/3.0) * S_p

    def grain_size(self, distribution = 'weingartner-draine'):
        """
        Generate a normalized dust grain size distribution based on measurements from external literature references.
        Supported distributions:
            weingartner-draine      :       Distribution of silicate dust grains from 2001ApJ...548..296W, obtained by fitting
                                            a functional form to the observed extinction law. Assumed Rv=Av/E(B-V)=3.1 (typical
                                            for the diffuse interstellar medium at high galactic latitudes) and the carbon abundance
                                            of 6e-5 atoms per H nucleus (consistent with IR observations in 2001ApJ...554..778L)

        arguments
            distribution      :   Machine name of the distribution of interest (defaults to 'weingartner-draine')

        returns
            a                 :   Grid of grain sizes, for which the distribution is defined in cm
            rho               :   Fractional abundance of each size. NumPy array of the same length as "a"
        """
        if distribution == 'weingartner-draine':
            # Values from table I of 2001ApJ...548..296W for Rv=3.1, b_C=6e-5 (Carbon abundance), case "A"
            alpha = -2.21; beta = 0.300; ats = 0.164e-4; acs = 0.1e-4    # "acs" given in the text
            # Sample sizes from 3.5 A to 0.5 micron
            a = 10 ** np.linspace(np.log10(3.5e-8), np.log10(0.5e-4), 100)
            # Equation 6 of 2001ApJ...548..296W
            if beta >= 0:
                F = 1 + beta * a / ats
            else:
                F = (1 - beta * a / ats) ** -1.0
            # Equation 5 of 2001ApJ...548..296W
            rho = (a / ats) ** alpha * F / a
            c = a > ats
            rho[c] *= np.e ** (- ((a[c] - ats) / acs) ** 3.0)
        else:
            raise ValueError('Unknown dust grain distribution: {}'.format(distribution))
        a = np.array(a)
        rho = np.array(rho)
        rho /= np.trapz(rho, a)
        return a, rho

    def dust_to_gas_ratio(self, distribution = 'weingartner-draine', mass_ratio = 0.01, dust_density = 3.5, dust_volume_factor = (4.0 / 3.0) * np.pi, gas_molecular_mass = 2.0):
        """
        Helper function: estimate the gas-to-dust ratio by particle number. In many cases, only the gas density
        at a given region in the ISM is known and the dust number density must be inferred from it

        arguments
            distribution       :   Machine name of the dust grain size distribution. See grain_size() (defaults to 'weingartner-draine')
            mass_ratio         :   Dust-to-gas ratio by mass. Local value for the Milky Way is usually taken to be ~ 0.01, originally
                                   introduced in Hildebrand 1983 (1983QJRAS..24..267H), section 7. Defaults to 0.01
            dust_density       :   Density of dust grains in g cm^-3. Defaults to 3.5 - the value used in Weingartner+2001 (2001ApJ...548..296W)
                                   for silicate dust grains
            dust_volume_factor :   The value of K in V=K*a^3, where V is the volume of a dust grain and a is its size. Defaults to (4/3)pi - an
                                   appropriate factor for nearly spherical grains
            gas_molecular_mass :   Average mass of a single gas molecule in proton masses. Usually 1 for atomic and 2 for molecular hydrogen.
                                   Defaults to 2 (molecular)

        returns
            N_dust             :   Dust-to-gas ratio by particle number
        """
        a, rho = self.grain_size(distribution)
        dust_mass = mass_ratio * gas_molecular_mass * spc.proton_mass * 1e3
        N_dust = dust_mass / np.trapz(rho * dust_volume_factor * a**3.0 * dust_density, a)
        return N_dust

    def receive_photon(self, wl, m, x, a, rho, F = False, initial_beam = [1, 0, 0], return_all = False, return_mueller = False, show_progress = 'none'):
        """
        Simulate propagation a photon of starlight through interstellar dust immersed in the galactic magnetic
        field to obtain its final fraction and direction of polarization observed at the telescope

        arguments
            wl                :   Wavelength of the photon (in cm)
            m                 :   Index of refraction for all dust grains. Must be a real number for
                                  dielectric particles. The metallic case is not currently supported
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            a                 :   Vector, defining the grain sizes, in terms of which the grain size distribution
                                  is defined. All values are in cm
            rho               :   Vector of the same length as "a", characterizing the abundance of each grain
                                  size in "a". The units of each value are cm^-1. Note that "rho" must integrate
                                  over "a" to unity to define a valid distribution
            F                 :   The value of the F-integral defined in equation 44 of DG51. The value determines
                                  the degree, to which the dust grains are aligned with the galactic magnetic field.
                                  For oblate ("x"<1) grains, in the strong field limit (where all grains are perfectly
                                  aligned), this value tends to -(2/3). Use (0) for random alignment and (1/3) for
                                  perfect antialignment. The sign is inverted for prolate grains. Set to False (default)
                                  for perfectly aligned grains
            initial_beam      :   Initial polarization state of the source. Defaults to [1,0,0] corresponding to fully
                                  unpolarized light
            return_all        :   If True, will return the Stokes vector of the beam at all points of integration
                                  as a function of distance from the source as well as the total optical depth
                                  along the line of sight. Defaults to False
            return_mueller    :   Only relevant when "return_all" is True. If True, calculate and return the Mueller
                                  matrix of the line of sight (i.e. a matrix that takes the initial (I,Q,U) at source
                                  and transforms them into final (I,Q,U) at telescope). Defaults to False
            progress          :   Show progress while carrying out the calculation. Must be "none" (string) to
                                  disable the feature (default), "tqdm" to use the normal TQDM progressbar or
                                  "tqdm_notebook" to use the Jupyter notebook widget provided by TQDM

        returns
            pol               :   Observed polarization fraction of starlight at the telescope
            psi               :   Observed polarization angle of starlight at the telescope in degrees. The axes,
                                  with respect to which this quantity is defined, are given by self.e1 and self.e2
            tau               :   Only returned if "return_all" is True. Optical depth (tau) along the line of sight
            dist              :   Only returned if "return_all" is True. Grid of distances from the source to the
                                  telescope, at which the Stokes vector is calculated. The unit is pc
            storage           :   Only returned if "return_all" is True. Stokes vector calculated along the starlight
                                  path at distances from the source given in "dist". NumPy 2D array. The first dimension
                                  is of length 3, containing the Stokes I, Q and U (V is assumed 0 and not returned) of
                                  the starlight photon. The second dimension is of the same length as "dist"
            mueller           :   Only returned if "return_all" and "return_mueller" are True. The Mueller matrix of the
                                  line of sight (as a NumPy matrix)
        """
        if type(F) == bool and (not F):
            F = np.sign(x - 1) * (2.0/3.0)

        # Initial guess for the path element (in pc). Later we will adjust the path element dynamically by either
        # dividing or multiplying it by two. "step_level" will store the number of such adjustments. Initially,
        # step_level==0
        step_level = 0
        dy = self.distance / 1000 * 2 ** step_level
        # Keep track of how much distance has been covered
        distance_travelled = 0.0
        # Keep track of where we are
        current_position = np.array(self.source).astype(np.float128) # Enhanced precision for assert-tests later

        # Precompute scattering cross-sections
        S_t = self.S_t(wl, m, x, a, rho)
        S_p = self.S_p(wl, m, x, a, rho, F)

        # Precompute the environment along the light path (i.e. B-field, dust density, unit vectors etc)
        # "path_level" stores the "step_level" the environment was precomputed for. We will be recomputing
        # the environment every time the "step_level" decreases to accommodate new points
        path_level = step_level
        calculate_path = lambda path_level : [np.linspace(self.source[0], self.telescope[0], 1000 * 2 ** (-path_level) + 1),
                                              np.linspace(self.source[1], self.telescope[1], 1000 * 2 ** (-path_level) + 1),
                                              np.linspace(self.source[2], self.telescope[2], 1000 * 2 ** (-path_level) + 1)]
        path = calculate_path(path_level)
        path_index = 0                        # Keep track of the index of the precomputed environment where
                                              # we currently are
        env = np.array(self.eval(*path))      # Precompute the environment

        # Initial Stokes vector of the beam. We will not be considering V at all
        beam = list(initial_beam)
        # Initial optical depth along the line of sight
        if return_all:
            if return_mueller:
                mueller_total = np.matrix(np.identity(3))
            tau = 0.0

        # Check that the grain size distribution is valid to avoid silly mistakes!
        if np.round(np.trapz(rho, a), 5) != 1.0:
            raise ValueError('The provided grain size distribution is not properly normalized!')

        if return_all:
            storage = [beam]
            dist = [0]

        # Initialize a progressbar
        if show_progress == 'tqdm_notebook':
            pbar = tqdm.tqdm_notebook(total = 100)
        elif show_progress == 'tqdm':
            pbar = tqdm.tqdm(total = 100)
        elif show_progress == 'none':
            pass
        else:
            raise ValueError('Unknown progressbar "{}"!'.format(show_progress))
        pbar_pos = 0

        # Now we step along the light path and compute the effect on the Stokes parameters
        while distance_travelled < self.distance:
            # I do not fully trust my indexing, so I am leaving those asserts here for now
            # If the indexing is correct, we should be able to recover the current position
            # from both "current_position" and "path_index"
            assert np.allclose(current_position, [path[0][path_index], path[1][path_index], path[2][path_index]])

            # Get the B-field, dust density, unit vectors and the angle of attack (nu) at current position
            # The asserts above are to ensure that out "path_index" is correct
            B_x, B_y, B_z, nu, n, E_sigma_x, E_sigma_y, E_sigma_z, E_pi_x, E_pi_y, E_pi_z = np.mean([env[:,path_index], env[:,np.min([np.shape(env)[1] - 1, path_index + 2 ** (step_level - path_level)])]], axis = 0)

            # Now we need to rotate the Stokes parameters from the telescope basis (self.e1, self.e2) into the
            # grain basis (E_pi, E_sigma). We start by computing the angular separations between those bases
            theta_pi = self.angle_2d(self.e1, [E_pi_x, E_pi_y, E_pi_z], self.e3)
            theta_sigma = self.angle_2d(self.e1, [E_sigma_x, E_sigma_y, E_sigma_z], self.e3)
            # Whichever of the two grain axes (E_pi or E_sigma) is closer to the x-axis of the telescope basis
            # (self.e1) will be considered as the x-axis of the grain. We can therefore take the angular separation
            # between the x-axes to work out the exact amount of rotation that needs to be done
            theta = np.min([theta_pi, theta_sigma])
            # Now rotate the Stokes vector by theta. We use the formula from
            # https://ned.ipac.caltech.edu/level5/Kosowsky/Kosowsky2.html
            beam_prime = [beam[0], beam[1] * np.cos(2 * theta) + beam[2] * np.sin(2 * theta),
                          -beam[1] * np.sin(2 * theta) + beam[2] * np.cos(2 * theta)]

            # Get attenuation coefficients along both axes using the radiative transfer equation (equation 49 in DG51)
            p_pi = 1 - self.S_pi(wl, m, x, a, rho, F, nu, S_t = S_t, S_p = S_p) * n * dy * spc.parsec * 1e2
            p_sigma = 1 - self.S_sigma(wl, m, x, a, rho, F, nu, S_t = S_t, S_p = S_p) * n * dy * spc.parsec * 1e2
            # Estimate the flux error introduced by this step (fehler)
            flux_error = np.max([(1 - p_pi), (1 - p_sigma)])
            # If the error is too large, repeat this step with a smaller step size
            if flux_error > self.flux_error:
                step_level -= 1
                # Recompute the environment if necessary
                if path_level > step_level:
                    path_index *= 2 ** (path_level - step_level)
                    path_level = step_level
                    path = calculate_path(path_level)
                    env = np.array(self.eval(*path))
                dy /= 2.0
                continue

            # We do not want to overrun the finish line (note: it is imperative that we do this check AFTER the flux error
            # check above, as otherwise there is a possibility that the shortened "dy" here will be halved due to larger
            # error, sending us off the grid of precalculated environments; I learned this the hard way)
            if distance_travelled + dy > self.distance:
                dy = self.distance - distance_travelled
                continue

            # Work out which of the axes is x and which is y
            if theta_pi < theta_sigma: # If E_pi is the x-axis
                p_x = p_pi
                p_y = p_sigma
            else:                      # Else if E_sigma is the x-axis
                p_x = p_sigma
                p_y = p_pi

            # Construct and apply the Mueller matrix using the attenuation coefficients and the imperfect linear polarizer
            # equation from https://www.fiberoptics4sale.com/blogs/wave-optics/104730310-mueller-matrices-for-polarizing-elements
            # The matrix has been trimmed to exclude Stokes V
            mueller = 0.5 * np.matrix([[p_x + p_y, p_x - p_y, 0], [p_x - p_y, p_x + p_y, 0], [0, 0, 2 * np.sqrt(p_x * p_y)]])
            beam_prime = np.array(mueller * np.matrix(beam_prime).T).T[0]

            # Rotate the Stokes vector back into the telescope coordinates
            theta = -theta
            beam = [beam_prime[0], beam_prime[1] * np.cos(2 * theta) + beam_prime[2] * np.sin(2 * theta),
                    -beam_prime[1] * np.sin(2 * theta) + beam_prime[2] * np.cos(2 * theta)]
            if return_all and return_mueller:
                mueller_total = np.matrix([[1, 0, 0], [0, np.cos(2 * theta), -np.sin(2 * theta)], [0, np.sin(2 * theta), np.cos(2 * theta)]]) * mueller_total
                mueller_total = mueller * mueller_total
                mueller_total = np.matrix([[1, 0, 0], [0, np.cos(2 * theta), np.sin(2 * theta)], [0, -np.sin(2 * theta), np.cos(2 * theta)]]) * mueller_total

            # Advance our position along the beam
            distance_travelled += dy
            current_position -= self.e3 * dy
            path_index += 1 * 2 ** (step_level - path_level)

            # Update the optical depth
            if return_all:
                tau += (storage[-1][0] - beam[0]) / (storage[-1][0])

            # Advance the progress bar
            pbar_pos_new = int(distance_travelled / self.distance * 100.0)
            while pbar_pos < pbar_pos_new:
                if show_progress != 'none':
                    pbar.update(1)
                pbar_pos += 1

            # If the flux error is too small, we are justified to go a little faster
            if flux_error <= self.flux_error / 4.0 and step_level < 0:
                dy *= 2.0
                step_level += 1

            if return_all:
                storage += [beam]
                dist += [distance_travelled]

        if show_progress != 'none':
            pbar.update(100 - pbar_pos)
            pbar.close()
            pbar.clear()

        # Finally, use the Stokes vector to work out polarization
        pol = np.sqrt(beam[1]**2 + beam[2]**2) / beam[0]
        psi = np.degrees(0.5 * np.arctan2(beam[2], beam[1]))
        if not return_all:
            return pol, psi
        else:
            if return_mueller:
                return pol, psi, tau, np.array(dist), np.array(storage).T, mueller_total
            return pol, psi, tau, np.array(dist), np.array(storage).T

    def receive_ism_emission(self, x, p0 = 0.2, F = False, initial_beam = [0, 0, 0], n_points = False, return_all = False, show_progress = 'none'):
        """
        In addition to interfering with the starlight photon, the dust along the line of sight will also
        emit an infrared photon, which may be polarized if the dust particles are aligned with the galactic
        magnetic field. This function returns the polarization fraction and angle of the observed dust radiation
        along the observer's line of sight using the treatment suggested in Planck XX (https://arxiv.org/pdf/1405.0872.pdf),
        i.e. by integrating the field and the density distribution along the line of sight.

        Note that while receive_photon() only accounts for the ISM between the source and the telescope, this
        function will extend this line of sight to the end of the data cube

        arguments
            x                 :   Aspect ratio of the grains, defined as the ratio of the length of the axis
                                  of symmetry to the length of the transverse axis
            p0                :   Intrinsic polarization parameter, following nearly the same definition as equation
                                  B.14 of Planck XX, but for R=1 (Rayleigh reduction factor for perfect alignment). If
                                  the alignment is imperfect, it must be specified in "F" instead. Values range between
                                  0 and 0.75. The value vaguely corresponds to the maximum allowed polarization fraction
                                  (the exact relationship is p_max=p0/[1-p0/3]). Defaults to 0.2, which is a value
                                  consistent with Planck XX measurements
            F                 :   The value of the F-integral defined in equation 44 of DG51. The value determines
                                  the degree, to which the dust grains are aligned with the galactic magnetic field.
                                  For oblate ("x"<1) grains, in the strong field limit (where all grains are perfectly
                                  aligned), this value tends to -(2/3). Use (0) for random alignment and (1/3) for
                                  perfect antialignment. The sign is inverted for prolate grains. Set to False (default)
                                  for perfectly aligned grains
            initial_beam      :   Stokes I,Q and U (list of three elements) of the radiation input into the line of sight
                                  from outside. Defaults to [0, 0, 0], corresponding to no input (i.e. all ISM emission is
                                  produced within the provided ISM data cube)
            n_points          :   Number of integration points: increase for precision, decrease for performance. Defaults
                                  to False, which enables automatic estimation of the value based on the dimensions of the
                                  data cube
            return_all        :   If True, will return the Stokes vector of the beam at all points of integration
                                  as a function of distance from the source. Defaults to False
            progress          :   Show progress while carrying out the calculation. Must be "none" (string) to
                                  disable the feature (default), "tqdm" to use the normal TQDM progressbar or
                                  "tqdm_notebook" to use the Jupyter notebook widget provided by TQDM

        returns
            emission_pol      :   Observed polarization fraction of ISM emission at the telescope
            emission_psi      :   Observed polarization angle of ISM emission at the telescope in degrees. The axes,
                                  with respect to which this quantity is defined, are given by self.e1 and self.e2
            dist              :   Only returned if "return_all" is True. Grid of distances from the end of the data cube to the
                                  telescope, at which the Stokes vector is calculated. The unit is pc
            storage           :   Only returned if "return_all" is True. Stokes vector calculated along the line of sight
                                  at distances from the end of the data cube given in "dist". NumPy 2D array. The first dimension
                                  is of length 3, containing the Stokes I, Q and U (V is assumed 0 and not returned) of
                                  the emitted radiation. The second dimension is of the same length as "dist"
        """
        if type(F) == bool and (not F):
            F = np.sign(x - 1) * (2.0/3.0)
        # "Rayleigh" refers to the Rayleigh reduction factor defined in equation 3.9 of 1985ApJ...290..211L. The factor sets the degree
        # of alignment of dust grains with the magnetic field, varying from 0 (no alignment) to 1 (perfect alignment). Here, we express
        # the factor in terms of the F-integral that is used throughout this codebase. Given that "F=0" implies no alignment and "F=+-2/3"
        # implies perfect alignment, we must make sure that "Rayleigh" is 0 and 1 respectively in those cases. For all intermediate values,
        # we do a simple linear interpolation. Finally, for all values outside of the range (antialignment), we set epsilon to 0 and display a warning.
        Rayleigh = (np.sign(x - 1) * F) / ((2.0/3.0))
        if Rayleigh < 0:
            Rayleigh = 0
            warnings.warn('Antialignment of dust in emission is not supported; the emitted photon will be assumed unpolarized')
        p0 *= Rayleigh

        # From the definition of p0 in 1985ApJ...290..211L, the value must vary between 0 and 0.75. Check to avoid silly mistakes
        if (p0 > 0.75) or (p0 < 0.0):
            raise ValueError('0 < p0 < 0.75 by definition!')

        # Initial Stokes vector for the emitted photon (blank, as no emission has occurred yet)
        emission_beam = list(initial_beam)

        # If "n_points" is not provided, suggest a value automatically. We want to take the longest possible path within the cube, i.e.
        # the path between opposite vertices and figure out how many data cube points lie along that path. The obtained number is multiplied
        # by 10 to account for any interpolation errors
        if type(n_points) == bool and (not n_points):
            dimensions = np.array([len(self.x_axis), len(self.y_axis), len(self.z_axis)])
            n_points = np.sqrt(np.sum(dimensions ** 2.0)) * 10
            n_points = max([n_points, 1000])  # Lower limit on n_points is 1000 to avoid ridiculously small numers in small cubes
        n_points = int(np.floor(n_points))

        # Now we need to find where the observer's line of sight ends, i.e. where it reaches the end of the cube. The equation of the
        # line of sight is telescope + (source - telescope) * x. We solve for the values of x where the line of sight reaches any of the
        # planes defined by the edges of the data cube. Among those "candidate origins", the closest one is the origin we want
        candidate_origins = []
        for i, axis in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            if self.source[i] - self.telescope[i] == 0.0:
                continue
            sf_max = (np.max(axis) - self.telescope[i]) / (self.source[i] - self.telescope[i])
            sf_min = (np.min(axis) - self.telescope[i]) / (self.source[i] - self.telescope[i])
            if sf_max > 0: # Only take positive values of "x", as the negative ones are behind the observer
                candidate_origins += [self.telescope + (self.source - self.telescope) * sf_max]
            if sf_min > 0:
                candidate_origins += [self.telescope + (self.source - self.telescope) * sf_min]
        candidate_origin_distances = []
        for candidate_origin in candidate_origins:
            candidate_origin_distances += [self.mag(candidate_origin - self.telescope)]
        origin = np.array(candidate_origins)[candidate_origin_distances == np.min(candidate_origin_distances)][0]
        # Check the origin for numerical bound errors
        for i, axis in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            if origin[i] > np.max(axis):
                assert np.allclose(np.max(axis), origin[i])
                origin[i] = np.max(axis)
            if origin[i] < np.min(axis):
                assert np.allclose(np.min(axis), origin[i])
                origin[i] = np.min(axis)

        # Define the integration path and obtain the environment at all integration points
        path = [np.linspace(origin[0], self.telescope[0], int(n_points)),
                np.linspace(origin[1], self.telescope[1], int(n_points)),
                np.linspace(origin[2], self.telescope[2], int(n_points))]
        env = np.array(self.eval(*path))
        dy = self.mag(origin - self.telescope) / (n_points - 1)

        if return_all:
            storage = [np.array(emission_beam)]
            dist = [0]

        # Initialize a progressbar
        if show_progress == 'tqdm_notebook':
            pbar = tqdm.tqdm_notebook(total = 100)
        elif show_progress == 'tqdm':
            pbar = tqdm.tqdm(total = 100)
        elif show_progress == 'none':
            pass
        else:
            raise ValueError('Unknown progressbar "{}"!'.format(show_progress))
        pbar_pos = 0

        for i, B_x, B_y, B_z, nu, n, E_sigma_x, E_sigma_y, E_sigma_z, E_pi_x, E_pi_y, E_pi_z in zip(range(int(n_points - 1)), *env[:,:-1]):
            # Use equations B.5-B.7 from https://arxiv.org/pdf/1405.0872.pdf to calculate the change in the Stokes vector of the emitted
            # photon. First, project the field into the observer's coordinates
            B_x_p, B_y_p = self.project([self.e1, self.e2], B_x, B_y, B_z)[:,0]
            B = self.mag([B_x, B_y, B_z])
            # Apply equations B.5-B.7
            if B != 0:
                emission_beam[0] += n * dy * spc.parsec * 1e2 * (1 - p0 * (((B_x_p ** 2.0 + B_y_p ** 2.0) / B ** 2.0)  - (2.0/3.0)))
                emission_beam[1] += p0 * n * dy * spc.parsec * 1e2 * (B_y_p ** 2.0 - B_x_p ** 2.0) / B ** 2.0
                emission_beam[2] -= p0 * 2 * n * dy * spc.parsec * 1e2 * (B_y_p * B_x_p) / B ** 2.0
            else:
                emission_beam[0] += n * dy * spc.parsec * 1e2
            
            if return_all:
                storage += [np.array(emission_beam)]
                dist += [dist[-1] + dy]

            # Advance the progress bar
            pbar_pos_new = int(i / n_points * 100.0)
            while pbar_pos < pbar_pos_new:
                if show_progress != 'none':
                    pbar.update(1)
                pbar_pos += 1

        if show_progress != 'none':
            pbar.update(100 - pbar_pos)
            pbar.close()
            pbar.clear()

        if emission_beam[0] != 0.0:
            emission_pol = np.sqrt(emission_beam[1]**2 + emission_beam[2]**2) / emission_beam[0]
        else:
            emission_pol = 0.0
        emission_psi = np.degrees(0.5 * np.arctan2(emission_beam[2], emission_beam[1]))
        if not return_all:
            return emission_pol, emission_psi
        else:
            return emission_pol, emission_psi, np.array(dist), np.array(storage).T
