#!/usr/bin/env python
"""
Some helper classes and methods to aid Northampton inversions
based on work by J Gomez-Dans in tip_helpers
"""
__author__  = "N Pounder"


import cPickle
from collections import OrderedDict
import numpy as np
import datetime as dt

import eoldas_ng

from tip_helpers import StandardStateTIP, ObservationOperatorTIP
from tip_helpers import bernards_prior, retrieve_albedo
from tip_inversion import tip_single_inversion
#from Northampton_tip_inversion import single_inversion


class state_configuration:
    def __init__(self):

        self.optimisation_options = {'gtol':1e-6,
                            'maxcor':50, 'maxiter':1500, "disp":20}
        self.state_grid = np.arange(1, 366, 8)
        print self.state_grid
        self.state_config = OrderedDict()
        self.state_config['omega_vis'] = eoldas_ng.VARIABLE
        self.state_config['d_vis'] = eoldas_ng.VARIABLE
        self.state_config['a_vis'] = eoldas_ng.VARIABLE
        self.state_config['omega_nir'] = eoldas_ng.VARIABLE
        self.state_config['d_nir'] = eoldas_ng.VARIABLE
        self.state_config['a_nir'] = eoldas_ng.VARIABLE
        self.state_config['lai'] = eoldas_ng.VARIABLE

class prior_config_polyleaves:
    def __init__(self):
        self.mu_prior = np.array ([0.17, 1, 0.1, 0.7, 2., 0.18, 2.])
        self.mu_prior_snow = np.array ([0.17, 1, 0.50, 0.7, 2., 0.35, 2.])

        self.prior_cov = np.diag ( [ 0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 5.])
        self.prior_cov[2,5] = 0.8862*0.0959*0.2
        self.prior_cov[5,2] = 0.8862*0.0959*0.2

        self.prior_cov_snow = np.diag ( [ 0.12, 0.7, 0.346, 0.15, 1.5, 0.25, 5.])
        self.prior_cov_snow[2,5] = 0.8670*0.346*0.25
        self.prior_cov_snow[5,2] = 0.8670*0.346*0.25

        self.iprior_cov = np.linalg.inv ( self.prior_cov )
        self.iprior_cov_snow = np.linalg.inv ( self.prior_cov_snow )

class prior_config_greenleaves:
    def __init__(self):
        mu_prior = np.array ([0.13, 1, 0.1, 0.77, 2., 0.18, 2.])
        mu_prior_snow = np.array ([0.13, 1, 0.50, 0.77, 2., 0.35, 2.])

        prior_cov = np.diag ( [ 0.0140, 0.7, 0.0959, 0.0140, 1.5, 0.2, 5.])
        prior_cov[2,5] = 0.8862*0.0959*0.2
        prior_cov[5,2] = 0.8862*0.0959*0.2

        prior_cov_snow = np.diag ( [ 0.0140, 0.7, 0.346, 0.0140, 1.5, 0.25, 5.])
        prior_cov_snow[2,5] = 0.8670*0.346*0.25
        prior_cov_snow[5,2] = 0.8670*0.346*0.25

        iprior_cov = np.linalg.inv ( self.prior_cov )
        iprior_cov_snow = np.linalg.inv ( self.prior_cov_snow )

def retrieve_albedo_season(startdate, enddate, fieldname, albedo_unc=[0.05, 0.07]):
# Uses retrieve_albedo but allows to get multiple years and parts of years.


    observations, mask, bu, passer_snow, dates = [], [],[],[], []
    #obs_all, mask_all, bu_all, passer_snow_all = [], [],[],[]
    for year in xrange(startdate.year, enddate.year+1):
        observations_yr, mask_yr, bu_yr, passer_snow_yr = retrieve_albedo ( year, fieldname,
                                                           albedo_unc=albedo_unc)
        date =np.array([(dt.date(year,1,1)+ dt.timedelta(doy-1)) for doy in mask_yr[:,0]])
        passer = [d>startdate and d<enddate for d in date]
        try:
            mask = np.vstack((mask, mask_yr[np.array(passer)]))
            observations = np.vstack((observations, observations_yr[np.array(passer)]))
            bu = np.vstack((bu, bu_yr[np.array(passer)]))
            passer_snow = np.hstack((passer_snow, passer_snow_yr[np.array(passer)]))
            dates = np.hstack((dates, date[np.array(passer)]))
        except ValueError:
            mask = mask_yr[np.array(passer)]
            observations = observations_yr[np.array(passer)]
            bu = bu_yr[np.array(passer)]
            passer_snow = passer_snow_yr[np.array(passer)]
            dates = np.array(date)[np.array(passer)]
    return observations, mask, bu, passer_snow, dates



def single_inversion (startdate, enddate, site, state_grid, 
                      vis_emu_pkl="tip_vis_emulator_real.pkl",
                      nir_emu_pkl="tip_nir_emulator_real.pkl" ):
    n_tries = 2
    observations, mask, bu, passer_snow, dates = retrieve_albedo_season( startdate, enddate, site, [0.05, 0.07])

    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))

    prior = prior_config_polyleaves()
    x0 = prior.mu_prior
    state = np.zeros((len(state_grid),7))
    
    for j,tstep in  enumerate(state_grid):
        state[j,:] = prior.mu_prior

        if tstep in mask[:,0]:
            i = mask[:,0] == tstep
            is_ok = mask[i,1]
            if is_ok == 1:
                if passer_snow[i]:
                    mu = prior.mu_prior_snow
                    inv_cov_prior = prior.iprior_cov_snow
                    cov_prior = prior.prior_cov_snow
                else:
                    mu = prior.mu_prior
                    inv_cov_prior = prior.iprior_cov
                    cov_prior = prior.prior_cov

            cost_list = np.zeros(n_tries)
            solutions = np.zeros((n_tries, 7))
            for trial in xrange(n_tries):
                 if trial > 0:
                     while True:
                         x0 = np.random.multivariate_normal(mu, cov_prior, 1)
                         if np.all ( x0 > 0 ):
                             break
                 retval = tip_single_inversion(x0, observations[i, :].squeeze(), bu[i,:].squeeze(),
                                           mu, inv_cov_prior, gp_vis, gp_nir)

                 cost_list[trial] = retval.fun
                 solutions[trial, :] = retval.x

            best_sol = cost_list.argmin()
            x0 = solutions[best_sol, :]
            state[j,:] = solutions[best_sol, :]
    return state



def tip_inversion ( startdate, enddate, fluxnet_site, albedo_unc=[0.05, 0.07], green_leaves=False,
                    prior_type="TIP_standard",
                    vis_emu_pkl="tip_vis_emulator_real.pkl",
                    nir_emu_pkl="tip_nir_emulator_real.pkl", n_tries=2, 
                    progressbar=None):
    """The JRC-TIP inversion using eoldas. This function sets up the
    invesion machinery for a particular FLUXNET site and year (assuming
    these are present in the database!)

    Parameters
    ----------
    year : int
        The year
    fluxnet_site: str
        The code of the FLUXNET site (e.g. US-Bo1)
    albedo_unc: list
        A 2-element list, containg the relative uncertainty
    prior_type: str
        Not used yet
    vis_emu_pkl: str
        The emulator file for the visible band.
    nir_emu_pkl: str
        The emulator file for the NIR band.
    n_tries: int
        Number of restarts for the minimisation. Best one (e.g. lowest
        cost) is chosen

    Returns
    -------
    Good stuff
    """
    # Retieve observatiosn and ancillary stuff from database
    observations, mask, bu, passer_snow, dates = retrieve_albedo_season (
                 startdate, enddate, fluxnet_site, albedo_unc )   

    # Get state configuration options
    state_config = state_configuration()
    #state_config.state_grid =  np.array([(d-dates[0]).days for d in dates])
    print enddate, startdate
    state_config.state_grid = np.arange(1, (enddate-startdate).days+1, 8)
    print state_config.state_grid
    print state_config.state_grid.size
    # Start by setting up the state 
    the_state = StandardStateTIP ( state_config.state_config, state_config.state_grid, 
                                  optimisation_options=state_config.optimisation_options)

    # Load and prepare the emulators for the TIP
    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r')) 
    # Set up the observation operator
    obsop = ObservationOperatorTIP ( the_state.state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )
    the_state.add_operator("Obs", obsop)
    # Set up the prior
    ### prior = the_prior(the_state, prior_type )
    prior = bernards_prior ( passer_snow, use_soil_corr=True,
                             green_leaves=green_leaves, N = len(dates))
    the_state.add_operator ("Prior", prior )
    # Now, we will do the function minimisation with `n_tries` different starting
    # points. We choose the one with the lowest cost...



    retval = single_inversion( startdate, enddate, fluxnet_site, the_state.state_grid, vis_emu_pkl, nir_emu_pkl)
    x_dict = {}
    for i,k in enumerate ( ['omega_vis', 'd_vis', 'a_vis', 'omega_nir', 'd_nir', 'a_nir', 'lai']):
        x_dict[k] = retval[:,i]

    results = []
    for i in xrange(n_tries):
        if n_tries > 1:
            x0 = np.random.multivariate_normal( prior.mu,
                       np.array(np.linalg.inv(prior.inv_cov.todense())))
            x_dict = the_state._unpack_to_dict ( x0 )
        print ('x_dict = ', x_dict)
        retval = the_state.optimize(x_dict, do_unc=True)
        results.append ( ( the_state.cost_history['global'][-1],
                         retval ) )
        if progressbar is not None:
            progressbar.value = progressbar.value + 1
    best_solution = np.array([ x[0] for x in results]).argmin()
    print [ x[0] for x in results]
    print "Chosen cost: %g" % results[best_solution][0]
    return results[best_solution][1], the_state, obsop, dates




def regularised_tip_inversion ( startdate, enddate, fluxnet_site, gamma, x0, albedo_unc=[0.05, 0.07], green_leaves=False,
                    prior_type="TIP_standard",
                    vis_emu_pkl="tip_vis_emulator_real.pkl",
                    nir_emu_pkl="tip_nir_emulator_real.pkl", n_tries=2,
                    prior=None, progressbar=None):
    """The JRC-TIP inversion using eoldas. This function sets up the
    invesion machinery for a particular FLUXNET site and year (assuming
    these are present in the database!)

    Parameters
    ----------
    year : int
        The year
    fluxnet_site: str
        The code of the FLUXNET site (e.g. US-Bo1)
    albedo_unc: list
        A 2-element list, containg the relative uncertainty
    prior_type: str
        Not used yet
    vis_emu_pkl: str
        The emulator file for the visible band.
    nir_emu_pkl: str
        The emulator file for the NIR band.
    n_tries: int
        Number of restarts for the minimisation. Best one (e.g. lowest
        cost) is chosen

    Returns
    -------
    Good stuff
    """
    # Retieve observatiosn and ancillary stuff from database
    observations, mask, bu, passer_snow, dates = retrieve_albedo_season ( startdate, enddate,
                                                       fluxnet_site, albedo_unc )

    # Get state configuration options
    state_config = state_configuration()
    #state_config.state_grid = np.array([(d-dates[0]).days for d in dates])
    state_config.state_grid = np.arange(0, (dates[-1]-dates[0]).days+1, 8)
    # Start by setting up the state
    the_state = StandardStateTIP ( state_config.state_config, state_config.state_grid,
                                  optimisation_options=state_config.optimisation_options )

    # Load and prepare the emulators for the TIP
    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))
    # Set up the observation operator
    obsop = ObservationOperatorTIP ( the_state.state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )
    the_state.add_operator("Obs", obsop)
    # Set up the prior
    ### prior = the_prior(the_state, prior_type )
    if prior is None:
        prior = bernards_prior(passer_snow, use_soil_corr=True,
                               green_leaves=green_leaves, N = len(dates))
        the_state.add_operator ("Prior", prior )
    else:
        the_state.add_operator("Prior", prior)
    # Now, we will do the function minimisation with `n_tries` different starting
    # points. We choose the one with the lowest cost...

    smoother = eoldas_ng.TemporalSmoother ( the_state.state_grid, gamma, required_params =
                                                            ["omega_vis", "d_vis", "a_vis",
                                                            "omega_nir", "d_nir", "a_nir", "lai"] )

    the_state.add_operator ( "Smooth", smoother)


    x_dict = x0
    results = []
    for i in xrange(n_tries):
        if n_tries > 1:
            x0 = np.random.multivariate_normal( prior.mu, np.array(np.linalg.inv(prior.inv_cov.todense())))
            x_dict = the_state._unpack_to_dict ( x0 )
        retval = the_state.optimize(x_dict, do_unc=True)
        results.append ( ( the_state.cost_history['global'][-1],
                         retval ) )
        if progressbar is not None:
            progressbar.value = progressbar.value + 1

    best_solution = np.array([ x[0] for x in results]).argmin()
    print [ x[0] for x in results]
    print "Chosen cost: %g" % results[best_solution][0]
    return results[best_solution][1], the_state, obsop, dates
