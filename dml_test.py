#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:40:40 2021
@author: harshparikh
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.model_selection as ms
from scipy.stats import norm


def dml_pval(Psi):
    N = Psi.shape[0]
    theta = np.nanmean(Psi, axis = 0) # mean of psi1, psi0
    sigma = np.sqrt(np.nanmean(np.square(Psi - theta), axis = 0)) # std of psi1, psi0
    
    Q = sigma / (N ** 0.5) #stderr
    
    pvals = np.minimum(-2 * (norm.cdf(-theta / Q) - 1), -2 * (norm.cdf(theta / Q) - 1))
    
    # For mu(1) - nu(1) = 0 and mu(0) - nu(0) = 0, respectively
    return pvals

def Psi(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, outcome, treatment, sample):
    mu0 = df_mu.iloc[:, 1].values
    mu1 = df_mu.iloc[:, 0].values

    nu0 = df_nu.iloc[:, 1].values
    nu1 = df_nu.iloc[:, 0].values

    S = df[sample].values
    T = df[treatment].values
    Y = df[outcome].values

    p = df_p.iloc[:,0].values
    pi1 = df_pi_obs.values[:,0]


    Psi1 = (nu1 - mu1 + (((T == 1) * (1 - S) * (Y - nu1)) / ((1 - p) * pi1)) - (((T == 1) * S * (Y - mu1)) / (p * pi1)))


    Psi0 = (nu0 - mu0 + (((T == 0) * (1 - S) * (Y - nu0)) / ((1 - p) * (1 - pi1))) - (((T == 0) * S * (Y - mu0)) / (p * (1 - pi1))))
    
    psi = np.vstack((Psi1, Psi0)).T
    
    return psi

def Psi_unadj(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, outcome, treatment, sample):
    nu = df_nu.values
    mu = df_mu.values
    S = df[[sample]].values
    T = df[[treatment]].values
    T = np.hstack((T, 1 - T))
    Y = df[[outcome]].values
    pi_obs = df_pi_obs.values.reshape(-1, 1)
    pi_obs = np.hstack((pi_obs, 1 - pi_obs))
    pi_exp = df_pi_exp.values.reshape(-1, 1)
    pi_exp = np.hstack((pi_exp, 1 - pi_exp))
    p = df_p.values.reshape(-1, 1)
    psi = (nu - mu)
    return psi

def cross_fit(folds, df, df2, sample, Y, T, estimator):
    """ Performs cross-fitting by training on `df`. 
    folds: created from a call to `sklearn.model_selection.KFold`
    df: the data frame used to train the models
    df2: the data frame from the other sample (e.g. obs if `df` is exp)
    Y: the outcome name
    T: the treatment name 
    sample: a string, either 'exp' or 'obs' indicating the sample in `df` 

    Returns: predicted outcomes and propensity scores for all units in `df` and `df2`"""

    df_t = df.loc[df[T] == 1]
    df_c = df.loc[df[T] == 0]
    
    df_mu = pd.DataFrame()
    df_pi = pd.DataFrame()

    for tr_index, tst_index in folds.split(df):
        train = df.iloc[tr_index]
        test = df.iloc[tst_index]

        train_t = train.loc[train[T] == 1]
        train_c = train.loc[train[T] == 0]

        if estimator == 'linear':
            mu1_model = lm.RidgeCV()
            mu0_model = lm.RidgeCV()
            pi_model = lm.LogisticRegressionCV()
        elif estimator == 'grt':
            mu1_model = en.GradientBoostingRegressor()
            mu0_model = en.GradientBoostingRegressor()
            pi_model = en.GradientBoostingClassifier()
        else:
            raise RuntimeError('estimator must be linear') 

        # Train outcome model 
        mu_1 = mu1_model.fit(train_t.drop(columns = [Y, T]), train_t[Y])
        mu_0 = mu0_model.fit(train_c.drop(columns = [Y, T]), train_c[Y])

        # Predict for all same S units in test 
        mu_hat = np.array([mu_1.predict(test.drop(columns = [Y, T])), 
                           mu_0.predict(test.drop(columns = [Y, T]))]).T

        # Add predictions to running df of predictions for all units across folds
        mu_hat = pd.DataFrame(mu_hat, index = test.index, columns = ['1', '0'])
        df_mu = df_mu.append(mu_hat)

        # Train propensity score model 
        pi = pi_model.fit(train.drop(columns = [Y, T]), train[T])

        # Predict for all same S units in test 
        pi_hat = pi.predict_proba(test.drop(columns = [Y, T]))[:, 1].reshape(-1, 1)
        
        # Add predictions to running df of predictions for all units across folds 
        pi_hat = pd.DataFrame(pi_hat, index = test.index, columns = ['pi_' + sample])
        df_pi = df_pi.append(pi_hat)
    
    # End loop over folds
       
    # Train outcome model on *all* same S units 
    mu_1 = mu1_model.fit(df_t.drop(columns = [Y, T]), df_t[Y])
    mu_0 = mu0_model.fit(df_c.drop(columns = [Y, T]), df_c[Y])

    # Predict for all different S units
    mu_hat = np.array([mu_1.predict(df2.drop(columns = [Y, T])), 
                       mu_0.predict(df2.drop(columns = [Y, T]))]).T
    
    # Add predictions to running df 
    mu_hat = pd.DataFrame(mu_hat, index = df2.index, columns = ['1', '0'])
    df_mu = df_mu.append(mu_hat)
    
    # Train propensity score model on *all* same S units 
    pi = pi_model.fit(df.drop(columns = [Y, T]), df[T])

    # Predict for all different S units
    pi_hat = pi.predict_proba(df2.drop(columns = [Y, T]))[:, 1].reshape(-1, 1)
    
    # Add predictions to running df
    pi_hat = pd.DataFrame(pi_hat, index = df2.index, columns = ['pi_' + sample])
    df_pi = df_pi.append(pi_hat)

    return df_mu, df_pi

def fit(Y, T, S, df, estimator = 'linear', n_splits = 5):
    df_exp = df.loc[df[S]==1].drop(columns=[S])
    df_obs = df.loc[df[S]==0].drop(columns=[S])
    
    if n_splits == 'max':
        n_splits_exp = df_exp.shape[0]
        n_splits_obs = df_obs.shape[0]
    else:
        n_splits_exp = n_splits
        n_splits_obs = n_splits
    
    skf_exp = ms.KFold(n_splits = n_splits_exp, shuffle = True)
    skf_obs = ms.KFold(n_splits = n_splits_obs, shuffle = True)
    skf_all = ms.KFold(n_splits = n_splits_exp + n_splits_obs, shuffle = True)
    
    df_p = pd.DataFrame()
    
    df_mu, df_pi_exp = cross_fit(skf_exp, df_exp, df_obs, 'exp', Y, T, estimator)
    df_nu, df_pi_obs = cross_fit(skf_obs, df_obs, df_exp, 'obs', Y, T, estimator)
        
    # Loop through all units 
    for tr_index, tst_index in skf_all.split(df):
        train = df.iloc[tr_index]
        test = df.iloc[tst_index]

        if estimator == 'linear':
            p_model = lm.LogisticRegressionCV() 
        elif estimator == 'grt':
            p_model = en.GradientBoostingClassifier()
        else:
            raise RuntimeError('estimator must be linear') 
    
        # Train selection model on all train units 
        p = p_model.fit(train.drop(columns = [Y, T, S]), train[S])

        # Predict selection scores on all test units
        p_hat = p.predict_proba(test.drop(columns = [Y, T, S]))[:, 1].reshape(-1, 1)
        
        # Add to running df of predictions 
        p_hat = pd.DataFrame(p_hat, index = test.index, columns = ['p'])
        df_p = df_p.append(p_hat)
#         df_p = pd.concat([df_p, p_hat], join = 'outer', axis = 1)
    
    
#     df_mu['1'] = df_mu['1'].mean()
#     df_mu['0'] = df_mu['0'].mean()
#     df_nu['1'] = df_nu['1'].mean()
#     df_nu['0'] = df_nu['0'].mean()
    
#     df_p = df_p.mean()
#     df_pi_exp = df_pi_exp.mean(axis = 1)
#     df_pi_obs = df_pi_obs.mean(axis = 1)
    
    return df_mu.iloc[:, :2].sort_index(), df_nu.iloc[:, :2].sort_index(), df_p.sort_index(), df_pi_exp.sort_index(), df_pi_obs.sort_index(), df.sort_index()

def Lambda(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, outcome, treatment, sample):
    """Construct the estimator Lambda for the population ATE."""
    mu0 = df_nu.iloc[:, 1].values # mu(X, 0, 0) in the current notation
    mu1 = df_nu.iloc[:, 0].values # mu(X, 1, 0) in the current notation
    
    S = df[sample].values
    T = df[treatment].values
    Y = df[outcome].values
    
    p = df_p.values[:,0]
    pi_exp = df_pi_exp.values[:,0]
    
    lambda0 = ( S * (1-T) * Y ) / ( p * ( 1 - pi_exp ) ) + ( mu0 * ( 1 - ( S*(1-T)/( p * ( 1 - pi_exp ) ) ) ) )
    lambda1 = ( S * (T) * Y ) / ( p * ( pi_exp ) ) + ( mu0 * ( 1 - ( S*(T)/( p * ( pi_exp ) ) ) ) )
    
    return np.vstack((lambda1, lambda0)).T

def Lambda_exp(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, outcome, treatment, sample):
    """Construct the estimator Lambda for the population ATE just using experimental ATE."""
    
    S = df.loc[df[sample]==1][sample].values
    T = df.loc[df[sample]==1][treatment].values
    Y = df.loc[df[sample]==1][outcome].values
    
    p = df_p.loc[df[sample]==1].values[:,0]
    pi_exp = df_pi_exp.loc[df[sample]==1].values[:,0]
    
    lambda0 = ( S * (1-T) * Y ) / ( p * ( 1 - pi_exp ) ) 
    lambda1 = ( S * (T) * Y ) / ( p * ( pi_exp ) ) 
    
    return np.vstack((lambda1, lambda0)).T