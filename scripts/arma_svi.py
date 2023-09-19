#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:11:52 2022

@author: cyberguli
"""


from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
import pandas as pd
import numpy as np
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.infer import Predictive
from statsmodels.tsa.stattools import bds
import pyro.contrib.autoguide as autoguide
figsize=(10,4)

series=read_csv('datipml.csv', header=0,names=["Data"])
plot_acf(series, title="original")
pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=0,names=["Data"])
pyro.clear_param_store()


temp=x.Data.tolist()

data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]

def modelA(data,start):
    z=[]
    a3=pyro.param("alpha1", torch.tensor(1.))
    sigma=pyro.param("sigma", torch.tensor(0.20), constraint=constraints.positive)
    k=pyro.param("k", torch.tensor(2), constraint=constraints.positive)

    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i<3:
            z.append(torch.tensor(-0.8))
        else:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(a3*z[i-3],sigma)))
        pyro.sample("obs_{}".format(i), dist.Bernoulli(1/(1+torch.exp(-k*z[i]))), obs=data[i])





guideA=autoguide.AutoNormal(modelA)
# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(modelA, guideA, optimizer, loss=Trace_ELBO())
'''
n_steps = 5000
# do gradient steps
for step in range(n_steps):
    svi.step(data,0)
    if step % 100 == 0:
        print('.', end='')

'''
losses=[]
n_steps = 5000
# do gradient steps
for step in range(n_steps):
    loss=svi.step(data,0)
    losses.append(loss)
    if step % 100 == 0:
        print('.', end='')
fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("ARMA loss");
fig.savefig("armaloss.png",dpi=900)


fitted = Predictive(modelA,None,guideA,num_samples=1)(datatrain,0)
zend=fitted["weiner_{}".format(len(datatrain)-1)].item()
pred=[]
zeta=[]
prob=[]
a3=pyro.param("alpha1", constraint=constraints.positive)
sigma=pyro.param("sigma")
k=pyro.param("k")


def BIC(data,a3,sigma,k):
    z=[]
    N=len(data)
    a3=pyro.param("alpha1", torch.tensor(-0.9))
    sigma=pyro.param("sigma", torch.tensor(0.20), constraint=constraints.positive)
    k=pyro.param("k", torch.tensor(2), constraint=constraints.positive)
    s=1
    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i<3:
            z.append(torch.tensor(-0.8))
        else:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(a3*z[i-3],sigma)))
        s=s+dist.Bernoulli(1/(1+torch.exp(-k*z[i]))).log_prob(data[i])
    
    return 3*torch.log(torch.tensor(N))-2*s




for i in range(len(data)):
        if i<2:
            zeta.append(torch.tensor(-0.8))
        else:
            zeta.append(pyro.sample("weiner_{}".format(i),dist.Normal(a3*zeta[i-3],sigma)))
        prob.append(1/(1+torch.exp(-k*zeta[i])))
        pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(1/(1+torch.exp(-k*zeta[i])))))

predtrain=pred[0:245]
predtest=pred[245:]



datatest_trans=[]
for i in range(len(datatest)):
    datatest_trans.append(datatest[i].item())


def compute_error(list1,list2):
    err=0
    for i in range(len(list1)):
        if abs(list1[i]-list2[i])>10e-02:
            err=err+1
    
    err=err/len(list1)
    return err
err=compute_error(datatest,predtest)


pred=torch.tensor(pred).reshape(len(pred)).numpy()
predtest=pred[245:]
predtrain=pred[:245]
datatest=torch.tensor(datatest).reshape(len(datatest)).numpy()

def autocorr(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    if variance==0:
        return 1
    else:
        result = r/(variance*(np.arange(n, 0, -1)))
    return result

bic=BIC(data,a3,sigma,k)
print("Autocorrelation STD is ", np.linalg.norm(autocorr(datatest)-autocorr(predtest))/len(datatest))
print("BIC is", bic)
    



'''
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

assert pyro.__version__.startswith('1.8.1')

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))
    
print(data)

def model(data):
    # define the hyperparameters that control the Beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the Beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the ernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
'''