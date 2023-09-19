import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro.infer import Predictive
import pyro.contrib.autoguide as autoguide
import warnings
import numpy as np
from pyro import poutine
warnings.filterwarnings("ignore")
from pyro.infer.mcmc import MCMC, HMC, NUTS

# clear the param store in case we're in a REPL
pyro.clear_param_store()


data_dim = 4
num_steps = 10

series=read_csv('datipml.csv', header=None,names=["Data"])
plot_acf(series, title="original")
pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=None,names=["Data"])

temp=x.Data.tolist()
k=2
data=[]
for i in range(len(temp)):
    data.append(temp[i]*1.)

data=torch.tensor(data)
datatrain=data[0:245]
datatest=data[245:]
print(data)
k=6
N=2
    

def hmm(data,k):
    N=len(data)
    
    if k==6:
        transition_prior=pyro.param("t_prior",torch.tensor([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]])+1,dtype=torch.float64)
        emission_prior=pyro.param("e_prior",torch.tensor([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])+1,dtype=torch.float64)
        transition_prob = pyro.sample("transition", dist.Dirichlet(transition_prior))
        emission_prob = pyro.sample("emission", dist.Dirichlet(emission_prior))
        

    else:
        transition_prior = pyro.param("t_prior",torch.ones(k,k)/k) # 3 transition priors
        emission_prior = pyro.param("e_prior",torch.tensor(torch.ones(k,N)/N)) # 10 emission priors
        transition_prob = pyro.sample("transition", dist.Dirichlet(transition_prior))
        emission_prob = pyro.sample("emission", dist.Dirichlet(emission_prior))
    # start with first weather category
    Z = 0 
    # evolve in time
    for t,y in pyro.markov(enumerate(data)):
        Z = pyro.sample(f"z_{t}", dist.Categorical(transition_prob[Z]),infer={"enumerate": "parallel"})
        pyro.sample(f"y_{t}", 
                    dist.Categorical(emission_prob[Z]), 
                    obs=y)

'''
kernel = NUTS(hmm, jit_compile=True, ignore_jit_warnings=True)
mcmc = MCMC(kernel, num_samples=100, warmup_steps=100, num_chains=1)
mcmc.run(data, k)
posterior = mcmc.get_samples()
'''

#hmm_guide = AutoNormal()
guide = autoguide.AutoDelta(poutine.block(hmm, expose=["transition", "emission"]))

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(hmm, guide, optimizer, loss=Trace_ELBO())
n_steps = 500
# do gradient steps
for step in range(n_steps):
    svi.step(data,k)
    if step % 100 == 0:
        print('.', end='')

fitted = Predictive(hmm,None,guide,num_samples=1)(datatrain,k)

'''      
t_prior=pyro.param("t_prior")
e_prior=pyro.param("e_prior")        
y=modelpred(data,A,B)
ytest=y[245:]
yhat_train=y[:245]
resid=(torch.tensor(yhat_train)-torch.tensor(datatrain)).numpy()
print(" BDS residual of N=: ", k, " is ",torch.mean(torch.tensor((bds(resid,12))[1])))
#print("Test error:",torch.mean(abs(torch.tensor(datatest)-torch.tensor(ytest))))
ytest=y[245:]
plot_acf(pd.DataFrame(torch.tensor(data)-torch.tensor(y)), title="Weiner unit residuals")
'''
