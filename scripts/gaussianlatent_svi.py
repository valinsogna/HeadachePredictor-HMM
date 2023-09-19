
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
import math
import os
import torch
import matplotlib.pyplot as plt
import torch.distributions.constraints as constraints
import pyro
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro.contrib.autoguide as autoguide
from pyro.infer import Predictive
figsize=(10,4)
import numpy as np

pyro.clear_param_store()


series=read_csv('datipml.csv', header=0,names=["Data"])
plot_acf(series, title="original")
pyplot.savefig("original.png",dpi=900)

pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=0,names=["Data"])

temp=x.Data.tolist()

data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]

def modelA(data,start):
    z=[]
    k=pyro.param("k", torch.tensor(1.0))
    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(start,1.0)))
        else:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(z[i-1],1)))
        pyro.sample("obs_{}".format(i), dist.Bernoulli(1/(1+torch.exp(-k*z[i]))), obs=data[i])




'''
def guideA(data,start):
    # register the two variational parameters with Pyro.
    alpha_q = pyro.param("alpha_q", torch.tensor(0.0))
    beta_q = pyro.param("beta_q", torch.tensor(1.0))
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    for i in range(len(data)):
        pyro.sample("weiner_{}".format(i), dist.Normal(alpha_q, beta_q))
'''
guideA=autoguide.AutoNormal(modelA)



def BIC(data,start,k):
    z=[]
    N=len(data)
    k=pyro.param("k", torch.tensor(1.0))
    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    s=0
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(start,1.0)))
        else:
            z.append(pyro.sample("weiner_{}".format(i),dist.Normal(z[i-1],1)))
        s=s+dist.Bernoulli(1/(1+torch.exp(-k*z[i]))).log_prob(data[i])
    
    return torch.log(torch.tensor(N))-2*s


# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(modelA, guideA, optimizer, loss=Trace_ELBO())


n_steps = 5000
tmp=svi.evaluate_loss(data,0)
losses=[]
# do gradient steps
for step in range(n_steps):
    tmpold=tmp
    loss=svi.step(data,0)
    losses.append(loss)
    tmp=svi.evaluate_loss(data,0)
    if step % 100 == 0:
        print('.', end='')
    
fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("Weiner loss");
fig.savefig("Weinerloss.png",dpi=900)



k = pyro.param("k").item()

bic=BIC(datatest,0,k)
print("BIC is" ,bic)


pred=[]
zeta=[]


    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
for i in range(len(data)):
    if i==0:
        zeta.append(pyro.sample("weiner_{}".format(i),dist.Normal(0,1.0)))
    else:
        zeta.append(pyro.sample("weiner_{}".format(i),dist.Normal(zeta[i-1],1)))
    pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(1/(1+torch.exp(-k*zeta[i])))))  





def compute_error(list1,list2):
    err=0
    for i in range(len(list1)):
        if abs(list1[i]-list2[i])>10e-02:
            err=err+1
    
    err=err/len(list1)
    return err
err=compute_error(data,pred)
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

print("Autocorrelation STD is ", np.linalg.norm(autocorr(datatest)-autocorr(predtest))/len(datatest))

plot_acf(pd.DataFrame(torch.tensor(data)-torch.tensor(pred)), title="Weiner unit residuals")
pyplot.savefig("weiner.png",dpi=900)


