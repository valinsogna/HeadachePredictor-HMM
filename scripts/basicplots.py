#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:29:02 2022

@author: cyberguli
"""

from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro import poutine
from pyro.optim import Adam
import pyro.contrib.autoguide as autoguide
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.infer import Predictive

import numpy as np
figsize=(10,4)
pyro.clear_param_store()




series=read_csv('datipml.csv', header=0,names=["Data"])
fig, ax = plt.subplots()
plt.plot(series,'o')
plt.xlabel("time")
plt.ylabel("Headache dummy variable")
plt.title("Data")
ax.set_yticks(np.arange(0, 2, 1))
pyplot.savefig("data.png",dpi=900)

fig, axes = plt.subplots(1)
axes.set_xlabel('Lag')
axes.set_ylabel('Autocorrelation')
fig=plot_acf(series, title="ACF of the data", ax=axes)
pyplot.savefig("original.png",dpi=900)

pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=0,names=["Data"])

temp=x.Data.tolist()

data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]