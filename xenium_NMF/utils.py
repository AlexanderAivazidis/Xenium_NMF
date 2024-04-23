from typing import Optional
from pyro.infer import SVI, Trace_ELBO
import torch
import numpy as np
import pyro
from pyro.infer import Predictive
import pandas as pd
import scanpy as sc
import random
from numpy.linalg import norm
import scipy
from scipy.sparse import csr_matrix
from numpy import inner
import matplotlib.pyplot as plt
from contextlib import contextmanager
import seaborn as sns
import os,sys

def G_a(mu, sd):
    # Converts mean and sd for Gamma distribution into parameter
    return mu**2/sd**2

def G_b(mu, sd):
    # Converts mean and sd for Gamma distribution into beta parameter
    return mu/sd**2