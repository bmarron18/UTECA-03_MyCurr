#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:15:03 2025

@author: bmarron18
"""

# %%

'''
    NumPy, Pandas, and SciPy for stats

    https://docs.scipy.org/doc/scipy/reference/stats.html


'''
# %%

'''
Pandas Internal Tests
'''

import pandas as pd
from pathlib import Path
import os



    # create paths to files
    # Retrieve files as PosixPaths
    
doc_dir = "/home/bmarron18/Desktop"
OUTPUT_FILE = "PandaTests.txt"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)

test = pd.test() 

   # Output
with open(output_f, "w", encoding="utf-8") as f:
     f.write(test)
# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
'''
# --- Statistical functions (scipy.stats) --------------------
# Binomial (discrete)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom
    
    # binom.ppf(q, n, p, loc=0)  <== % point fxn, percentiles (inverse of cdf)
    # binom.pmf(k, n, p, loc=0)  <== prob mass fxn
    #   k = count of successes/type a
    #   n = total count of trials/slots
    #   p = prob of a success/type a
    #   1-p = prob of a failure/type b
    #   loc=0 <== endpoint of distro
    
    # ctrl+Shift+U 03B8  θ
    
# Beta (continuous)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta

    # beta.ppf(q, a, b, loc=0, scale=1) <== % point fxn, percentiles (inverse of cdf)
    # beta.pdf(x, a, b, loc=0, scale=1) <== prob density fxn
    # beta.cdf(x, a, b, loc=0, scale=1) <== cumulative density fxn (the integral of the pdf)



# --- Problem Statement and Prior Info --------------------------------------

    # WANT TO KNOW: fraction/% of infected in the city 
    #   == ct all infected/ct of total pop 
    #   == Y/Z == θ (true) (unk)
    
    # PRIOR INFO:
    #   θ (true) may be [0,1] (thus, a rv)
    #   Other cities report/believe θ (true) == [0.05, 0.20] centered ~0.10


# --- Technical/Modeling Approach ---------------------------------------
    # Estimate θ (true) from a (random) sample of the city pop 
    #   == ct all infected in (random) sample/ct of (random) sample from pop 
    #   == y^/z^ == θ^

    # Can set z^ (sample size) but the value of  y^ is TBD (thus, a rv)
    #   z^ == 20
    # ASSUME a prob model for the rv y^ (sampling model) 
    #   == p(y^|θ^) == binomial dist
    
    # θ (true) is UNK (thus, a rv) with a parameter space [0,1]
    # ASSUME a prob model for the rv θ (true) (prior model)
    #   == p(θ (true)) == beta(2, 20)


# --- Outcome Space from Actual Sampling -----------------------------------------  
    # If take a sample of 20 people and the output for each person is either
    # type a or type b then there is/are
    #   1048576 possible unique sequences (permutations) for the data
    #       == 2^20 == 2**20
    #   1 sequence with 0 infected == 20!/0!20! == math.comb(20,0)
    #   20 sequences with one infected == 20!/1!19! == math.comb(20,1)
    #   190 sequences with two infected == 20!/2!18! == math.comb(20,2)
    #   1140 sequences with three infected == 20!/3!17! == math.comb(20,3)
    #   4845 sequences with four infected == 20!/4!16! == math.comb(20,4)
    #   15504 sequences with five infected == 20!/5!15! == math.comb(20,5)
    #   ...
    #   1 sequence with 20 infected == 20!/20!0! == math.comb(20,20)

# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
(cont'd)
'''


# --- Plot Various pmfs of the Sampling Model --------------------------
import math
import numpy as np
import pandas as pd
from scipy.stats import binom
from plotnine import ggplot, aes, labs, geom_line, geom_col, ggsave

    # 
# n, p = 20, 0.05
# n, p = 20, 0.10
n, p = 20, 0.20

#x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
x = np.arange(binom.ppf(0.001, n, p), binom.ppf(1.0, n, p))
y = binom.pmf(x, n, p)
df = pd.DataFrame({'x':x,'y':y})


theta3 = (
    ggplot(df)
    + aes(x="x", y="y")
    + labs(
        x="Infected",
        y="Prob",
    )
#    + geom_line()
    + geom_col()
)

    # see plot in "Plots" window Spyder
theta3

    # saves in current working directory (/home/bmarron18)
ggsave(theta3, "theta3.jpeg")



# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
(cont'd)
'''

import math
import numpy as np
import pandas as pd
from scipy.stats import binom, beta
from plotnine import ggplot, aes, labs, geom_line, geom_col, ggsave


# --- Plot pdf of the Prior --------------------------

a, b = 2, 20
x = np.linspace(beta.ppf(0.0, a, b), beta.ppf(1.0, a, b), 100)
#x = np.linspace(0, 1.0, 100)
y = beta.pdf(x, a, b)
df = pd.DataFrame({'x':x,'y':y})


theta_prior = (
    ggplot(df)
    + aes(x="x", y="y")
    + labs(
        x="theta",
        y="???",
    )
    + geom_line()
#    + geom_col()
)

    # see plot in "Plots" window Spyder
theta_prior

    # saves in current working directory (/home/bmarron18)
ggsave(theta_prior, "theta_prior.jpeg")


# --- Prob Calcs of the Prior --------------------------
# https://www.statology.org/how-to-use-the-beta-distribution-in-python/

    # mean
mean = beta.mean(a, b, loc=0, scale=1)
    
    
    # mode (max in pdf)
df['y'].max()       # <== the max point (y-value) in the distro
df['y'].idxmax()    # <== location (index) of the max in the df
df.loc[5]           # <== values of the x,y coords at the location
                    #   x-value is the (approx) mode

    # NB. Inverses of cdf / percentiles (NOT of pdf)
beta.cdf(0.05, a, b, loc=0, scale=1)                # <== gives 0.2830281551829153
beta.ppf(0.2830281551829153, a, b, loc=0, scale=1)  # <== gives 0.05

   
    # Pr (theta < 0.10)
beta.cdf(0.1, a, b, loc=0, scale=1)    
    
    
    # # Pr (0.05 < θ < 0.20)
(beta.cdf(0.2, a, b, loc=0, scale=1)) - (beta.cdf(0.05, a, b, loc=0, scale=1))



# %%















