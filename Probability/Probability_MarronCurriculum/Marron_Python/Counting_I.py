#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:15:03 2025

@author: bmarron18
"""

# %%

'''
Counting:
    NumPy, Pandas, and SciPy for stats

    https://docs.scipy.org/doc/scipy/reference/stats.html


'''
# %%

'''
Pandas Internal Tests
skip
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
Greek letters Unicode

'''
    # theta θ
    ctrl+Shift+U 03B8  
    
# %%

'''
Statistical Distros
'''

import math
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, binom, beta

# --- Statistical functions (scipy.stats) --------------------

# Hypergeometric (discrete)
    scipy.stats.hypergeom
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
    
    # Prob Mass Fxn
    # hypergeom.pmf(k, M, n, N, loc=0)
    * M == ct of all entities in a set; two types
    * N == ct of Type I entities
    * M-N == ct of Type II entities
    * n == ct of draws/trials/slots
    * k == ct of Type I in n draws/trials/slots

    # Cumulative distribution function.
    # hypergeom.cdf(k, M, n, N, loc=0)

    # Random variates.
    # hypergeom.rvs(M, n, N, loc=0, size=1, random_state=None)
	

Example 1
    There are 11 balls in an urn; 7 are white and 4 are red. The balls have 
    no unique labels and are identical except for color. The Principle of Indifference
    applies.5 balls are pulled at random from the urn.
    
    (1) What are all of the observable collections of outcomes that are possible given that 
    there are no unique labels on the balls? That is, how many observable collection categories
    are possible regardless of whether the balls carry unique labels or not.
    
        Observable Collection Categories of Outcomes
        ---------------------------------------------
        4 Type I; 1 Type II
        3 Type I; 2 Type II
        2 Type I; 3 Type II
        1 Type I; 4 Type II
        0 Type I; 5 Type II

    (2) Iff the balls carried unique labels AND a single outcome is defined as drawing a 
    single ball from the urn and placing it, sequentially, in a set of 5 slots, how many 
    unique sequences could be genrated? That is, there would be 11 options for drawing the
    first ball, 10 options for drawing the second ball, 9 options for drawing the third ball,
    ..., and 7 options for drawing the fifth ball. 
    
    ct of possible unique sequences = n! / (n-5)!
                                    = 11*10*9*8*7
    
    
    
     * M = 11 (ct of all entities in a set; two types)
     * N = 4 Type I entities
     * M-N = 7 Type II entities
     * n = 5 (ct of draws/trials/slots)
     * k =2 (ct of Type I in n draws/trials/slots)
     
 

hypergeom.pmf(2, 11, 5, 4, loc=0)
	




# Binomial (discrete)
     scipy.stats.binom
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom
       
    # binom.pmf(k, n, p, loc=0)  <== prob mass fxn
    #   n = ct of slots/trials/draws that result from sampling
    #   k = ct of type a entities 
    #   n-k = ct of type b entties
    
    #   p = prob of type a entity
    #   1-p = prob of type b entity
    #   loc=0 <== endpoint of distro
    
    # binom.ppf(q, n, p, loc=0)  <== % point fxn, percentiles (inverse pmf)
    #   q = percentile
  
    
    # NB. Inverses of pmf and percentiles
n, p = 20, 0.10
binom.ppf(0.12157665459056918, n, p)    #<== gives np.float64(0.0)
binom.pmf(0, n, p)                      #<== gives np.float64(0.12157665459056918)

binom.ppf(0.2701703435345983, n, p)     #<== gives np.float64(1.0)
binom.pmf(1, n, p)                      #<== gives np.float64(0.2701703435345983)

binom.ppf(0.28517980706429846, n, p)    #<== gives np.float64(2.0)
binom.pmf(2, n, p)                      #<== gives np.float64(0.28517980706429846)


    # NB the discrete distros are binned, thus
binom.ppf(0.28, n, p)                    #<== gives np.float64(1.0)
binom.ppf(0.29, n, p)
binom.ppf(0.39, n, p)
binom.ppf(0.399, n, p)                  #<== gives np.float64(2.0)


    
# Beta (continuous)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta

    # beta.ppf(q, a, b, loc=0, scale=1) <== % point fxn, percentiles (inverse cdf)
    # beta.pdf(x, a, b, loc=0, scale=1) <== prob density fxn
    # beta.cdf(x, a, b, loc=0, scale=1) <== cumulative density fxn (the integral of the pdf)

     # NB. Inverses of cdf and percentiles (NOT of pdf)
a, b = 2, 20
beta.cdf(0.05, a, b, loc=0, scale=1)                # <== gives 0.2830281551829153
beta.ppf(0.2830281551829153, a, b, loc=0, scale=1)  # <== gives 0.05



# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
'''

# ---Counting in Discrete Outcome Spaces --------------------------------------

# --- Outcome Space from Actual Sampling -----------------------------------------  
    # If take a sample of 20 people and the output for each person is either
    # type a or type b (entities with two possible characteristics) then there 
    # is/are 1048576 possible unique sequences (permutations) for the data
    #       ==> 2^20 == 2**20 (python)
    # The output space of unique sequences is now closed (with 1048576 ct of 
    # unique sequences)
    
    # Imagine that each of the slots/positions of the sequences are also identifiable.
    # Then there 20! arrangements for each possible unique sequence. But actually
    # realizing these arrangements (combinations) would lead to a set of the unique
    # sequences that share equal cts of type a and type b characteristics. If slot 
    # position is unimportant (ie interest is focused only on the characterics) then
    # the unique sequence in a set of combinations generated by 20! are degenerate.
    # That is they contain equal cts of type a and type b characteristics. Note that
    # all of the sets of combinations generated by 20! are symmetric about the cts 
    # of type a and type b characteristic. For example, there are 20 sequences in the
    # set where there is 1 type a and 19 type b. And symmetrically, there are 20
    # sequences in the set where there are 19 type a and 1 type b.
    
    # To find the ct of degenrate sequences in a set:
    #   1 sequence with 0 infected / 20 non-infected 
    #       == 20!/0!20! == math.comb(20,0) in python
    #   20 sequences with 1 infected / 19 non-infected
    #       == 20!/1!19! == math.comb(20,1)
    #   190 sequences with 2 infected / 18 non-infected
    #       == 20!/2!18! == math.comb(20,2)
    #   1140 sequences with 3 infected / 17 uninfected 
    #       == 20!/3!17! == math.comb(20,3)
    #   ...
    #   1140 sequences with 17 infected / 3 uninfected 
    #       == 20!/3!17! == math.comb(20,17)
    #   190 sequences with 18 infected / 2 non-infected
    #       == 20!/2!18! == math.comb(20,18)
    #   20 sequences with 19 infected / 1 non-infected
    #       == 20!/1!19! == math.comb(20,19)
    #   1 sequence with 20 infected / 0 uninfected
    #       == 20!/20!0! == math.comb(20,20)

# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
(cont'd)
'''


# --- Indiv. Plots of the Sampling Model Based on Prior Info --------------------------
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import binom
from plotnine import  ggplot, aes, labs, geom_col, geom_bar, ggsave, position_dodge2

    # Binomial distro for possible values of θ
# n, p = 20, 0.05     # <== theta1.jpeg
n, p = 20, 0.10      # <== theta2.jpeg
#n, p = 20, 0.20      # <== theta3.jpeg

#x = np.arange(binom.ppf(0.01, n, p), binom.ppf(1.0, n, p))
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(1.0, n, p))
y = binom.pmf(x, n, p)
df = pd.DataFrame({'x':x,'y':y})

    # plot a single dataset 
theta2 = (ggplot()
    + geom_col(df, aes(x="x", y="y"), width=0.25)    
    + labs(
        x="Infected",
        y="Prob"
    )
)

    # see plot in "Plots" window Spyder
theta2

    # saves in current working directory (/home/bmarron18)
#ggsave(theta3, "theta3.jpeg")

    # saves to ~/Desktop
doc_dir = "/home/bmarron18/Desktop"
OUTPUT_FILE = "theta2.jpeg"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)
ggsave(theta2, output_f)


# %%

'''
Hoff, 1.2.1 Est. prob of a rare event
(cont'd)
 https://www.sthda.com/english/wiki/ggplot2-barplots-quick-start-guide-r-software-and-data-visualization
 https://jeroenjanssens.com/plotnine/
'''


# --- Single Plot of all of the Sampling Models Based on Prior Info --------------------------
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import binom
from plotnine import (ggplot, aes, labs, geom_col, 
                      geom_bar, ggsave, position_dodge2,
                      scale_fill_grey, scale_colour_grey,
                      theme_bw, scale_fill_manual, scale_color_brewer)



    # quick way to set limits of x for the distro
n, p = 20, 0.10       
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(1.0, n, p))

    # Binomial distro for possible values of θ
n1, p1 = 20, 0.05     # <== theta1.jpeg
n2, p2 = 20, 0.10      # <== theta2.jpeg
n3, p3 = 20, 0.20      # <== theta3.jpeg

y1 = binom.pmf(x, n1, p1)
y2 = binom.pmf(x, n2, p2)
y3 = binom.pmf(x, n3, p3)

df1 = pd.DataFrame({'x':x,'y':y1,'theta':p1})
df2 = pd.DataFrame({'x':x,'y':y2, 'theta':p2})
df3 = pd.DataFrame({'x':x,'y':y3, 'theta':p3})

combined_df = pd.concat([df1, df2, df3])

    # plot all datasets in a single graph
prior_thetas = (ggplot(combined_df, aes(x='x', y='y', fill='theta'))
#   + geom_bar(mapping=aes(x='x', y='y', fill='theta'), stat="identity",
   + geom_bar(stat="identity",
              position = position_dodge2(preserve = "single"),
               width=1.0
               )
#   + scale_colour_grey(start = 0.0, end = 0.8,)
#    + scale_fill_manual(values={'p1':'red', 'p2':'blue', 'p3':'green'})
#    + scale_color_brewer()
    + theme_bw()
    + labs(title = "Plotting Multiple Datasets",
           x = "ct Infected",
           y = "Probability")
    )



    # save in current working directory (/home/bmarron18)
#ggsave(prior_thetas, "thetas.jpeg")


    # save to ~/Desktop
doc_dir = "/home/bmarron18/Desktop"
OUTPUT_FILE = "prior_thetas.jpeg"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)
ggsave(prior_thetas, output_f)


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
   
    # Pr (theta < 0.10)
beta.cdf(0.1, a, b, loc=0, scale=1)    
    
    
    # # Pr (0.05 < θ < 0.20)
(beta.cdf(0.2, a, b, loc=0, scale=1)) - (beta.cdf(0.05, a, b, loc=0, scale=1))



# %%















