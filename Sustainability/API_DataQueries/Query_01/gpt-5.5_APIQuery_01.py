# %%
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 19 2025
Modified  18 May 2026
@author: bmarron
"""

This script must be run in a virtual envirnment:
    /home/bruce-mx/spyder-6/envs/ai-apis/bin/python3.12

    $ cd ~/spyder-6/envs
	$ python3 -m venv ai-apis
	$ source ./ai-apis/bin/activate


the following packages have been installed into (ai-apis:

    (ai-apis)$ pip install spyder-kernels google-genai openai
    (ai-apis)$ pip install pandas NumPy python-dateutil pytz tzdata
    (ai-apis)$ pip install "pandas[performance]"
    (ai-apis)$ pip install "pandas[computation]"
    (ai-apis)$ pip install plotnine
    (ai-apis):~$ deactivate

"pandas[performance]" contains:
    numexpr
    bottleneck
    numba
    
"pandas[computation]" contains:
    SciPy
    xarray


Successfully installed NumPy-2.4.6 pandas-3.0.3 pytz-2026.2 tzdata-2026.2
Successfully installed bottleneck-1.6.0 llvmlite-0.47.0 numba-0.65.1 numexpr-2.14.1
Successfully installed scipy-1.17.1 xarray-2026.4.0
Successfully installed contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 kiwisolver-1.5.0 matplotlib-3.10.9 \
mizani-0.14.4 patsy-1.0.2 pillow-12.2.0 plotnine-0.15.4 pyparsing-3.3.2 statsmodels-0.14.6

# %%

### Mime-type files ####

'''
Document MIME types available for OpenAI output:
    text/plain   ==> .txt
    text/html    ==> .html
    text/json    ==> ,json
    text/x-tex   ==> .tex
    text/csv    ==> .csv
    
'''

# %%


'''
Load pkgs
'''

from openai import OpenAI

from pathlib import Path
import os

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import beta

from plotnine import (ggplot, aes, labs, geom_col, facet_grid,
                      geom_bar, ggsave, position_dodge2,
                      scale_fill_grey, scale_colour_grey,
                      theme_bw, scale_fill_manual, scale_color_brewer)



# %%

'''
General Query Type I
Model ==> gpt-5.5

'''

    # API_KEY is saved as an ENV VARIABLE on home compu
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

    # API_KEY can be inserted directly
#client = OpenAI(api_key="ACTUAL_API_KEY")


    # label of output file
OUTPUT_FILE = "SusCurrQuery01_gpt-5.5.txt"
    
    
    # set up the file paths for the OUTPUT_FILE
    # set the file path to your Desktop
    # Path() represents file+directory paths in a platform-independent manner.
doc_to_print = OUTPUT_FILE
doc_dir = "/home/bruce-mx/Desktop"   #<== Old HP


    # create paths to files
    # Retrieve files as PosixPaths
output_filepath = os.path.join(doc_dir, doc_to_print)
output_f = Path(output_filepath)



	# User level message
user_prompt= "Search the internet and extract data for wheat yields from the USA for the years \
2000 - 2025. Organize this data and output the data in a .csv format"


	# Developer level message 
sys_prompt = "You are an expert agricultural researcher."



response = client.responses.create(
  model = "gpt-5.5",
  instructions = sys_prompt,
  input = user_prompt
)

with open(output_f, "w", encoding="utf-8") as f:
     f.write(response.output_text)
     
print(f"Query complete! Outputsaved to '{output_f}'.")




 
# %%


'''
Import clean data .csv 
'''

    # find INPUT FILE
doc_dir = "/home/bruce-mx/Desktop"
INPUT_FILE = "API-Output_Query01_clean.csv"
input_filepath = os.path.join(doc_dir, INPUT_FILE)
input_f = Path(input_filepath)


    # Read in data as dataframe1
    # cols:  "year", "yield"
df1 = pd.read_csv(input_f)

    #Rename cols in df1
df1 = df1.rename(columns={"yield": "Actual"})

# %%

'''
Model the effects on crop yields from an increasingly severe El Niño

Create a stat distribution that runs from 5.1 to 20.1 and is skewed toward 20.1 over a 25-year 
interval. The stat distro provides subsequent crop loss as percentages

'''


# Parameters for the distribution
low = 5.1
high = 20.1
years = 26


# In a Beta distribution, alpha (v) > beta (w) skews it toward the upper bound (1).
# v=10 and w=2 for a moderate skew.
v, w = 5, 2


# To show a distribution skewed towards 20.1, we can sample 26 values.
# However, for a 26-year interval, showing yearly values makes sense.
samples = beta.rvs(v, w, size=years)

# Scale to 5.1 - 20.1
scaled_samples = low + (samples * (high - low))

# To make it look like a "progression" skewed toward the end, we can sort them or use a trend.
# But want a "distribution". Provide 26 points.
print("Yearly Values:")
for i, val in enumerate(np.sort(scaled_samples), 1):
    print(f"Year {i}: {val:.2f}")
    
# %%

'''
OPTION 1

Crop-drop trend increases over time
'''

    # Assuming scaled_samples exists as an  array/list
sorted_samples = np.sort(scaled_samples)

    # Create a df directly using col 'year' from df1 and crop-drop percentages
df2 = pd.DataFrame({
    "year":df1['year'],
    "crop-drop": sorted_samples
})

    # Create new df1a from df1
    # Create a new col "EnvStress' in df1
    # Multiply columns df1 and df2 and store back in df1
df1a = df1
df1a['EnvStress'] = (df1a['Actual'].values) - (df1a['Actual'].values * df2['crop-drop'].values * 0.01)



# %%
'''
OPTION 2

Random hit of crop-drop
'''

    # dont sort scaled-samples
scaled_samples

    # Create a df directly using col 'year' from df1 and crop-drop percentages
df3 = pd.DataFrame({
    "year":df1['year'],
    "crop-drop": scaled_samples
})


     # Create new df1b from df1
    # Create a new col "EnvStress' in df1
    # Multiply columns df1 and df2 and store back in df1
df1b = df1
df1b['EnvStress'] = (df1b['Actual'].values) - (df1b['Actual'].values * df3['crop-drop'].values * 0.01)


 


# %%

'''
ggplot1

'''


    # Reshape to 'Long' format w/ new variable headers
df_long = df1.melt(id_vars=['year'], var_name='Climate', value_name='Yield')


 # plot a single dataset 
plot1 = (ggplot()
    + geom_col(df_long, aes(x='year', y='Yield', color='Climate'), width=0.25)    
    + labs(
        x="Year",
        y="Yield"
    )
)

    # see plot in "Plots" window Spyder
plot1


    # saves to ~/Desktop
doc_dir = "/home/bruce-mx/Desktop"
OUTPUT_FILE = "plot1.jpeg"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)
ggsave(plot1, output_f)



# %%

'''
ggplot2

'''
    #  Reshape to 'Long' format w/ new variable headers
    # toggle df1a or df1b
df_long = df1a.melt(id_vars=['year'], value_vars= ['Actual', 'EnvStress'])
#df_long = df1b.melt(id_vars=['year'], value_vars= ['Actual', 'EnvStress'])


# plot a single dataset 
plot_df1a = (ggplot()
    + geom_col(df_long, aes(x='year', y='value', color="factor(variable)"), width=0.25)
    + facet_grid(". ~ variable") # . means no rows; variable means columns
    + labs(
        x="Year",
        y="Yield"
    )
)

    # saves to ~/Desktop
doc_dir = "/home/bruce-mx/Desktop"
OUTPUT_FILE = "plot_df1a.jpeg"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)
ggsave(plot1, output_f)


# %%

    # Misc
    
range(1, len(sorted_samples) + 1)
    
# generate 26 evenly spaced values sampled from this distribution.
x = np.linspace(0, 1, years)

    # remove col from df
df1 = df1.drop(columns=['EnvStressYield'])


       # round new column to one decimal
df2['yield'] = df2['yield'].round(1)



    # plot a single dataset 
data1 = (ggplot()
    + geom_col(df1, aes(x="year", y="yield"), width=0.25)    
    + labs(
        x="Year",
        y="Yield"
    )
)

    # see plot in "Plots" window Spyder
data1



