# %%
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 19 2025
Modified  18 May 2026
@author: bmarron
"""

This script must be run in a virtual envirnment:
    /home/bruce-mx/spyder-6/envs/ai-apis/bin/python3.12
    
into which the following packages have been installed:

    $ cd ~/spyder-6/envs
	$ python3 -m venv ai-apis
	$ source ./ai-apis/bin/activate
    
    (ai-apis):~$ pip install spyder-kernels google-genai openai
    (ai-apis)$ pip install pandas NumPy python-dateutil pytz tzdata
    (ai-apis)$ pip install "pandas[performance]"
    (ai-apis)$ pip install "pandas[computation]"
    (ai-apis)$ pip install plotnine

    (ai-apis):~$ deactivate

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
General Query Type I
Model ==> gpt-5.5

'''


from openai import OpenAI

from pathlib import Path
import os



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

    # Select/Unselect as needed


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

### ggplots ####

'''

Import .csv and create graphics
    
'''

from pathlib import Path
import os

import math
import numpy as np
import pandas as pd

from plotnine import (ggplot, aes, labs, geom_col, 
                      geom_bar, ggsave, position_dodge2,
                      scale_fill_grey, scale_colour_grey,
                      theme_bw, scale_fill_manual, scale_color_brewer)
 
    # find INPUT FILE
doc_dir = "/home/bruce-mx/Desktop"
INPUT_FILE = "clean01.csv"
input_filepath = os.path.join(doc_dir, INPUT_FILE)

input_f = Path(input_filepath)


    # Read in data
df1 = pd.read_csv(input_f)


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

    # saves in current working directory (/home/bruce-mx)
#ggsave(theta3, "query01.jpeg")

    # saves to ~/Desktop
OUTPUT_FILE = "query01.jpeg"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)
ggsave(data1, output_f)




# %%

'''
Create a stat distribution that runs from 0.1 to 15.1 and is skewed toward 15.1 over a 25-year 
interval. Models an increasingly severe El Niño with subsequent crop loss (percentages)

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the distribution
low = 0.1
high = 15.1
years = 26

# We want it skewed toward 15.1.
# In a Beta distribution, alpha > beta skews it toward the upper bound (1).
# Let's pick alpha=5 and beta=2 for a moderate skew.
alpha, b = 5, 2

# Generate 20 samples (one for each year) sorted to represent a "progression" or just 20 random samples.
# The user said "over a 20 year interval", which usually implies a time series.
# If it's a trend, we might want it to increase over time.
# But "stat distribution" usually implies a set of values.
# Let's generate 20 values sampled from this distribution.

x = np.linspace(0, 1, years)
# To show a distribution skewed towards 15.1, we can sample 20 values.
# However, for a 20-year interval, showing yearly values makes sense.
samples = beta.rvs(alpha, b, size=years)
# Scale to 0.1 - 15.1
scaled_samples = low + (samples * (high - low))

# To make it look like a "progression" skewed toward the end, we can sort them or use a trend.
# But the prompt asks for a "distribution". I'll provide 20 points.
print("Yearly Values:")
for i, val in enumerate(np.sort(scaled_samples), 1):
    print(f"Year {i}: {val:.2f}")
    
    
# %%


    # Assuming scaled_samples exists as an  array/list
sorted_samples = np.sort(scaled_samples)

    # Create a df directly
df2 = pd.DataFrame({
    "Year": range(1, len(sorted_samples) + 1),
    "Value": sorted_samples
})


    # Multiply columns df1 and df2 and store back in df1
df1['EnvStress'] = (df1['yield'].values) - (df1['yield'].values * df2['Value'].values * 0.01)





# %%
import pandas as pd
from plotnine import ggplot, aes, geom_line


# 1. Reshape to 'Long' format w/ new variable headers
df_long = df1.melt(id_vars=['year'], var_name='Yield', value_name='Climate')

# 2. Plot using ggplot/plotnine
print(
    ggplot(df_long, aes(x='Year', y='Yield', color='Climate')) 
    + geom_bar()
)

 # plot a single dataset 
data2 = (ggplot()
    + geom_col(df_long, aes(x='Year', y='Yield', color='Climate'), width=0.25)    
    + labs(
        x="Year",
        y="Yield"
    )
)

    # see plot in "Plots" window Spyder
data2


# %%

    # Misc

df1 = df1.drop(columns=['EnvStressYield'])


    # rename df2 column
df1 = df1.rename(columns={"yield": "Actual"})


       # round new column to one decimal
df2['yield'] = df2['yield'].round(1)
