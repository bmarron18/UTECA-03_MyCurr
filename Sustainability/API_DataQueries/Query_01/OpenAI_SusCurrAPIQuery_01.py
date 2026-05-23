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
