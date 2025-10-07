#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:15:03 2025

@author: bmarron18
"""

# %%

'''
    Pandas, SciPy, NumPy for Stats

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