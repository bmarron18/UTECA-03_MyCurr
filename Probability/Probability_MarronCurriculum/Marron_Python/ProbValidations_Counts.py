#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 17:15:03 2025

@author: bmarron18
"""
    
# %%

'''
Counts
    Outcome Space 
'''

from pathlib import Path
import os

import math
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, binom, beta
from itertools import combinations


# %%
'''
Example 1
    based on Jaynes, pp 

'''
    There are 11 balls in an urn; 7 are white (type a) and 4 are red (type b). The balls have 
    unique labels and are identical except for color and label. The Principle of Indifference
    applies. Five (5) balls are pulled at random from the urn.
    
    
    (1) 
    Observation Level I: 
        Can discern label AND color on each ball.
    Atomic Outcome: 
        The pull of one (1) single ball from the urn
    Atomic Outcome Space Ct: 
        11 (each ball is a unique entity per labels; colors are secondary characteristics)
    Collective Outcome: 
        A time-based sequence of five (5) atomic outcomes as a unique permutation
        of labels. The members of the set of collective outcomes are mutually exclusive and 
        exhaustive(MEE) per the observation level
    Collective Outcome Space Cts:
        11 options for the first draw
        10 options for the second draw
        9 options for the third draw
        8 options for the fourth draw
        7 options for the fifth draw
        
        11!/(11-5)! = 11*10*9*8*7 = 55440 unique sequences of labeled entities
    
        # breaking it down by collection of types
        5 type a ; 0 type b ==> 7C5              = 21
        4 type a ; 1 type b ==> 7C4 * 4C1 = 35*4 = 140
        3 type a : 2 type b ==> 7C3 * 4C2 = 35*6 = 210
        2 type a : 3 type b ==> 7C2 * 4C3 = 21*4 = 84
        1 type a : 4 type b ==> 7C1 * 4C4 = 7 * 1 = 7
        
        But there are 5! = 120 permutations for each collection of types
        
        (21*120)+(140*120)+(210*120)+(84*120)+(7*120) = 55440





    (2) 
    Observational Level II: 
        Can discern ONLY the color of each ball
    Atomic Outcome:
        The pull of one (1) single ball from the urn
    Atomic Outcome Space Ct: 
        2 (type a or type b)
    Collective Outcome: 
        A time-based sequence of five (5) atomic outcomes as a combination 
        of type a and type b cts. The members of the set of collective outcomes are
        mutually exclusive and exhaustive (MEE) per the observation level
        
    Collective Outcome Space Cts:
        5 type a ; 0 type b ==> 7C5              = 21
        4 type a ; 1 type b ==> 7C4 * 4C1 = 35*4 = 140
        3 type a : 2 type b ==> 7C3 * 4C2 = 35*6 = 210
        2 type a : 3 type b ==> 7C2 * 4C3 = 21*4 = 84
        1 type a : 4 type b ==> 7C1 * 4C4 = 7 * 1 = 7
                                            SUM = 21+140+210+84+7
                                                = 462
        
        # python
        >>>
        data = list(range(1, 12))
        ct = len(list(combinations(data, 5))  # 11C5
        print (ct)
            462
        <<<
    
    What are all of the observable outcomes that are possible given that 
    there are no unique labels on the balls? That is, how many observable 
    are possible regardless of whether the balls carry unique labels or not. The observable
    outcomes are mutually exclusive and exhaustive (MEE) 
    
     
       

    (2) Q: Iff the balls carried unique labels AND a single outcome is defined as drawing 
    five balls from the urn, one after the other,  and placing them sequentially in order 
    in a set of 5 slots, how many unique sequences could be genrated? The sequences are
    NOT exchangeable because the resolution of observability is at the level of the
    labels, regardless of other characteristics of the balls.
    A: There would be 11 options for drawing the first ball, 10 options for drawing the 
    second ball, 9 options for drawing the third ball,..., and 7 options for drawing the 
    fifth ball. 
    
    
    ct of possible unique sequences = 11! / (11-5)!
                                    = 11*10*9*8*7
                                    = 55440
    
    
    
     * M = 11 (ct of all entities in a set; two types)
     * N = 4 Type I entities
     * M-N = 7 Type II entities
     * n = 5 (ct of draws/trials/slots)
     * k =2 (ct of Type I in n draws/trials/slots)
     
 

hypergeom.pmf(2, 11, 5, 4, loc=0)
	


# %%
'''
Example 2
    from Hoff pp
'''

# --- Outcome Space from Actual Sampling -----------------------------------------  
    # If take a sample of 20 people and the output for each person is either
    # type a or type b (entities with two possible characteristics) 
    
Observation Level 1 
    # Imagine that each of the slots/positions of the sequences are also identifiable
    # or that the entities themselves are labeled: Sequences are NOT EXCHANGEABLE
    # Thus there are 1048576 possible unique sequences (permutations) for the data
    
    # The output space of unique sequences is now closed (with 1048576 ct of 
    # unique sequences)
    
 2^20 = 1048576 unique sequences in the outcome space
 
     # python
 >>> 
 2**20
 >>>
    
Observational Level 2 (Jaynes pp 62)
    # The labels on individual entities is unimportant or not observable, only the 
    # chracteristics are important or observable. Then the set of sequences is an EXCHANGEABLE.
    # DISTRIBUTION of sequences. That is, only the TOTAL CTS of type a and type b  
    # in any sequence are important or observable; the actual location/position  of 
    # characteristics in the sequence is not important.
    
    # Take any one of the 1048576 unique sequences. This unique sequence will have either 
    # a type a characteristic or a type b characteristic in/at each of the 20 slots/positions
    # in the sequence. For example, the unique sequence
    #      a1a2a3a4b5a6a7a8a9a10a11a12a13b14a15a16a17a18a19a20

>>>
"""
If the output from your iteration is very large, or you only need to process one item at a
time without storing the entire result in memory, you can use a generator function. This 
allows for "lazy" evaluation, yielding results as needed.

Fxn: generate_large_permutations(items) 
Generates permutations of a list of items using itertools.permutations.
Args:
  items: The list or iterable for which to generate permutations.
Yields:
  Each permutation as a tuple.
"""

def generate_large_permutations(items):
  yield from itertools.permutations(items)

# a large list
sequence = ['a1', 'a2', 'a3', 'a4','b5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'b14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']

# Create the generator
permutation_generator = generate_large_permutations(sequence)

# Iterate through the permutations without storing them all in memory
print("First 3 permutations:")
for i, perm in enumerate(permutation_generator):
  if i == 20081700:
      print(perm)

>>>




>>> python
    # create paths to files
    # Retrieve files as PosixPaths
    
doc_dir = "/home/bmarron18/Desktop"
OUTPUT_FILE = "PandaTests.txt"
output_filepath = os.path.join(doc_dir, OUTPUT_FILE)
output_f = Path(output_filepath)

output_data = []
sequence = ['a1', 'a2', 'a3', 'a4','b5', 'a6']
all_permutations = permutations(sequence)

  # Output
for i in all_permutations:
    result = i
    output_data.append(result) 
    
with open(output_f, "w", encoding="utf-8") as f:
     f.write(output_data)
>>>
   
    # There are 20!=2432902008176640000 arrangements for this possible unique sequence. Realizing
    
>>>
math.factorial(20)

>>>
    # these arrangements would generate a subset from the complete set of 1048576
    # sequences that all share EQUAL CTS of type a and type b characteristics. 
    # The subset contains degenrate sequences, those sequences that contain equal cts of
    # type a and type b characteristics. Degenerate sequences are combinations.
    
    # 
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













