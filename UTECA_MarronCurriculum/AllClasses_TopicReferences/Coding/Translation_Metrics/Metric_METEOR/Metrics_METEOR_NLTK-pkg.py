#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:35:19 2025
Updated 13 Oct 2025
@author: bmarron
"""

# %%

'''
METEOR Metric using the Natural Language Tool Kit (ntlk)

'''

# %%

    # after import NLTK package, need to download all the corpus, etc files
    # the command below opens a separate window to select items
    # select all
    
import nltk
nltk.download()



# %%

'''
METEOR Score: Test Run
'''
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


 
    # Define the reference text
reference_text= "The quick brown fox jumps over the lazy dog. \
    And finds himself caught." 
    
    # Define the translated (candidate) text
candidate_text = "A fast brown fox leaps over a lazy dog. \
    And is captured."


ref = word_tokenize(reference_text, "english")
hypo = word_tokenize(candidate_text, "english")


# Calculate METEOR score 
score = single_meteor_score(ref, hypo)

# Print the result 
print(f"METEOR Score: {score:.4f}")
# METEOR Score: 0.6621

# %%

'''
METEOR Score for Large Corpus
'''
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


 
   # Define a reference text created by a human translator
   # The quotes define a "string" in Python
reference_text= "The Ministry of Labor and Social Welfare (MLSW) must approve \
    or reject the registration request within twenty days after its receipt in the \
    offices of the MLSW. If the MLSW fails to do so, the applicants may formally \
    require that the MLSW resolve the issue within three business days. If, after \
    this second period has elapsed without notification of the status of the \
    registration request, the registration request will be deemed approved for \
    all applicable legal purposes."
    
    # Define the candidate text created by an AI translator
    # The quotes define a "string" in Python
candidate_text = "The Ministry of Labor and Social Welfare must decide on the \
    registration request within twenty days of receiving it. If it fails to do so \
    the applicants may request that it issue the corresponding resolution within three \
    days of submitting the request. After thid period has elapsed without notification \
    of the resolution, the registration will be deemed to have been made for the legal \
    purposes to which it gives rise"


ref = word_tokenize(reference_text, "english")
cand = word_tokenize(candidate_text, "english")


# Calculate METEOR score 
score = single_meteor_score(ref, cand)

# Print the result 
print(f"METEOR Score: {score:.4f}")


# %%

'''
METEOR Score for My Work
'''
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


 
   # Define a reference text created by a human translator
   # The quotes define a "string" in Python
reference_text= "Copy your reference text here. Watch the quotes!."
    
    # Define the candidate text created by an AI translator
    # The quotes define a "string" in Python
candidate_text = "Copy the AI translation here. In quotes!!"


ref = word_tokenize(reference_text, "english")
cand = word_tokenize(candidate_text, "english")


# Calculate METEOR score 
score = single_meteor_score(ref, cand)

# Print the result 
print(f"METEOR Score: {score:.4f}")


