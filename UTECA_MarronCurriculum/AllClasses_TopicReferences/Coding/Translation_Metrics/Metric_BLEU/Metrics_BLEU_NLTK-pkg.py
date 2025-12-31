#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:35:19 2025
Updated: 13 Oct 2025
@author: bmarron
"""

# %%

    # after import NLTK package, need to download all the corpus, etc files
    # the command below opens a separate window to select items
    # select all
    
import nltk
nltk.download()



# %%

'''
The BLEU (Bilingual Evaluation Understudy) Metric 
using the Natural Language Tool Kit (ntlk) in Python

https://www.nltk.org/howto/bleu.html

'''

The BLEU score requires:
	Language A-to-Language B translation by a prof. translator and an LLM
	reference texts ==> the human translations or summaries in Language B
	candidate text ==> the machine-generated text in Language B

# %%
'''
BLEU Score: Test Run
'''

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize


    # Define reference texts 
reference_text_1 = "The quick brown fox jumps over the lazy dog. \
    And finds himself caught." 
reference_text_2 = "A fast brown fox leaps over a lazy dog. \
    And is captured."

    # Define translated (candidate) text
candidate_text =  "The quick brown fox hops over the slow-moving dog. \
    And gets caught."


   # make list of the refernce sentences and then tokenize to words
sentences = [reference_text_1, reference_text_2]
references = [word_tokenize(sentence, "english") for sentence in sentences]

    # only one candidate sentence so can tokenize directly to words 
candidate = word_tokenize(candidate_text, "english")

    # set the n-gram weights
weights=(.50, 0.50, 0, 0)


bleu_score = sentence_bleu(references, candidate, weights)

 
print(bleu_score)
# 0.6504436355879909


# %%
'''
BLEU Score: Evaluation of a Small Amount of Text (a few sentences)
using the "sentence_blue" method
'''

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize


    # Define reference texts (human translator)
reference_text_1 = "The quick brown fox jumps over the lazy dog. \
    And finds himself caught." 
reference_text_2 = "A fast brown fox leaps over a lazy dog. \
    And is captured."

    # Define candidate text (AI translator)
candidate_text =  "The quick brown fox hops over the slow-moving dog. \
    And gets caught."


   # make Python list of the refernce sentences and then tokenize to words
sentences = [reference_text_1, reference_text_2]
references = [word_tokenize(sentence, "english") for sentence in sentences]

    # only one candidate sentence so can tokenize directly to words 
candidate = word_tokenize(candidate_text, "english")

    # set the n-gram weights
weights=(.50, 0.50, 0, 0)


bleu_score = sentence_bleu(references, candidate, weights)

 
print(bleu_score)
# 0.6504436355879909




# %%

'''
BLEU Score: Evaluation of a large Amount of Text (a paragraph or so)
'''

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import sent_tokenize, word_tokenize


    # Define a reference text created by a human translator
    # The quotes define a "string" in Python
reference_text_1 = "The Ministry of Labor and Social Welfare (MLSW) must approve \
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


    # Create a list(list(string)) of tokenized words from reference_text_1
    # INCLUDES tokenized periods "."
token_refs = [word_tokenize(reference_text_1, "english")]


    # Create a list(string) of tokenized words from candidate_text
    # INCLUDES tokenized periods "."
token_cand = word_tokenize(candidate_text, "english")


    # set the n-gram weights
#weights=(1,)
#weights=(0.99, 0.01,)
#weights=(0.75, 0.25,)
#weights=(0.50, 0.25, 0.25,)
weights=(0.40, 0.20, 0.20, 0.20)



bleu_score = sentence_bleu(token_refs, token_cand, weights,
            smoothing_function=None)

bleu_score2 = sentence_bleu(token_refs, token_cand, weights,
            smoothing_function=SmoothingFunction().method0)

bleu_score3 = corpus_bleu(token_refs, token_cand, weights,
            smoothing_function=SmoothingFunction().method4)

print(bleu_score, blue_score2, blue_score3)


# %%



