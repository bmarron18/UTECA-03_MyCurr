#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:35:19 2025

@author: bmarron
"""

# %%

'''
Using the Natural Language Tool Kit (ntlk)
[see installation in 'Install-Info_NLTK-package.txt']

'''

# %%

    # after impot package, need to download all the corpus, etc files
    # opens a separate window to select items
    # select all
    
import nltk
nltk.download()

# %%

'''
NLTK Test Code

'''

from nltk.translate.bleu_score import sentence_bleu

    # Define your desired n-gram weights
    # Weights for uni-gram, bi-gram, tri-gram, and 4-gram
weights = (0.25, 0.25, 0, 0) 

    # Reference and predicted texts
    # These are tokenized lists (not tokenized strings)
reference = [["the", "picture", "is", "clicked", "by", "me"],
             ["this", "picture", "was", "clicked", "by", "me"]]
predictions = ["the", "picture", "the", "picture", "by", "me"]

    # Calculate BLEU score with weights
score = sentence_bleu(reference, predictions, weights=weights)
print(score)
# 0.7186082239261684


# %%

from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import sent_tokenize, word_tokenize


candidate_text =  "The Ministry of Labor and Social Welfare must decide on the \
    registration request within twenty days of receiving it. If it fails to do so \
    the applicants may request that it issue the corresponding resolution within three \
    days of submitting the request. After thid period has elapsed without notification \
    of the resolution, the registration will be deemed to have been made for the legal \
    purposes to which it gives rise"
    
    # creates a list(string) of tokenized words 
    # EXCLUDES periods "."
split_fxn =  "The Ministry of Labor and Social Welfare must decide on the \
    registration request within twenty days of receiving it. If it fails to do so \
    the applicants may request that it issue the corresponding resolution within three \
    days of submitting the request. After thid period has elapsed without notification \
    of the resolution, the registration will be deemed to have been made for the legal \
    purposes to which it gives rise".split()
 
    
    # creates a list(string) of tokenized words 
    # INCLUDES tokenized periods "."
token_fxn2 = word_tokenize(candidate_text, "english")


   # creates a list(list(string)) of tokenized words 
   # EXCLUDES tokenized periods "."
token_fxn3 = [word_tokenize(candidate_text, "english")]


    # creates a list(string1, string2, string3) of tokenized sentences
    # EXCLUDES tokenized periods "."
token_fxn4 = sent_tokenize(candidate_text)


token_fxn5 = word_tokenize(str(token_fxn4), "english")

token_fxn6 = [word_tokenize(sentence, "english") for sentence in candidate_text]


# %%

from nltk.tokenize import sent_tokenize

    # tokenizes individual strings to a SINGLE list

text = "NLTK is a great NLP toolkit. It makes processing text easy!\
    This is a single string with three sentences"
sentences = sent_tokenize(text)
print(sentences)


# %%
from nltk.tokenize import sent_tokenize

    # This is a list of three(3) strings with sentences in each string
text_list = [
        "This is the first sentence. This is the second sentence in the first string.",
        "Another string of text. It has multiple sentences too.",
        "A single sentence string."
    ]
    
    #create empty list
tokenized_sentences_list = []


for text in text_list:
        sentences = sent_tokenize(text)
        tokenized_sentences_list.append(sentences)
        
print(tokenized_sentences_list)

# %%

from nltk.tokenize import word_tokenize

sentences = [
        "This is the first sentence.",
        "And this is the second one.",
        "NLTK makes tokenization easy."
    ]

tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

print(tokenized_sentences)


# %%

'''
Tokenizing
https://www.nltk.org/howto/tokenize.html

'''
from nltk.tokenize import sent_tokenize, word_tokenize

    # text is a single string with multiple sentences
text = "NLTK is a powerful library for natural language processing. \
    It's widely used! You can do a lot with it. And a third sentence."

sentences = sent_tokenize(text, "english")
words = word_tokenize(text, "english")

print(sentences)

#print(words)


# %%
'''
Tokenizing

'''

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

text = "NLTK is a powerful library for natural language processing. It's widely used! You can do a lot with it."
sentences = sent_tokenize(text)
sent_strg = "".join(sentences)
words = word_tokenize(sent_strg)

print(sentences)

print(words)
# %%

'''
Check tokenized output

'''

import os
from nltk.tokenize import sent_tokenize, word_tokenize

    # Setup for output
OUTPUT_FILE = "tokenized_refs.txt"

doc_to_print = OUTPUT_FILE;
doc_dir = "/home/bmarron18/Desktop" ;
output_filepath = os.path.join(doc_dir, doc_to_print)

    # Define reference texts
    # tokenize to sentences and then to words
reference_text_1 = "This is a long and complex reference text for evaluating the machine translation output. It contains multiple sentences and aims to provide a comprehensive example."
reference_text_2 = "Another reference text that is also long and provides a different perspective on the same topic. This helps in capturing more nuances."


    # original code
references = [
        [word_tokenize(sent) for sent in reference_text_1.split('. ') if sent],
        [word_tokenize(sent) for sent in reference_text_2.split('. ') if sent]
        ]

    # If the list contains elements of other data types 
    # you need to convert them to strings before joining
    # use map(str, <list_name>)
#refs_string = ",".join(map(str, references))

with open(output_filepath, "w", encoding="utf-8") as f:
     f.write(references)
#     f.write(refs_string)

#print(references)

# %%
'''
BLEU Score
'''

import os
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize

    # Setup for output
OUTPUT_FILE = "tokenized_refs.txt"

doc_to_print = OUTPUT_FILE;
doc_dir = "/home/bmarron/Desktop" ;
output_filepath = os.path.join(doc_dir, doc_to_print)

    # Define reference texts 
 
reference_text_1 = "I think therefore I am."
#reference_text_2 = "Another reference text that is also long and provides a different perspective on the same topic. This helps in capturing more nuances."

candidate_text = "I think therefore I am."
   # split sentences and then tokenize to words
#references = [
#        [word_tokenize(sent) for sent in reference_text_1.split('. ') if sent],
#        [word_tokenize(sent) for sent in reference_text_2.split('. ') if sent]
#    ]

references = [word_tokenize(sent) for sent in reference_text_1.split('. ') if sent]
candidate = [word_tokenize(sent) for sent in candidate_text.split('. ') if sent]

ref_sents = sent_tokenize(reference_text_1)
ref_sents_strg = "".join(ref_sents)
ref_words = word_tokenize(ref_sents_strg)


cand_sents = sent_tokenize(candidate_text)
cand_sents_strg = "".join(cand_sents)
cand_words = word_tokenize(cand_sents_strg)


weights=(.50, 0.50, 0, 0)

bleu_score = sentence_bleu(ref_words, cand_words, weights)
#bleu_score = corpus_bleu(references, candidate, weights)
#bleu_score = corpus_bleu(ref_words, cand_words, weights)
 
print(bleu_score)


# %%

'''
METEOR Score
'''
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.meteor_score import single_meteor_score 

# Define candidate and reference sentences 
reference_text= "The quick brown fox jumps over the lazy dog. And finds himself caught." 
candidate_text = "A fast brown fox leaps over a lazy dog. And is captured."


ref = word_tokenize(reference_text)
hypo = word_tokenize(candidate_text)


# Calculate METEOR score 
score = single_meteor_score(ref, hypo) 

# Print the result 
print(f"METEOR Score: {score:.3f}")


# %%


