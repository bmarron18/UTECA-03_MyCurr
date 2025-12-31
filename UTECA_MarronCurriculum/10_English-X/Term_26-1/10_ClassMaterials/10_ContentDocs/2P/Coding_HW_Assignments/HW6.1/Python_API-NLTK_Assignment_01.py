#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:38:36 2025

@author: bmarron18
"""

# %%

======== HW Assignemts: Assignment 1 ======================

# %%

#######################################
Translation with Google Gemini 2.5 Flash
########################################

# %%
'''
Step 1:
Translation excerpt from "Joy Luck Club" with Google Gemini 
'''

from google import genai
from google.genai import types
from pathlib import Path
import os


#--- API Key ---------------------
    # API_KEY is saved as an ENV VARIABLE on home compu
    # (See "Info_Gemini_API.txt for procedure)

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


    # API_KEY also can be inserted directly
#client = genai.Client(api_key="ACTUAL_API_KEY")




# --- Prep Files, and Define Files and File Paths -------
    # INPUT_FILE
        #==> this is the .txt file (T)o (B)e (T)ranslated
        #==> ADD quotation marks to the begiining and end of the INPUT_FILE 
        #==> Move .txt file (T)o (B)e (T)ranslated to the Desktop
 
   # OUTPUT_FILE 
       #==> will hold the (T)ranslated text from the AI
       #==> this file will be generated automatically by Python after \
       # the translation is complete
   
    
    # Name the INPUT_FILE and OUTPUT_FILE
    # Change names of files as needed
        # ==> INPUT_FILE must stay as .txt
        # ==> OUTPUT_FILE can be any of the Document types available for \
        # Gemini output (see header of this script)
    
INPUT_FILE = "Gemini_Doc-English_TBT.txt"  
OUTPUT_FILE = "Gemini_Doc-Spanish_T.txt"


    # Set up the file paths for the INPUT_FILE and the OUTPUT_FILE
    # Set the file path to your Desktop
    # Path() represents file+directory paths in a platform-independent manner.
    
doc_to_translate = INPUT_FILE
doc_to_print = OUTPUT_FILE
doc_dir = "/home/bmarron18/Desktop"   #<== OLD HP


    # create paths to files
    # Retrieve files as PosixPaths

input_filepath = os.path.join(doc_dir, doc_to_translate)
input_f = Path(input_filepath)

output_filepath = os.path.join(doc_dir, doc_to_print)
output_f = Path(output_filepath)



#--- Read and process file to be translated ---
    # File UPLOADED to Google
    # (NOT recommended)
'''
uploadedfile = client.files.upload(
    file=input_f,
    config=dict(mime_type='text/plain')
    )
'''

    # File NOT UPLOADED to Google
    # (Recommended)
processedfile = types.Part.from_bytes(
        data=input_f.read_bytes(),
        mime_type='text/plain'
        )



#--- The API call to the AI
    # ==> modify "prompt" (user level) as needed
    # ==> modify "system instruction" (behavior)as needed
    

prompt = "Translate the following English text into standard, natural, \
        and fluent Spanish. Maintain all specific formatting (line breaks, \
        indents, spaces, paragraphs).Do not translate any Chinese characters"


response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are an expert linguist \
            specializing in translation. Maintain the original \
            meaning and tone. Provide ONLY the requested translation without \
            any additional commentary, introductory phrases, other language \
            translations or conversational remarks."),

    # Select upload or process in-place         
#    contents=[uploadedfile, prompt]    #<== upload file
    contents=[processedfile, prompt]    #<== process in place
    )



#--- Output from AI (response.text) ---
   # Send output to IPython window
#print(response.text)

    # Send output to OUTPUT_FILE
GeminiOutput = response.text

    # Options for encoding: "utf-8" OR "latin-1"
#with open(output_f, "w", encoding="utf-8") as f:
with open(output_f, "w", encoding="latin-1") as f:
     f.write(GeminiOutput)
     
print(f"Translation complete! Translated text saved to '{output_f}'.")

# %%

#######################################
Translation with OpenAI gpt-5
########################################


# %%
'''
Step 2a:
Convert .txt file to .pdf
'''

# Open .txt file in a word processor and save as .pdf
# Save as "OpenAI_Doc-English_TBT.pdf" on Desktop

# %%

'''
Step 2b:
Translation excerpt from "Joy Luck Club" with OpenAI
'''

from openai import OpenAI
from pathlib import Path
import os

    # API_KEY is saved as an ENV VARIABLE on home compu
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

    # API_KEY is inserted directly
#client = OpenAI(api_key="ACTUAL_API_KEY")




# --- Prep Files, and Define Files and File Paths -------
    # INPUT_FILE
        #==> this is the .pdf file (T)o (B)e (T)ranslated
        #==> Move .pdf file (T)o (B)e (T)ranslated to the Desktop
 
    # OUTPUT_FILE 
        #==> will hold the (T)ranslated text from the AI
        #==> this file will be generated automatically by Python after \
        # the translation is complete

      
   # Name the INPUT_FILE and OUTPUT_FILE
   # Change names of files as needed
       # ==> INPUT_FILE must stay as .pdf
       # ==> OUTPUT_FILE can be any of the Document types available for \
       # OpenAI output (see header of this script)
    
INPUT_FILE = "OpenAI_Doc-English_TBT.pdf"
OUTPUT_FILE = "OpenAI_Doc-Spanish_T.txt"
    
    
    # set up the file paths for the INPUT_FILE and the OUTPUT_FILE
    # set the file path to your Desktop
    # Path() represents file+directory paths in a platform-independent manner.
    
doc_to_translate = INPUT_FILE
doc_to_print = OUTPUT_FILE
doc_dir = "/home/bmarron18/Desktop"   #<== Old HP


    # create paths to files
    # Retrieve files as PosixPaths

input_filepath = os.path.join(doc_dir, doc_to_translate)
input_f = Path(input_filepath)

output_filepath = os.path.join(doc_dir, doc_to_print)
output_f = Path(output_filepath)


#--- Read and process file to be translated ---
    # File UPLOADED to OpenAI Vector Store
    # (NOT recommended)


    # File NOT UPLOADED to OpenAI
    # (Recommended)
#file = client.files.create(
#   file=open(input_f, "rb"),
#   purpose="user_data",
#    )


with open(input_f, "rb") as fp:
    file = client.files.create(
        file=(fp),
        purpose='user_data'
    )


	# User level message
user_prompt= "Translate the file (a .pdf file in English) to standard, natural, \
    and fluent Spanish. Maintain all specific formatting (line breaks, \
    indents, spaces, paragraphs, and quotations). Generate the output as \
    MIME type= text/plain. That is, generate the output as a plain text \
    file (.txt)."


	# Developer level message
sys_prompt = "You are an expert linguist \
    specializing in translation. Maintain the original \
    meaning and tone. Provide ONLY the requested translation without \
    any additional commentary, introductory phrases, other language \
    translations or conversational remarks."



	# Response API call
response = client.responses.create(
    model="gpt-5",
#    temperature=1.3,
    instructions= sys_prompt,
    input=[
        {
           "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    "text": user_prompt,
                },
            ]
        }
    ]
)


#--- Output from AI (response.output_text) ---

	# Send to IPython window
#print(response.output_text)


   # Send to OUTPUT_FILE with "utf-8" OR "latin-1"
#with open(output_f, "w", encoding="latin-1") as f:
with open(output_f, "w", encoding="utf-8") as f:
     f.write(response.output_text)
     
print(f"Translation complete! Translated text saved to '{output_f}'.")

# %%

'''
Step 3:
METEOR Score fron NLTK
'''

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


 
   # Define a reference text created by a human translator
   # The quotes define a "string" in Python
reference_text= "Y justo después de que mi padre muriera el año pasado, ella dijo \
    que sabía que esto pasaría. Porque una planta de filodendro que mi padre le \
    había regalado se había marchitado y muerto, a pesar de que la regaba fielmente. \
    Dijo que la planta había dañado sus raíces y que el agua no podía llegarle. \
    El informe de la autopsia que recibió más tarde mostró que mi padre había \
    tenido un noventa por ciento de obstrucción de las arterias antes de morir de \
    un ataque al corazón a la edad de setenta y cuatro años. Mi padre no era chino \
    como mi madre, sino anglo-irlandés-americano, que disfrutaba de sus cinco \
    rebanadas de tocino y tres huevos fritos con la yema entera cada mañana."
    
    # Define the candidate text created by an AI translator
    # The quotes define a "string" in Python
candidate_text = "Y justo después de que mi padre muriera el año pasado, dijo que \
    sabía que esto pasaría. Porque \
    una planta de filodendro que mi padre le había regalado se había marchitado y muerto, a pesar \
    de que la regaba fielmente. Dijo que la planta tenía las raíces dañadas y no le podía llegar el \
    agua. El informe de la autopsia que recibió más tarde mostraba que mi padre había tenido un \
    noventa por ciento de obstrucción en las arterias antes de morir de un ataque al corazón a la \
    edad de setenta y cuatro años. Mi padre no era chino como mi madre, sino anglo-irlandés-estadounidense, \
    que disfrutaba de sus cinco lonchas de beicon y tres huevos fritos con la yema hacia arriba \
    todas las mañanas."


ref = word_tokenize(reference_text, "spanish")
cand = word_tokenize(candidate_text, "spanish")


# Calculate METEOR score 
score = single_meteor_score(ref, cand)

# Print the result 
print(f"METEOR Score: {score:.4f}")


# %%


