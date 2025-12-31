#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:20:22 2025
Updated: 02 Oct 2025
@author: bmarron / Gemini 2.5 Flash /

"""


# %%

### Translation of mime-type = "text/plain" files ####

'''
This script will:
1.  Read an English (or any other language) text file.
2.  Construct a clear translator prompt (sys and user) for the 
    Gemini 2.5 Flash model.
3.  Send the config to the Gemini 2.5 Flash model.
4.  Write the AI translation output to a new file.

Document MIME types available for Gemini output:
    text/plain   ==> .txt
    text/html    ==> .html
    text/json    ==> ,json
    text/x-tex   ==> .tex

'''


# %%

'''
Simple Test Run

'''

from google import genai
from google.genai import types
import os


#import google.generativeai as genai
#from google.generativeai import types
#import os


#--- API Key ---------------------
    # API_KEY is saved as an ENV VARIABLE on home compu
    # (See "Info_Gemini_API.txt for procedure)
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)



#--- The API call to the AI
    # ==> modify "prompt" (user level) as needed
    # ==> modify "system instruction" (behavior) as needed

prompt = "Write a one-sentence bedtime story about a unicorn."

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    config = types.GenerateContentConfig(
               system_instruction="Talk like a pirate."),
    contents = [prompt]
    )

#--- Output from AI (response.text) ---

   # Send to IPython window
print(response.text)

'''
Ahoy, me little matey, close yer peepers and dream of a grand unicorn \
whose horn shines brighter than any buried treasure, guidin' ye through \
the starlit seas of slumber 'til mornin's light.

'''


# %%

'''
Translate content of .txt files
    * returns UTF-8 output
    * output may be requested as any accepted MIME type file

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
#client = genai.Client(api_key=ACTUAL_API_KEY)




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
doc_dir = "/home/bmarron/Desktop"


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

processedfile = types.Part.from_bytes(
        data=input_f.read_bytes(),
        mime_type='text/plain'
        )



#--- The API call to the AI
    # ==> modify "prompt" (user level) as needed
    # ==> modify "system instruction" (behavior)as needed
    

prompt = "Translate the following English text into standard, natural, \
        and fluent Spanish. Maintain all specific formatting (line breaks, \
        indents, spaces, paragraphs).Do not translate Chinese characters"


response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are an expert linguist \
            specializing in translation. Maintain the original \
            meaning and tone. Provide ONLY the requested translation without \
            any additional commentary, introductory phrases, other language \
            translations or conversational remarks."),

    # Select upload or process in-place         
#    contents=[uploadedfile, prompt]
    contents=[processedfile, prompt]
    )



#--- Output from AI (response.text) ---

   # Send to IPython window
#print(response.text)



    # Send to OUTPUT_FILE
GeminiOutput = response.text

    # Options for encoding: utf-8, Latin-1
with open(output_f, "w", encoding="utf-8") as f:
     f.write(GeminiOutput)
     
print(f"Translation complete! Translated text saved to '{output_f}'.")




# %%
