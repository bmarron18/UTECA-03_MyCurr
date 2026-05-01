#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 12 11:20:22 2025
@author: bmarron / Gemini 2.5 Flash


Google API Studio
https://aistudio.google.com/welcome


Gemini API: Authentication Quickstart
https://ai.google.dev/gemini-api/docs/quickstart?lang=python
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb


Gemini API libraries
https://ai.google.dev/gemini-api/docs/libraries


Gemini Developer API
https://ai.google.dev/gemini-api/docs


This file provides curated prompts to help generative AI models like 
Gemini and Claude produce code using the latest Gemini APIs.
https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md

"""
# %%

# in Linux terminal (available to internal Spyder Python)
    $ pip install google-genai
  
# in (spyder_env) thru Anaconda (available to Spyder in (spyder_env) Python)
    $ conda activate spyder_env
    (spyder_env) $ conda install google-genai
    (spyder_env) $ conda deactivate
    

# %%

'''
A simple first calls to Gemini 2.5 Flash
Basic Inference (Text Generation)

'''


from google import genai

    # Set the API Key
     
API_KEY = "MY_KEY_ALPHANUMERIC"
client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Explain how AI works in a few words")
print(response.text)


# %%

from google import genai

    # Set the API Key
 
    
API_KEY = "MY_KEY_ALPHANUMERIC"
client = genai.Client(api_key=API_KEY)


response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Can Gemini 2.5 Flash be used for natural language \
    translation?")
print(response.text)


# %%


'''
Yes, **Gemini 2.5 Flash can absolutely be used for natural 
language translation.**

Here's why and what its strengths and considerations are for 
this task:

1.  **It's a Large Language Model (LLM):** Translation is a 
fundamental capability of modern LLMs. They are trained on vast 
amounts of text data, including multilingual corpora, allowing 
them to understand the nuances of different languages and generate
accurate translations.

2.  **Part of the Gemini Family:** Gemini Flash is a member of 
the Gemini model family, designed for speed and efficiency. While
it's optimized for lower latency and cost compared to Gemini Pro 
or Ultra, it still retains strong language understanding and 
generation capabilities.

3.  **Strengths for Translation:**
**Speed:** 
    As its name suggests, Flash is designed for speed. This makes
    it excellent for real-time translation applications, chat 
    interfaces, or scenarios requiring high-throughput translation

**Cost-Effectiveness:
    Being a lighter model, it generally incurs lower costs per 
    token/request, making it economical for large-scale 
    translation needs.

**Contextual Understanding:
    Unlike older rule-based or phrase-based translation systems, 
    LLMs like Flash can leverage broader context within a sentence
    or even across multiple sentences to provide more natural and
    accurate translations.
    
**Flexibility:
    Beyond just literal translation, you could potentially prompt
    it for things like:
        * Translating while maintaining a specific tone 
        (e.g., formal to informal).
        * Summarizing a text and then translating the summary.
        * Translating specific entities or terms consistently.

**Considerations:
    * Accuracy vs. Nuance (compared to larger models):
        While very capable, for extremely sensitive, highly 
        idiomatic, or specialized domain texts (e.g., legal, 
        medical), Gemini Ultra or Pro *might* offer slightly 
        more nuanced or precise translations due to their larger 
        parameter count and deeper understanding. However, for 
        most general-purpose translation, Flash will perform very
        well.
    * Dedicated Translation Services:
        For pure, high-volume, general-purpose translation without
        any other LLM tasks, dedicated translation APIs (like 
        Google Cloud Translation API, which uses specialized 
        models optimized solely for translation) might sometimes 
        be even more cost-effective or offer specific features 
        like glossary support, though an LLM offers much more 
        flexibility.

**In summary:** 
Gemini 2.5 Flash is an excellent choice for natural 
language translation, especially when speed, cost-efficiency, and good 
general-purpose accuracy are key. You would typically interact 
with it via an API (e.g., through Google Cloud's Vertex AI 
platform) by sending a prompt that requests the translation.

'''

# %%

from google import genai


    # Set the API Key
  
    
API_KEY = "MY_KEY_ALPHANUMERIC"
client = genai.Client(api_key=API_KEY)


response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Give me a sample Python prompt for translating a \
        text file from Englsh to Spanish using Gemini 2.5 Flash \
        for natural language translation")

print(response.text)


# %%

'''
Python Code Prompt: Spanish to English Translator

'''

from google import genai


# Set the API Key
    # Get my Gemini API Key here:
    # /home/bmarron/Desktop/UTECA/GithubToken_UTECALogins.txt
    
API_KEY = "MY_KEY_ALPHANUMERIC"
client = genai.Client(api_key=API_KEY)


response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Write a Python code prompt to translate from \
        Spanish into English")
print(response.text)


# %%

'''
Here's a Python code prompt that allows a user to input Spanish 
text and get an English translation using the `googletrans` 
library.

**Note:** The `googletrans` library is an unofficial Google 
Translate API client. While generally reliable for casual use, 
it can sometimes be rate-limited or break due to changes in 
Google's translation service. For production environments or 
mission-critical applications, consider using official, paid APIs
like Google Cloud Translation, DeepL API, or Amazon Translate.
'''

# --- Python Code Prompt: Spanish to English Translator ---

# Prerequisites:
# You need to install the 'googletrans' library.
# Open your terminal or command prompt and run:
# pip install googletrans==4.0.0-rc1
# (Note: Use the specified version as newer versions might have 
breaking changes or require specific setups)



### How to use this prompt:
'''
1.  **Install the Library:**
    Open your terminal or command prompt and run:
    ```bash
    pip install googletrans==4.0.0-rc1
    ```
    (It's important to use the `4.0.0-rc1` version as newer ones might 
     have compatibility issues with this simple usage pattern).

2.  **Save the Code:**
    Save the code above as a Python file (e.g., `translator_prompt.py`).

3.  **Run the Script:**
    Open your terminal or command prompt, navigate to the directory where you saved the file, and run:
    ```bash
    python translator_prompt.py
    ```

4.  **Interact:**
    The script will then prompt you to enter Spanish text. Type your Spanish sentence and press Enter. The English translation will be displayed. You can continue entering text until you type `quit`.
'''


# %%

from googletrans import Translator

def translate_spanish_to_english():
    """
    Prompts the user for Spanish text and translates it to English.
    The translation continues until the user types 'quit'.
    """
    translator = Translator()

    print("--- Spanish to English Translator ---")
    print("Enter Spanish text to translate, or type 'quit' to exit.")
    print("-" * 35)

    while True:
        spanish_text = input("\nSpanish Text: ")

        if spanish_text.lower() == 'quit':
            print("Exiting translator. ¡Hasta luego!")
            break
        elif not spanish_text.strip():
            print("Please enter some text to translate.")
            continue

        try:
            # Translate from Spanish ('es') to English ('en')
            # The 'src' and 'dest' parameters ensure correct language detection,
            # though the library often auto-detects well.
            translation = translator.translate(spanish_text, src='es', dest='en')

            print(f"English Translation: {translation.text}")
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            print("Please check your internet connection or try again later.")
            print("If the issue persists, the unofficial 'googletrans' library might be experiencing temporary issues.")

# --- Run the translator prompt (in terminal as .py script ---
if __name__ == "__main__":
    translate_spanish_to_english()


# %%


### Example Interaction:

```
--- Spanish to English Translator ---
Enter Spanish text to translate, or type 'quit' to exit.
-----------------------------------

Spanish Text: Hola, ¿cómo estás?
English Translation: Hello, how are you?

Spanish Text: El sol brilla hoy.
English Translation: The sun shines today.

Spanish Text: Me gusta aprender Python.
English Translation: I like to learn Python.

Spanish Text: quit
Exiting translator. ¡Hasta luego!
```
# %%



