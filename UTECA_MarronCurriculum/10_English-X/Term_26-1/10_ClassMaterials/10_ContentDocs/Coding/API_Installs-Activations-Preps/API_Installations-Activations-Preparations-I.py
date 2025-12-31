#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12 Oct 2025
Revised: 17 Oct 2025

@author: bmarron
"""


# %%

=====   [Installations / Activations / Preparations] ===================


# %%

'''
** Install Spyder ****
'''

	- Go to https://www.spyder-ide.org/
	- Spyder will automatically detect your OS and suggest a download
	- Let the installer decide where to install Spyder OR define a simple location.


# %%

'''
** Verify the Spyder installation ****
'''
	- Open a (regular) terminal and go to your home directory
		> cd C:\Users\<your_user_name>\			<== Windows
		
		$ cd ~								<== Linux/MacOS
		
	
	- Check / find the Spyder installation directory (folder)
		> dir /b/s *spyder*						<== Windows
		> where spyder*
		
		$ find ./ -name "*spyder*" -type d			<== Linux/Mac OS
    
# %%
'''
** Install Python 3.13 ****
'''
	- Follow the instruction in this website (Windows and MacOS)
		https://www.python.org/downloads/
		
        
        Python 3.13.7 - Aug. 14, 2025
            Download Windows installer (64-bit)			<== Windows
			Download macOS 64-bit universal2 installer	<== MacOS
			
        
        
		Stable Releases
		----------------
		sudo apt update &&								<== Linux
		sudo apt upgrade $$
		sudo aptitude install python3 python3-pip python3-dev python3-venv build-essential		
		
		
		

# %%

'''
** Verify "pip" installed ("pip" is a Python package manager) ****
'''
	- pip should already be installed with Python 3.13.7
	- Open a terminal and go to the top of the filesystem
		> cd\
		> pip --version		<== Windows
		
		$ pip --version		<== Linux/MacOS
			(eg, pip 25.2 from /home/bmarron18/.local/lib/python3.10/site-packages/pip (python 
			3.10)
					
	- If needed, follow the instructions in this website
		https://phoenixnap.com/kb/install-pip-windows

# %%
'''
    Upgrade to latest pip
'''	

	
		> python -m ensurepip --upgrade				<== Windows


		$ python3 -m pip install --upgrade pip			<== Linux/MacOS

# %%
'''
** Verify "virtualenv" package installed  ****
'''

	- search (and pipe the output to a file on the Desktop)
>C:\> dir /b/s virtualenv.exe									<== Windows
>C:\Users\<you>\...\python313\site-packages\virtualenv\virtualenv.exe


$ sudo find ./ -name "*virtualenv*" -type d >> ~/Desktop/test.txt	<== Linux/MacOS
./.local/lib/python3.10/site-packages/virtualenv
./.local/lib/python3.10/site-packages/virtualenv-20.34.0.dist-info

# %%
'''
################################################
Create
	(ai-apis)			<== a separate, virtual environment in python for 
                        handling the AI APIs

Install into (ai-apis)
	spyder-kernels 	<== tools to allow Spyder access to the virtual environment
	google-genai 	<== most recent SDK for Google Gemini models
	openai			<== most recent SDK for OpenAI models
	jiter			<== A standalone version of the JSON parser
	nltk 			<== Natural Language Toolkit
###################################################################
'''

# %%

'''
** Specify the use of python to create (ai-apis), a virtual environment) ****
	The virtual environment is created with the Python package, "virtualenv" 
    using the command "venv"
'''

								<== in Windows use python 3.13 in (ai-apis)
								<== NB. in (ai-apis) will see all python distros available on compu 
> cd \...\AppData\Local\spyder-6\...\env\
> python -m venv ai-apis



								<== in Linux use python 3.11-venv to create (ai-apis)
								<== in MacOS use python 3.13 to create (ai-apis)
								<== in (ai-apis) are all python distros available on compu 
									(eg python 3.10 and 3.11.orc1)
$ python3.11 -m venv ./spyder-6/envs/ai-apis


# %%

'''
*** Activate the virtual environment, (ai-apis)
*** Install packages into (ai-apis)
    (special packages, Python packages and SDKs) ****
*** Deactivate the virtual environment
'''
	Step 1. Activate the (ai-apis) virtual environment
	  NB. When activated, you will see (ai-apis) at the command line

>C:\Users\<you>\...\spyder-6\...\envs\ai-apis\Scripts> activate		<== Windows
(ai-apis)>


$ source ./spyder-6/envs/ai-apis/bin/activate				<== Linux/MacOS
(ai-apis):~$



	Step 2. Install packages
(ai-apis)> pip install spyder-kernels==3.1.* google-genai openai nltk jiter	<== Windows


(ai-apis):~$ pip install spyder-kernels==3.1.* google-genai openai nltk jiter		<== Linux/MacOS




	Stwp 3. close the (ai-apis) virtual environment
(ai-apis)> deactivate 												<== Windows
>


(ai-apis):~$ deactivate 												<== Linux/MacOS
$

    Step 4. Close Spyder and re-open this file 
    (API_Installations-Activations-Preparations-I.py)


# %%

'''
Set Spyder to run in (ai-apis)

NB change the .exe file from "ipython.exe" to "python.exe"
in >C:\Users\<you>\...\ai-apis\Scripts

'''
Follow the instructions given here:
https://docs.spyder-ide.org/5/faq.html#using-existing-environment
	Q. How do I get SPyder to work with my existing pkgs/env?



*** Additional info / instructions ****
** open Spyder and clic on the "wrench" in the toolbar to open the Preferences window ****
	- in left-hand column clic on "Python Interpreter"
	- in the right-hand window clic on the button "Use the following interpreter:"
		-- in the empty box search your compu directory until you find the (ai-apis) directory
			(eg .../spyder-6/envs/ai-apis)
        -- open this directory and find the /bin (\Scripts) directory <== Windows
		-- open this directory and find the /bin (\bin) directory <== Linux
		-- open the "Scripts" OR "bin" directory and select
			--- find the "ipython3" icon  	<== Windows
            --- once the path is listed in the "Selected Interpreter" field,
            change "ipython3" to just "python3"  <== Windows
            --- find "python3.11" icon	<== Linux
		-- at the bottom of the window clic 'Apply' and then 'OK
		-- close the "Preferences" window

	- back in the Spyder main windows, find the IPython interpreter window (bottom right)
		-- find the tab that says, "Console 1/a"
		-- close the tab by clicking on the "X"
		-- Spyder will try to re-start Python using
			"Python 3.13.7"	<== Windows/MacOS
			"Python 3.11.0rc1" 	<== Linux

** Verify Spyder is operating in (ai-apis) ****
	- if all good, you will see a green-colored "In [1]" in the IPython interpreter window (bottom right)
	- if an Error Box appears in the interpreter window something went wrong (try closing and re-opening Spyder)


# %%

'''
** You are now ready to access the NLTK and the APIs! ****
'''

NLTK
    Metrics_BLEU_NLTK-pkg.py
    Metrics_METEOR_NLTK-pkg.py

APIs
	-open any of the following in Spyder
		Gemini_APITranslation_pdf-files.py
		Gemini_APITranslation_text-files.py
		OpenAI_APITranslation_pdf-files.py
		
	- modify the prompts and run the blocks/cells of interest

