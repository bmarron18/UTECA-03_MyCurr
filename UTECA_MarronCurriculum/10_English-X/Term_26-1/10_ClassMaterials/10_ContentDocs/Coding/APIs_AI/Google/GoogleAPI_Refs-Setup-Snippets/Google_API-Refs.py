#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:54:30 2024

@author: bmarron
"""




# %%

'''
References
'''

 # Google DeepL
https://transcy.crisp.help/en/article/how-to-translate-target-language-by-deepl-service-1ftwdt4/?bust=1712201846082


  #Google Cloud Translation API
https://cloud.google.com/translate/?hl=en


  # Tutorial--Getting Started with Google Cloud Translation API
  # [This tutorial confuses local and cloud shell machines]
https://codelabs.developers.google.com/codelabs/cloud-translation-python3#0

    # google-cloud-translate documentation
https://pypi.org/project/google-cloud-translate/
https://cloud.google.com/python/docs/reference/translate/latest/summary_overview
https://cloud.google.com/python/docs/reference/translate/latest/summary_method.html

    #This one!!
Class TranslationServiceClient (3.20.1) 
https://cloud.google.com/python/docs/reference/translate/latest/google.cloud.translate_v3.services.translation_service.TranslationServiceClient#google_cloud_translate_v3_services_translation_service_TranslationServiceClient_TranslationServiceClient
https://cloud.google.com/python/docs/reference/translate/latest/google.cloud.translate_v3.services.translation_service.TranslationServiceClient#google_cloud_translate_v3_services_translation_service_TranslationServiceClient
https://cloud.google.com/python/docs/reference/translate/latest/summary_method#google_cloud_translate_v3_services_translation_service_TranslationServiceClient_summary

# %%

'''
 References
'''
  # Google Cloud Services Account
https://cloud.google.com/
https://console.cloud.google.com/
https://cloud.google.com/iam/docs/service-accounts-create
https://cloud.google.com/iam/docs/service-account-overview
https://cloud.google.com/resource-manager/docs/creating-managing-projects#before_you_begin


  # Installing Google Cloud SDK (includes CLI)
https://cloud.google.com/sdk
https://cloud.google.com/sdk/docs/initializing
https://stackoverflow.com/questions/71086225/how-to-install-or-uninstall-gcloud


  # Google Cloud Shell (available from Google Cloud Console)
  ''' IDE with a command line interface. This virtual machine is loaded with all the 
  development tools needed. It offers a persistent 5 GB home directory and runs in Google Cloud.
  Cloud Shell is a Debian-based virtual machine '''


  # Google Authentication and Application Default Credentials (ADC)
https://cloud.google.com/docs/authentication
https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
https://cloud.google.com/docs/authentication/application-default-credentials#personal
https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
https://cloud.google.com/docs/authentication/provide-credentials-adc
https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
https://cloud.google.com/docs/authentication/application-default-credentials
https://stackoverflow.com/questions/51554341/google-auth-exceptions-defaultcredentialserror
https://stackoverflow.com/questions/40032678/where-are-google-application-default-credentials-stored
https://medium.com/datamindedbe/application-default-credentials-477879e31cb5
https://www.googlecloudcommunity.com/gc/Serverless/Why-i-have-the-error-raise-exceptions-DefaultCredentialsError-in/m-p/726413
https://stackoverflow.com/questions/73451173/google-auth-exceptions-defaultcredentialserror-could-not-automatically-determin
https://stackoverflow.com/questions/74502095/setting-application-default-credentials-adc-on-google-cloud


  # Google Cloud Storage
https://cloud.google.com/storage/docs/gsutil/commands/config] 


  # Google Cloud Client Libraries -- General
https://cloud.google.com/sdk/docs/install
https://cloud.google.com/apis/docs/cloud-client-libraries
https://cloud.google.com/apis/docs/client-libraries-explained
https://developers.google.com/api-client-library/
https://developers.google.com/apis-explorer/
 

  # Google Cloud Client Libraries -- Python
https://cloud.google.com/python
 https://github.com/googleapis/google-cloud-python
https://cloud.google.com/python/docs/getting-started
https://cloud.google.com/python/docs/setup
https://cloud.google.com/python/docs/reference


  # Google Cloud Client Libraries -- Translation / General
https://cloud.google.com/translate/docs
https://cloud.google.com/translate/docs/setup
https://cloud.google.com/translate/docs/reference/rest
https://cloud.google.com/translate/docs/advanced/translate-documents
https://cloud.google.com/translate/docs/supported-formats
https://pypi.org/project/google-cloud-translate/


  # Google Cloud Client Libraries  -- Translation / Python
https://cloud.google.com/python/docs/reference/translate/latest
https://cloud.google.com/translate/docs/advanced/translating-text-v3#translate_v3_translate_text-python
https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/translate/samples/snippets


  # Google Advanced Translation
https://cloud.google.com/translate/docs/intro-to-v3


  # Google Zones and Location of Servers for Cloud Computing
https://cloud.google.com/compute/docs/regions-zones
https://stackoverflow.com/questions/66841594/google-cloud-translate-400-invalid-resource-name-location-even-though-it-is-vali



    #On Set the PROJECT_ID environment variable (to be used in your application)
https://stackoverflow.com/questions/35599414/get-the-default-gcp-project-id-with-a-cloud-sdk-cli-one-liner



---- $PYTHONPATH
https://askubuntu.com/questions/384996/what-does-my-pythonpath-contain


---- Option 3: pip install spyder-kernels==3.0.* in virtual environment
https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment


---- trying to install google-cloud-translate in Spyder Python
https://stackoverflow.com/questions/59200541/why-does-my-google-translate-api-work-in-the-terminal-but-not-an-executable
https://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
https://stackoverflow.com/questions/44389630/using-spyder-with-virtualenv



 # translaring text
https://cloud.google.com/translate/docs/basic/translating-text#translating_text
https://cloud.google.com/translate/docs/basic/translating-text#translate_translate_text-python



