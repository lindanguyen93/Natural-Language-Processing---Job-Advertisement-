#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Linda Nguyen
# #### Student ID: s3651761
# 
# Date: 2/10/2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# 
# ## Introduction
# 
# In task 2 of this assignment, we'll perform text pre-processing on a job advertisement dataset. We'll focus on pre-processing the Description only. We'll tokenize, remove single character token, stopwords, most/less frequent words.
# 
# We'll focus on pre-processing the Description only. We'll perform the following steps:
# 
# 1. Extract information from each job advertisement. Perform the following pre-processing steps to the description of each job advertisement;
# 2. Tokenize each job advertisement description use regularexpression, r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?";
# 3. Convert all word to lower case;
# 4. Remove words with length less than 2.
# 5. Remove stopwords using the provided stop words list use `stopwords_en.txt`.
# 6. Remove the word that appears only once in the document collection, based on term frequency.
# 7. Remove the top 50 most frequent words based on document frequency.
# 8. Save all job advertisement text and information in a txt file;
# 9. Build a vocabulary of the cleaned job advertisement descriptions, save it in a txt file;

# ## Importing libraries 

# In[1]:


# importing libraries
import pandas as pd
import re
import numpy as np
from sklearn.datasets import load_files
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from itertools import chain
from nltk.probability import *
from nltk.util import ngrams


# ### 1.1 Examining and loading data

# Before doing pre-processing, we need to load the data into a proper format. To load the data, we have to explore the data folder. Inside `data`, we have 4 sub-folders which is 4 job categories namely 'Accounting_Finance', 'Engineering', 'Healthcare_Nursing' and 'Sales'. Each sub-folder contains numbers of text files. Each text file contains Title, Webindex, Company(some have no info of Company) and Description. Now, let find out the inforation of the imported datset match the data desscription in this assignment. 

# In[2]:


# load the data files
data = load_files(r'data')


# In[3]:


# check number of attributes of data 
print(len(data))


# In[4]:


# check attributes of data: 
print(data.keys())


# The loaded data has 5 attributes as below:
#     
# - data - a list of text reviews
# - target - the corresponding label of the text reviews (integer index)
# - target_names - the names of target classes.
# - filenames - the filenames holding the dataset.
# - DESCR - description of data

# In[5]:


# display data
data.data


# In[6]:


# display data filenames
data.filenames


# In[7]:


print( 'We have total', len(data['data']), 'job ads.')


# In[8]:


# check number of job category of the data set
len(data.target_names)


# In[9]:


# check name of the 4 job category
data.target_names


# In[10]:


# again, check number of jobs in all 4 job catagories
len(data.target)


# In[11]:


# check label of each target name or their coresponding job category
print(data.target_names[0])
print(data.target_names[1])
print(data.target_names[2])
print(data.target_names[3])


# We can see that the outputs of the above exploration of imported dataset matched the data description in the assisgnment. Thus, we can move on to the next task and start doing data pre-processing.

# In[12]:


# extract all job ads save in job and target in labels
job , lables = data.data, data.target


# ### 1.2 Pre-processing data

# In the following tasks, we'll tackle the following basic text pre-processing for extracted description: 
#     
# - Word Tokenization
# - Case Normalisation
# - Removing Single Character Tokens
# - Removing stopwords
# - Removing words appear once only
# - Removing top 50 most common words
# 
# Then, we'll save the cleaned description in `vocab.txt` and all job ads info in `job_ads.txt`.

# ###  Extract Description, Word Tokenization, Case Normalization for each job advertisement

# In this task, we'll tokenize each of the text description. In particular, we'll perform extract description from `job`,sentence segmentation, normalize description to lower case, tokenize by sentence, used the provided tokenizer pattern. We then tokernize each sentence in description into tokens, put all tokens of description into a list.     
# 
# Be careful, when you try to use the sent_tokenize on the review text files, you may get a TypeError. This is because each job text is read as a byte object, however, the tokenizer cannot apply a string pattern on a bytes-like object. To resolve this, you need to decode each read job text using utf-8, e.g. `raw_job = raw_job.decode('utf-8')`

# In[13]:


def tokenize_job (raw_job): 
    
    raw_job = raw_job.decode('utf-8') # decode job ads 
    raw_job = re.split(r'\n', raw_job) # split job ads by line
    des = [i for i in raw_job if i.startswith('Description:')] # extract description 
    description = re.sub(r'^Description:\s*', '', des[0]).lower() # normalize lower case for description
    sentences = sent_tokenize(description ) # tokenize by sentence
    
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" # create tokenizer pattern
    tokenizer = RegexpTokenizer(pattern)
    
    # tokenize each sentence of description into tokens
    tokenize_each_sentence = [tokenizer.tokenize(sent) for sent in sentences]
    # put all tokens of description into a list
    tokens_job = list(chain.from_iterable(tokenize_each_sentence))  
    return tokens_job


# In[14]:


# tokenize each job ad
tk_job = [tokenize_job (i) for i in job]


# Here, we need to check the size of the vocabulary at this stage, as well as the total number of tokens, etc. in this job dataset. 

# In[15]:


# it takes a tokenize job list and display the number of vocab, words, Lexical diversity, max/min/average review length
def stats_print(tk_job):
    words = list(chain.from_iterable(tk_job)) # we put all the tokens in the corpus in a single list 
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set
    lexical_diversity = len(vocab)/len(words) # compute lexical_diversity
    
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of job:", len(tk_job))
    
    lens = [len(j) for j in tk_job]  # compute number of tokens 
    print("Average job length:", np.mean(lens))
    print("Maximun job length:", np.max(lens))
    print("Minimun job length:", np.min(lens))
    print("Standard deviation of job length:", np.std(lens))


# In[16]:


# print the orginal tokens stats
stats_print(tk_job)


# We'll compare the stats of each stage with this original stats to see how much words be reduced. 

# ### Removing Single Character Token

# In this sub-task, we'll remove any token that only contains a single character (a token that of length 1). Again, we'll use the stats to check number of removed words. 

# In[17]:


st_list= [[w for w in t if len(w) <=1] for t in tk_job ] # create a list contain single character token for each job
list(chain.from_iterable(st_list)) # merge them together in one list


# In[18]:


tk_job = [[ w for w in t if len(w) > 1] for t in tk_job] # filter out the single token 


# In[19]:


# check stats
stats_print(tk_job)


# ### Removing stop words

# In this sub-task, we'll remove the stop words from the tokenized text. You use the provided stopword list.  

# In[20]:


# load the provided stopword file
stopwords_en = []
with open('./stopwords_en.txt') as f:
    stopwords_en = f.read().splitlines()


# In[21]:


print( "There are", len(stopwords_en), "stopwords in the provided list.")


# In[22]:


# filter out the tokens that belong in stopword_en
tk_job = [ [w for w in t if w not in stopwords_en]  for t in tk_job]


# In[23]:


# check stat
stats_print(tk_job)


# ### Removing the word that appears only once in the document collection, based on term frequency.

# Term frequency counts the number of times a word occurs in the whole corpus regardless which document it is in. 
# 
# Frequency distribution based on term frequency tells us how the total number of word tokens are distributed across all the types.
# 
# We use the built-in function `FreqDist` of NLTK to compute this distribution from a set of word tokens. 
# 
# Now, let's move on to the less frequent words. 
# 
# - find out the list of words that appear only once in the entire corpus
# - remove these less frequent words from each tokenized review text
#  
# We first need to find out the set of less frequent words by using the `hapaxes` function applied on the term frequency  dictionary.

# In[24]:


words = list(chain.from_iterable(tk_job)) # get all words from tk_job by put in tk_job list
term_fd = FreqDist(words)  # compute term frequency for each word 
lessFreqWords = set(term_fd.hapaxes()) # extract words that appearces once only


# In[25]:


# remove word in lessFreqWords
tk_job = [[ word for word in t if word not in lessFreqWords] for t in tk_job] 


# In[26]:


# check stats
stats_print(tk_job)


# ###  Removing the top 50 most frequent words based on document frequency.

# Document frequency is slightly different then term frequency as it counts the number of documents a word occurs.<br>
# For instance, if a word appear 4 times in a document, when we count the term frequency, this will be added 4 into the total number of occurrence; however, for document frequency, this will stil be counted as 1 only.
# 
# We use the built-in function `FreqDist` of NLTK to compute this distribution from a set of unique word tokens. 

# In[27]:


set_words = list(chain.from_iterable([set(i) for i in tk_job])) # get the set tokens for each job ad
doc_fd = FreqDist(set_words) # compute document frequency for each unique word/type
doc_fd.most_common(50)  # choose 50 most frequent word
mostFreq =set([ i[0] for i in doc_fd.most_common(50)])
mostFreq


# In[28]:


# remove 50 most frequent words
tk_job = [[word for word in t if word not in mostFreq] for t in tk_job]


# In[29]:


# check stats
stats_print(tk_job)


# Recall: from the begining we have:
# 
# `Vocabulary size:  9834
# Total number of tokens:  186952
# Lexical diversity:  0.052601737344345076
# Total number of job: 776
# Average job length: 240.91752577319588
# Maximun job length: 815
# Minimun job length: 13
# Standard deviation of job length: 124.97750685071483`
# 
# We've shrunk about 47% vocabilary size. 

# ## Saving required outputs

# ### Saving all job advertisement text and information in `job_ads.txt`

# In this sub-task, we'll need to find a way to save all job advertisement information. This will help us to continue process task 2 and task 3. <br>
# First, we'll extract job ID which is from each text file name. But here we only keep the 5 digits of the text file. 

# In[30]:


file_name = data.filenames.tolist() # convert filenames to list and save in file_name

# extract 5 numbers of job_ID from text file name, replace with empty string, save them in job_id
job_id = [re.sub(r'[^\d{5}]', '', i) for i in file_name]


# In[31]:


job_id


# Then, we'll find the corresponding name for each target lable. To do this, we need to create a dictionary for categores, lable each job category and convert them into a list. 

# In[32]:


# create dictionary for Category
target_names_dict = {'Accounting_Finance':0, 'Engineering':1, 'Healthcare_Nursing':2, 'Sales':3}

# label and convert category into list by using tolist()
category = lables.tolist()

# convert lable to corresponding category
for i in range(len(category)):
    for key, value in target_names_dict.items():
        if category[i] == value:
            category[i] = key  


# In[33]:


category


# To extract Title and Webindex, we will define coressponding function. In this function, we will take the raw_job as a string, decode and split them into line, then find line start with Tile: or Webindex by using Regex patterns. Next, the function will extract the infomation after these strings. Finally, we'll add the extract info into a list. 

# In[34]:


# finding Title for each job advertisement
def title (raw_job):   
    raw_job = raw_job.decode('utf-8') # decode ad 
    raw_job = re.split(r'\n', raw_job)   # split ad by new line 
    all_titles = [re.sub(r'^Title:\s*', '', i)            for i in raw_job if i.startswith('Title:')]  # extract the digit number which after Title: use regex pattern
    return all_titles   
all_titles =  list(chain.from_iterable([title (i) for i in job ])) # put them into list all_titles


# In[35]:


all_titles


# In[36]:


# find Webindex for each job advertisement
def webindex (raw_job):     
    raw_job = raw_job.decode('utf-8') # decode the raw info
    raw_job = re.split(r'\n', raw_job)  # split raw info into a list
    webindex = [re.sub(r'^Webindex:\s*', '', i)         for i in raw_job if i.startswith('Webindex:')] # extract 8 digit number after 'Webindex:' use regex pattern
    return webindex
webindex  = list(chain.from_iterable([webindex(i) for i in job ])) # put them into list webindex


# In[37]:


# have a look on list webindex
webindex


# Before we store all job infomation in a txt file. Let's check if each feature contain 776 data infomation

# In[38]:


len(job_id)


# In[39]:


# check if they all equal to 776 
len(job_id) == len(all_titles) == len(category) == len(webindex) == len(tk_job)


# Next, we'll create the job_ads.txt to save job Id, Category, Title, Webindex, and Description of each job advertisement. The info is from the the lists we have above. 
# - ID contains 5 digits from job_id list
# - Category is from category list
# - Title is from all_titles list
# - Webindex: 8 digits from webindex list 
# - Description from the processed tk_job list
# 
# Each job advertisement will be stored as a show_string. We should have total 776 job ads stored in job_ads.txt. We can check it after the text file generated.  

# In[40]:


job_ads = open('job_ads.txt ', 'w')  # create file in write mode

# join all job info 
show_string = '\n'.join(['\n'.join(('ID: ' + job_id[i],                 
                               'Category: ' + category[i] ,
                               'Title: ' + all_titles[i],
                               'Webindex: ' + webindex[i],            
                               'Description: ' + 
                               ' '.join(tk_job[i]))) for i in range(len(job))])
job_ads.write(show_string) # save each job in file
job_ads.close() # close file


# Here, we skipped infomation of `Company` because task 2 and 3 don't need Company to carry out the tasks. 

# ### Build a vocabulary of the cleaned job advertisement descriptions and save in `vocab.txt`

# Now, we complete all the basic pre-process step and we are ready to move to feature generation. Before we start, in this task, we'll construct the final vocabulary. 

# In[41]:


vocab  = sorted(list(set(words))) # sort words in alphabet order
voc = open('vocab.txt', 'w')  # create a file in write mode

# loop vocab, extract each vocab, index, format word:index for each line
show = '\n'.join( ':'.join([vocab[i], str(i)]) for i in range(len(vocab)))   
voc.write(show) # save each job ad in the created textfile 
voc.close() # close file


# ## Summary
# 
# In this task, we've done text pre-processing for job description by: 
# * word tokenization
# * case normalisation
# * removing single character words
# * removing stopwords
# * removing words appear only once by term frequency
# * removing top 50 most common words by doccument frequency
# 
# After the text preprocessing, the vocabulary size have been reduced by about 47%.We can check the details at each stage by using the created stats. <br>
# Then, we'll save the cleaned description in `vocab.txt` and all job ads info in `job_ads.txt`. <br>
# 
# The `vocab.txt` contains the unigram vocabulary, one each line, in the format of word_string:word_integer_index. <br>
# Words in the vocabulary are sorted in alphabetical order, and the index value starts from 0. 
# 
# The `job_ads.txt` contains 776 job advertisment with the infomation of job ID, Category, Title, Webindex and Description. We'll need to use this file to carry out the next tasks. 

# In[ ]:




