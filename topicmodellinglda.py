#Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import sys
import traceback
import pandas as pd
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os
os.environ.update({'MALLET_HOME':r'C:/mallet/mallet-2.0.8/'})

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = "C:/mallet/mallet-2.0.8/bin/mallet.bat" # update this path

# ###########################################################################################################################
# Author        : Tapas Mohanty
# Co-Author     : Jahar Sil and Tapas Mohanty
# Modified      :   
# Reviewer      :                                                                                           
# # Functionality : Find Number of topics using mallet
# ###########################################################################################################################

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def convert_corpus(pData):
    try:
        data_words = list(sent_to_words(pData))

        # Create Dictionary
        id2word = corpora.Dictionary(data_words)

        # Create Corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
    
    except Exception as e:
        raise(e)
        print(traceback.format_exc())
        print('*** ERROR[001]: corpus ***', sys.exc_info()[0],str(e))
        return(-1, corpus)
    return (0, id2word, corpus)

def compute_coherence_values(pDictionary, pCorpus, pStart, pLimit, pStep, pWorkers):
    try:
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        model_list = []
        for NumTopics in range(pStart, pLimit, pStep):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus = pCorpus, num_topics=NumTopics, id2word=pDictionary, workers = pWorkers)
            model_list.append(model)
        
    except Exception as e:
        raise(e)
        print(traceback.format_exc())
        print('*** ERROR[002]: topic_model ***', sys.exc_info()[0],str(e))
        return(-1, model_list)
    return (0, model_list) 
      


def format_topics_sentences(pData, pNumWords, pLdamodel, pCorpus):
    try:
        topic_df = pd.DataFrame()
        # Get dominant topic in each document
        for i, row in enumerate(pLdamodel[pCorpus]):                   
            row = sorted(row, key = lambda x: (x[1]), reverse=True)        
            # Get the Dominant topic, Perc Contribution and Keywords for each doc
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:                                           
                    wp = pLdamodel.show_topic(topic_num)              
                    topic_keywords = "__".join([word for word, prop in wp])
                    topic_keywords = topic_keywords.split("__")[:int(pNumWords)]
                    topic_keywords = "__".join([word for word in topic_keywords])
                    topic_df = topic_df.append(pd.Series(
                                                           [round(int(topic_num),0),
                                                           round(prop_topic,4),
                                                           topic_keywords]),
                                                           ignore_index=True)
                else:
                    break
        topic_df.columns = ['Topic', 'Topic_Confidence_Level','Keywords'] # Create dataframe title
        topic_df['Topic'] = topic_df['Keywords'].astype(str)
        
        pData = pd.concat([pData, topic_df], axis = 1)

    except Exception as e:
        raise(e)
        print(traceback.format_exc())
        print('*** ERROR[003]: topic_model ***', sys.exc_info()[0],str(e))
        return(-1, pData)  
    return(0, pData)    

def topicmodel(pData, pNumWords):
    try:
        # # Convert to list
        data = pData['Sample'].values.tolist()
        # #converting to doc matrix
        _, id2word, corpus = convert_corpus(data)
        # #Finding the topic
        _, model_list = compute_coherence_values(pDictionary = id2word, pCorpus = corpus, pStart = 2, pLimit = 40, pStep = 6, pWorkers = 6)
        # # Select the model 
        _, pData = format_topics_sentences(pData, pNumWords, pLdamodel = model_list[3], pCorpus = corpus)
    
    except Exception as e:
        raise(e)
        print(traceback.format_exc())
        print('*** ERROR[004]: topic_model ***', sys.exc_info()[0],str(e))
        return(-1, pData)
    return(0, pData)
