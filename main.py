# =========================================================================================================================
#   File Name           :   main.py
# -------------------------------------------------------------------------------------------------------------------------
#   Purpose             :   Purpose of this script is to generate topic names and provide the output text 
#   Author              :   Abhisek Kumar
#   Co-Author           :   
#   Creation Date       :   07-January-2021
#   History             :
# -------------------------------------------------------------------------------------------------------------------------
#   Date            | Author                        | Co-Author                                          | Remark
#   07-Jan-2021    | Abhisek Kumar                                         | Initial Release
# =========================================================================================================================
# =========================================================================================================================
# Import required Module

import config
import topicmodellinglda
import daskPrep
import pandas as pd
import logging
import warnings
import traceback
warnings.filterwarnings('ignore' ) 



def top_asg(pData, pColAsg, pCnt='Count'):
    try:
        pAsgData, pAsgGroup = pd.DataFrame(),[]
        pCountAsg = pData.groupby(pColAsg)[pCnt].sum()
        pAsgGroup = pCountAsg.index.values           
        pAsgData = pData.loc[pData[pColAsg].isin(pAsgGroup)]    
    except Exception as e:
        raise(e)
    return pAsgData, pAsgGroup 


def get_topics(pData, pDesc, pNumWords, pColAsg, pCnt='Count'):
    try:
        pData[pDesc].fillna("unknown", inplace=True)
        pData[pCnt] = len(pData)*[1]        
        pTopicDf = pd.DataFrame()
        _, pAsgGroup = top_asg(pData, pColAsg, pCnt)
        _, pTopicPreProcessDf = daskPrep.preprocess(pData, pDesc)
        for index in range(len(pAsgGroup)):
            pAsgData = pTopicPreProcessDf.loc[pTopicPreProcessDf[pColAsg] == pAsgGroup[index]].reset_index(drop=True)  
            _, pTopicModelDf = topicmodellinglda.topicmodel(pAsgData, pNumWords)
            pTopicDf = pTopicDf.append(pTopicModelDf)

    except Exception as e:
        print(traceback.format_exc())
    return pTopicDf



if __name__ == "__main__":
    
    resultData = get_topics(pd.read_excel(config.pRootDir),config.pDesc,config.pNumWords,config.pColAsg)
    resultData.to_excel(config.pOutputDir + 'output.xlsx', index=False)
    print('Successful...!!')