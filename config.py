#-*- coding:utf-8 -*-
# author: Racle
# project: autosummarization


class CONFIG:
    stopWordsPath = r'data/stopwords.txt'
    wordVecPath = r'data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    # wordVecPath = r'data/samll_w2v'

    ldaModelPath = r'data/lda_model.bin'
    ldaDictPath = r'data/lda_dictionary.bin'
    wordFreqPath = r'data/frequence.bin'

    # hanlp
    allowPosTags = set('nz ni ntc j ntcb nt nhm nic nn t g n nnd ntch nit '
                       'gb gbc nb nnt nba nr an gc nbc nr1 gg nbp nr2 gi nf '
                       'nrf gm ng nrj gp nh ns nhd nsf i v vl vi vd nl'.split())
    # jieba
    # allow_speech_tags =  ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg',
    #                       'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

    numTopics = 10  # 根据lda训练时的数据调整，与训练模型保持一致

    embedModelParaA = 0.0001

    debug = False