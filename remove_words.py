import sys
import nltk
from nltk.corpus import stopwords
from utils import clean_str, clean_str_sst, clean_str_naver
from parse import parse_args
from konlpy.tag import Mecab

args_config = parse_args()  

def remove_words_naver():

    mecab = Mecab()       
    
    stopwords = []
    doc_content_list = []
    doc_list = [] 
    n_train = 0
    n_test = 0

    with open('data/korean_stopword_list.txt','rb') as f:
        for line in f.readlines():         
            stopwords.append(line.strip().decode('utf-8').split('\t')[0])
   
    with open('data/corpus/naver_train.txt','rb') as f:
        for line in f.readlines():         
            doc_content_list.append(line.strip().decode('utf-8').split('\t')[1])
            doc_list.append(line.strip().decode('utf-8').split('\t'))
            n_train += 1
    with open('data/corpus/naver_test.txt','rb') as f:
        for line in f.readlines():         
            doc_content_list.append(line.strip().decode('utf-8').split('\t')[1])
            doc_list.append(line.strip().decode('utf-8').split('\t'))
            n_test += 1

    word_freq = {} # remove rare words
    index = 0    
    print('doc list len: ', len(doc_list))
    
    for doc_content  in doc_content_list:        
        tmp = func(doc_content)       
        tokenized = mecab.morphs(tmp)
        for word in tokenized:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        tmp = func(doc_content)   
        tokenized = mecab.morphs(tmp)
        stopword_removed_tokenized = [word for word in tokenized if not word in stopwords]       
        doc_words = []
        for word in stopword_removed_tokenized:
            if word_freq[word] >= least_freq:
                doc_words.append(word)
        if len(doc_words) > 0:
            if index < n_train:   
                f = open('data/corpus/naver.clean.train.txt', 'a')             
                f.write('\t'.join(doc_list[index])+'\n')
                f.close()
            else:
                f = open('data/corpus/naver.clean.test.txt', 'a')             
                f.write('\t'.join(doc_list[index])+'\n')
                f.close()        
            doc_str = ' '.join(doc_words).strip()
            clean_docs.append(doc_str)
        index += 1
        print(index)
               
    clean_corpus_str = '\n'.join(clean_docs)
    with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
        f.write(clean_corpus_str)
    
    len_list = []
    with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            temp = line.strip().split()
            len_list.append(len(temp))

    print('min_len : ' + str(min(len_list)))
    print('max_len : ' + str(max(len_list)))
    print('average_len : ' + str(sum(len_list)/len(len_list)))
    print('word count: ' + str(len(word_freq)))
    '''
    min_len : 1
    max_len : 78
    average_len : 12.708584801706447
    word count: 56574
    '''


def remove_words_orig():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    doc_content_list = []
    with open('data/corpus/' + dataset + '.txt', 'rb') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip().decode('latin1'))

    word_freq = {}  # to remove rare words

    for doc_content in doc_content_list:
        temp = func(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        temp = func(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            if dataset == 'mr' or 'SST' in dataset:
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= least_freq:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)


    clean_corpus_str = '\n'.join(clean_docs)
    with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
        f.write(clean_corpus_str)


    len_list = []
    with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            temp = line.strip().split()
            len_list.append(len(temp))

    print('min_len : ' + str(min(len_list)))
    print('max_len : ' + str(max(len_list)))
    print('average_len : ' + str(sum(len_list)/len(len_list)))


dataset = args_config.dataset

if 'SST' in dataset:
    func = clean_str_sst
    least_freq = 5
    remove_words_orig()

elif 'naver' in dataset:
    func = clean_str_naver
    least_freq = 3
    remove_words_naver()

else:
    func = clean_str
    least_freq = 5
    remove_words_orig()