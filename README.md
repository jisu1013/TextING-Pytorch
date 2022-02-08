# TextING-Pytorch
pytorch implementation of TextING

## Original code (tensorflow version)
https://github.com/CRIPAC-DIG/TextING

the ACL2020 paper Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks
(https://arxiv.org/abs/2004.13826)

## Requirements
- python 3.6+
- Pytorch 
- Scipy 1.5.1

## Dataset
네이버 영화 리뷰 감성 분류하기 데이터셋 (https://github.com/e9t/nsmc)

## Usage
한국어 pre-trained word embeddings을 먼저 다운 받습니다. 

https://github.com/ratsgo/embedding/releases 에서 word-embeedings.zip을 다운 받아 한국어 위키피디아, 네이버 영화 말뭉치, KorQuAD 데이터셋을 Mecab으로 형태소 분석한 말뭉치를 가지고 Glove로 임베딩한 결과를 활용했습니다.
data/corpus에 glove.naver.txt 파일로 저장되어 있습니다.

graph를 build 하기 위해서 다음 코드를 실행합니다.
``` python
python build_graph_naver.py [WINSIZE] [WEIGHTED_GRAPH]
```
default sliding window size는 3입니다.
remove_words.py 다음과 같이 실행합니다.
```python
python remove_words.py 
```
training과 inference를 시작하기 위해서는
```python
python main.py
```
parser.py에서 hyperparameter 값 변경

## Result
learning rate=0.001, epoch=100, batch size=5000, input dim=100, hidden=64, steps=2,dropout=0.3 으로 실험한 결과 (3회를 수행한 뒤 평균값)
- ACC: 0.815
- Time: 2007.051(s)

## Reference
- pre-trained word embedding : https://github.com/ratsgo/embedding/releases
- model/preprocessing reference : https://github.com/CRIPAC-DIG/TextING
- preprocessing reference : my github (https://github.com/jisu1013/DL_starting_with_PyTorch/blob/main/%EB%84%A4%EC%9D%B4%EB%B2%84_%EC%98%81%ED%99%94_%EB%A6%AC%EB%B7%B0_%EB%B6%84%EB%A5%98.ipynb) - 수강했던 수업 실습 자료입니다.
- stopword list reference : https://bab2min.tistory.com/544#google_vignette
