# 의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크

[2022 대한전자공학회 추계학술대회](https://conf.theieie.org/2022f/) 발표<br/>

## Paper
[의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11195634&googleIPSandBox=false&mark=0&ipRange=false&b2cLoginYN=false&isPDFSizeAllowed=true&accessgl=Y&language=ko_KR&hasTopBanner=true)<br/>
[의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크(Github 첨부파일)](./Materials/paper.pdf)

## Dataset
Web scrapping from Musinsa web-site. (https://www.musinsa.com/app/)

## About paper
의류 이미지 및 여러 관련 정보를 활용하여 의류 수요 정보를 예측하는 인공지능 모델을 구현 및 학습했습니다.

## Stack
```
Python  
CNN(Convolutional Neural Network)
BERT(Bidirectional Encoder Representations from Transformer)
Multi-modal learning
Image classification
Pytorch
Web scrapping
```

## Running the Code
To train the deep neural network for predicting clothing demand.
```
main.py
```
To train the uni-modal(only image) settings.
```
main_noword_ablation.py
```
To perform web scrapping.
```
web_scrapping_python.py
```

## Poster

<img src = "./Materials/poster.png" width="100%"> 

## Citation
```
@article{김동주2022의류,
  title={의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크},
  author={김동주 and 이민식},
  journal={대한전자공학회 학술대회},
  pages={788--791},
  year={2022}
}
```
```
김동주, and 이민식. "의류 수요 정보 예측을 위한 멀티모달 기반 딥 뉴럴 네트워크." 대한전자공학회 학술대회 (2022): 788-791.
```
