# 이종 그래프 융합 목적 지향 대화 시스템

**목적 지향 대화 시스템**(Task-Oriented Dialogue System)이란 특정 업무를 달성하기 위한 목적을 가지고 사용자에게 도움을 주기 위한 대화 시스템

- 해당 대화 시스템은 **음악 추천**을 목적으로 하는 대화 시스템으로 사용자의 개인정보, 상황, 발화 상태에 맞춘 응답을 생성


- 목적 지향 응답 생성 시 상식, 도메인, 개인화 그래프 등의 **이종 지식그래프 베이스**를 활용  

## Dataset
|제목|내용|위치|
|------|---|---|
|**Commonsense**|한국어 상식 외부지식 그래프|./CommonsenseDataset|
|**Domain**|음악 도메인, 개인화 외부지식 그래프|./GraphDataset|

## Model
|제목|내용|위치|
|------|---|---|
|**[1] Dialogue state tracking**|대화 상태 추적 모델(koBART)|./DST_model|
|**[2] Response generation**|대화 응답 생성 모델(koBART)|./Response_model|

### Setup
```
sudo docker build -t nrf:latest -f DockerFile .
sudo docker run -itd --gpus all --name nrf -v $PWD:/workspace -t nrf:latest
```
