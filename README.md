# 이종 그래프 융합 목적 지향 대화 시스템

**목적 지향 대화 시스템**(Task-Oriented Dialogue System)이란 특정 업무를 달성하기 위한 목적을 가지고 사용자에게 도움을 주기 위한 대화 시스템

- 해당 대화 시스템은 **음악 추천**을 목적으로 하는 대화 시스템으로 사용자의 개인정보, 상황, 발화 상태에 맞춘 응답을 생성


- 목적 지향 응답 생성 시 상식, 도메인, 개인화 그래프 등의 **이종 지식그래프 베이스**를 활용  

## Dataset
### 대화 코퍼스
|제목|내용|위치|
|------|---|---|
|**음악 도메인 대화 데이터셋**|음악 도메인 기반의 대화 데이터|./CommonsenseDataset|
|**음악 도메인, 일상 대화 데이터셋**|음악 도메인 대화에 일반 상식 일상 대화가 추가된 데이터|./DialogueDataset|
|**매뉴얼 기반 대화 데이터셋**|전자기기 도메인 기반의 대화 데이터|./DialogueDataset|
|**대화 수집기**|대화 수집 시스템|./DialogueDataset/system|

### 지식 그래프
|제목|내용|위치|
|------|---|---|
|**Commonsense**|한국어 상식 외부지식 그래프|./CommonsenseDataset|
|**Domain**|음악 도메인 외부지식 그래프|./DialogueDataset/Music_knowledge_graph|
|**Personal**|개인화 외부지식 그래프|./DialogueDataset/Music_personal_graph||

## Model
|제목|내용|위치|
|------|---|---|
|**[1] Dialogue state tracking**|대화 상태 추적 모델(koBART), 대화 히스토리를 바탕으로 현 발화의 대화 상태 추적|./DST_model|
|**[2] Response generation**|대화 응답 생성 모델(koBART), 대화 상태 추적 결과를 바탕으로 서브그래프 추출, 대화 히스토리와 입력하여 대화 응답 생성 |./Response_model|

### Setup
```
sudo docker build -t nrf:latest -f DockerFile .
sudo docker run -itd --gpus all --name nrf -v $PWD:/workspace -t nrf:latest
```
