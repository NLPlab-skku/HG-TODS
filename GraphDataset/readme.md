# Graph Dataset

- 음악 추천 사이트에서 크롤링한 음악 도메인 그래프
- 개인화 정보 그래프

## Dataset
|이름|내용|
|------|---|
|nodemap.json|"title", "person" entity의 노드 ID가 매핑된 nodemap|
|personal.json|"name", "playlist" 등으로 구성된 임의의 사용자 별 개인화 정보 그래프|
|song.json|"title", "genre", "artist" 등으로 구성된 음악 도메인 그래프|
|bartenc_embedding|BART encoder 임베딩으로 초기화 된 음악 도메인 그래프 노드 임베딩|
### 예시
- song.json

'''
 {"title": "It's Raining Men (The Weather Girls) (Remix ver.)", "genre": ["일렉트로닉"], "artist": ["한용진"], "writer": [""], "composer": [""]}
'''
