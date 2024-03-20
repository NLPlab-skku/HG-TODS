# Response Generation model

- 음악 추천 사이트에서 크롤링한 음악 도메인 그래프
- 개인화 정보 그래프


### 학습 방법

- {'train.json', 'valid.json', 'test.json', 'songs.json', 'personal.json', 'nodemap.json'}을 ./data 디렉토리에 위치

```
bash run.sh
```

```
python main.py  \
    -- datamodule.batch_size=8
    -- gradient_accumulation_steps=1
    -- method='BART-gnn-dualrgat'
```

- method 옵션을 ['BART', 'BART-triple', 'BART-gnn-dualrgat']으로 설정하여 BART Fine-tuning, 트리플, 그래프 인코더 방식으로 학습 수행

- 학습 결과를 ROUGE-1, ROUGE-2, ROUGE-L로 평가

- 생성 응답 결과 ./experiment/method/generated.json에 저장

### 학습 결과
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">메소드</th>
    <th class="tg-c3ow">ROUGE-1</th>
    <th class="tg-c3ow">ROUGE-2</th>
    <th class="tg-c3ow">ROUGE-L</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">KoBART </td>
    <td class="tg-c3ow">17.09</td>
    <td class="tg-c3ow">5.68</td>
    <td class="tg-c3ow">17.90</td>
  </tr>
  <tr>
    <td class="tg-c3ow">KoBART /w Song R-GAT</td>
    <td class="tg-c3ow">30.67</td>
    <td class="tg-c3ow">13.32</td>
    <td class="tg-c3ow">30.33</td>
  </tr>
  <tr>
    <td class="tg-c3ow">KoBART /w Song R-GAT & Commonsense R-GAT</td>
    <td class="tg-c3ow">30.98</td>
    <td class="tg-c3ow">13.94</td>
    <td class="tg-c3ow">30.59</td>
  </tr>
</tbody>
</table>

### 응답 예시
```
User : "어쿠맨이 무슨 곡들 작곡했는지 검색해줄래?"
System : "어쿠맨이 작곡한 곡으로는 남진의 오빠 아직 살아있다, 고우리의 단화가 있어요."
User: "남자 아이돌 노래는 유독 신나는 노래밖에 없는 것 같은데?"
System : "그럴리가요. 사람이면 다양한 감정을 모두 노래하죠. 잔잔한 아이돌 노래 들어보실래요?"
User: "마지막으로 내가 가장 많이 들은 가수가 부른 노래 부탁할게!"

System Response(generated) : 
"가장 많이 들은 가수인 차병원의 곡이 검색되지 않아, 좋아하는 가수인 혜진이의 연약한 여자 틀어드릴게요."
```
