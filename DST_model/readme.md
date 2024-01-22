# Dialogue State Tracking

음악 도메인 대화와 일반 상식 활용 대화를 통합한 말뭉치를 대상으로 학습

### 음악 도메인, 일상 대화 데이터셋
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow">Train</th>
    <th class="tg-c3ow">Dev</th>
    <th class="tg-c3ow">Test</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Dialogues</td>
    <td class="tg-c3ow">1,481</td>
    <td class="tg-c3ow">298</td>
    <td class="tg-c3ow">297</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Utterances</td>
    <td class="tg-c3ow">18,386</td>
    <td class="tg-c3ow">3,776</td>
    <td class="tg-c3ow">3,732</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Triples of music domain KG</td>
    <td class="tg-c3ow" colspan="3">226,823</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Triples of commonsense KG</td>
    <td class="tg-c3ow" colspan="3">7,971</td>
  </tr>
</tbody>
</table>
  

### 학습 방법
'''
bash DO.train.sh
'''

'''
python main_dst.py \
    --output_dir ${OUT_DIR} \
    --do_train \
    --do_dev \
    --do_eval \
    --dataset_path "./data_230222" \
    --eval_concept \
    --decoding "sequential" \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --n_epochs 5
'''

decoding 옵션을 sequential, ind_decoding으로 설정하여 순차적 디코딩, 독립적 디코딩 수행

학습 결과를 Joint Goal Accuracy(JGA)로 평가

### 학습 결과
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">메소드</th>
    <th class="tg-c3ow">JGA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">KoBART w/ 독립적 디코딩</td>
    <td class="tg-c3ow">0.597</td>
  </tr>
  <tr>
    <td class="tg-c3ow">KoBART w/ 순차적 디코딩</td>
    <td class="tg-c3ow">0.680</td>
  </tr>
</tbody>
</table>