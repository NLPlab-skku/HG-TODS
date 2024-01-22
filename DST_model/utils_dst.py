import torch

SLOT_tokens = {
    "노래 제목": "<unused0>",
    "가수": "<unused1>",
    "장르": "<unused2>",
    "작곡가": "<unused3>",
    "작사가": "<unused4>",
    "재생목록 종류": "<unused5>",
    "재생목록 제목": "<unused6>",
    "노래 추천 기준": "<unused7>",
    "개념어": "<unused8>"
}

SLOT_desc = {
    "노래 제목": "추천하는 노래의 제목",
    "가수": "노래를 부른 가수의 활동명",
    "장르": "노래의 장르",
    "작곡가": "노래를 작곡한 사람",
    "작사가": "노래의 가사를 창작한 사람",
    "재생목록 종류": "플레이리스트의 종류: 가장 좋아하는 노래, 최근 들은 노래, 사용자 지정",
    "재생목록 제목": "플레이리스트의 제목",
    "노래 추천 기준": "시스템이 사용자에게 노래를 추천한 이유",
    "개념어": "대화에 포함된 상식"
}

class DSTExample():
    def __init__(self,
                 usr_utt=None,
                 sys_utt=None,
                 history=None,
                 user_id=None,
                 dial_state=None,
                 prev_state=None,
                 guid=None):
        self.usr_utt = usr_utt
        self.sys_utt = sys_utt
        self.user_id = user_id
        self.history = history
        self.dial_state = dial_state
        self.prev_state = prev_state
        self.guid = guid


class DSTFeature():
    def __init__(self,
                 input_ids=None,
                 attention_mask=None,
                 decoder_input_ids=None,
                 decoder_attention_mask=None,
                 labels=None,
                 gold_seq=None,
                 dialog_state=None,
                 index=None,
                 guid=None,
                 slot_type=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels
        self.gold_seq = gold_seq
        self.dialog_state = dialog_state
        self.index = index
        self.guid = guid
        self.slot_type = slot_type


def form_ds_to_seq(dialog_state):
    ds_seq = ""
    for slot, value in dialog_state.items():
        slot_token = SLOT_tokens[slot]
        # 다중 값의 경우 우선 첫번째만
        if "//" in value:
            value = value.split("//")[0]
        ds_seq += slot_token
        ds_seq += value

    return ds_seq


def process_ids(input_ids, decoder_seq, tokenizer, guid, PAD, SOS, EOS, args):
    origin_len = len(input_ids)
    while len(input_ids) > args.max_len - 2:
        input_ids = input_ids[1:]

    # trunc_len = len(input_ids)
    # if origin_len != trunc_len:
    #     print("Truncate {} with length {} to length {}".format(guid, origin_len, trunc_len))

    
    input_ids = SOS + input_ids + EOS
    attention_mask = [1] * len(input_ids)

    if len(input_ids) < args.max_len:
        input_ids = input_ids + PAD * (args.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (args.max_len - len(attention_mask))
    
    decoder_ids = SOS + tokenizer.encode(decoder_seq) + EOS
    labels = decoder_ids[1:]
    decoder_attn_mask = [1] * len(decoder_ids)

    if len(decoder_ids) < args.max_len:
        decoder_ids = decoder_ids + PAD * (args.max_len - len(decoder_ids))
        decoder_attn_mask = decoder_attn_mask + [0] * (args.max_len - len(decoder_attn_mask))

    if len(labels) < args.max_len:
        labels = labels + [-100] * (args.max_len - len(labels))

    assert len(input_ids) == len(attention_mask) == len(decoder_ids) \
        == len(labels) == len(decoder_attn_mask) == args.max_len
        
    return input_ids, attention_mask, decoder_ids, decoder_attn_mask, labels


def convert_examples_to_features(examples, tokenizer, args):

    PAD = tokenizer.encode("<pad>")
    SOS = tokenizer.encode("<s>")
    EOS = tokenizer.encode("</s>")
    MUSIC_PROMPT = tokenizer.encode("<unused9>")
    CONCEPT_PROMPT = tokenizer.encode("<unused10>")
    features = []
    num = 0
    for example in examples:
        # decoder input ids / attention mask
        # slot, value를 text 형태로 늘려서 넣어줌
        if args.decoding == "ind_decoding":
            input_seq = " ".join(example.sys_utt + example.usr_utt)
            input_ids = tokenizer.encode(input_seq)
        
            for key in example.dial_state.keys():
                slot_prompt_token = SLOT_tokens[key]
                # dial_history + domain_related_prompt + slot_related_prompt
                if key =='개념어':
                    input_ids = input_ids + CONCEPT_PROMPT + tokenizer.encode(slot_prompt_token)
                else:
                    input_ids = input_ids + MUSIC_PROMPT + tokenizer.encode(slot_prompt_token)
                decoder_seq = example.dial_state[key]
                
                input_ids, attention_mask, decoder_ids, decoder_attn_mask, labels = \
                    process_ids(
                        input_ids=input_ids, 
                        decoder_seq=decoder_seq, 
                        tokenizer=tokenizer, 
                        guid=example.guid, 
                        PAD=PAD, 
                        SOS=SOS,
                        EOS=EOS,
                        args=args
                    )

                features.append(DSTFeature(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_attn_mask,
                    labels=labels,
                    dialog_state=example.dial_state,
                    gold_seq=decoder_seq,
                    guid=example.guid,
                    slot_type=key,
                    index=num
                ))
                num += 1
        elif args.decoding == "ind_decoding_prompt":
            input_seq = " ".join(example.sys_utt + example.usr_utt)
            input_ids = tokenizer.encode(input_seq)
            
            for key in example.dial_state.keys():
                slot_prompt_token = SLOT_tokens[key]
                slot_desc = SLOT_desc[key]
                # dial_history + domain_related_prompt + slot_related_prompt 
                if key =='개념어':
                    input_ids = input_ids + CONCEPT_PROMPT + tokenizer.encode(slot_prompt_token) + tokenizer.encode(slot_desc)
                else:
                    input_ids = input_ids + MUSIC_PROMPT + tokenizer.encode(slot_prompt_token) + tokenizer.encode(slot_desc)
                decoder_seq = example.dial_state[key]
                
                input_ids, attention_mask, decoder_ids, decoder_attn_mask, labels = \
                    process_ids(
                        input_ids=input_ids, 
                        decoder_seq=decoder_seq, 
                        tokenizer=tokenizer, 
                        guid=example.guid, 
                        PAD=PAD, 
                        SOS=SOS,
                        EOS=EOS,
                        args=args
                    )

                features.append(DSTFeature(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_attn_mask,
                    labels=labels,
                    dialog_state=example.dial_state,
                    gold_seq=decoder_seq,
                    guid=example.guid,
                    slot_type=key,
                    index=num
                ))
                num += 1
        # sequential generation
        else:
            # encoder input ids / attention mask

            # prev O
            # input_seq = " ".join(example.history + example.sys_utt + example.usr_utt)

            # prev X
            input_seq = " ".join(example.sys_utt + example.usr_utt)

            input_ids = tokenizer.encode(input_seq)
            decoder_seq = form_ds_to_seq(example.dial_state)
            
            input_ids, attention_mask, decoder_ids, decoder_attn_mask, labels = \
                process_ids(
                    input_ids=input_ids, 
                    decoder_seq=decoder_seq, 
                    tokenizer=tokenizer, 
                    guid=example.guid,
                    PAD=PAD, 
                    SOS=SOS,
                    EOS=EOS, 
                    args=args
                )

            features.append(DSTFeature(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_ids,
                decoder_attention_mask=decoder_attn_mask,
                labels=labels,
                dialog_state=example.dial_state,
                gold_seq=decoder_seq,
                guid=example.guid,
                slot_type=None, 
                index=num
            ))
            num += 1

    return features

