import json
from utils_dst import DSTExample
import pdb

SLOT_MAPS = {
    "slot1": "노래 제목", "slot2": "가수",
    "slot3": "장르", "slot4": "작곡가",
    "slot5": "작사가", "slot7": "재생목록 종류",
    "slot8": "재생목록 제목"
}

RC_criteria = {"slot10": "노래", "slot11": "가수",
               "slot12": "장르", "slot13": "작곡가",
               "slot14": "작사가"}

NEW_SLOT_MAPS = {
    "song title": "노래 제목", "singer": "가수",
    "genre": "장르", "composer": "작곡가",
    "lyricist": "작사가", "playlist type": "재생목록 종류",
    "playlist title": "재생목록 제목"
}

NEW_RC_criteria = {"recommendation": "노래 추천 기준"}


def get_ds(state_dict):
    ds = {}
    
    for slot, lex_slot in NEW_SLOT_MAPS.items():
        value = state_dict[slot]
        
        if value == '':
            ds[lex_slot] = '없음'
        else:
            ds[lex_slot] = value
        
    for slot, lex_slot in NEW_RC_criteria.items():
        value = state_dict[slot]
        
        if value:
            ds["노래 추천 기준"] = value
        

    if "노래 추천 기준" not in ds:
        ds["노래 추천 기준"] = "없음"

    if state_dict["concept"] == "":
        ds["개념어"] = "없음"
    else:
        ds["개념어"] = state_dict["concept"]
    
    assert len(ds) == 9

    return ds


def create_example(dial_json, mode, args=None):

    dial_data = json.load(open(dial_json, encoding="utf-8"))

    examples = []
    for dial in dial_data:
        if dial["cur_size"] == 0:
            continue

        user_id = dial["user_id"]
        dial_num = dial["id"]

        prev_ds = {}
        prev_sys = []
        usr_utt = []
        history = []
        for idx, turn in enumerate(dial["turns"]):

            history = history + usr_utt

            usr_utt = ["사용자: %s" %turn["user_message"]]
            next_sys_utt = ["시스템: %s" %turn["system_message"]]

            # system_slots 가 user slot
            # get dialogue state
            cur_ds = get_ds(turn["system_slots"])

            guid = "%s-%d-%d" %(mode, dial_num, idx)

            examples.append(DSTExample(
                usr_utt=usr_utt,
                sys_utt=prev_sys,
                history=history,
                user_id=user_id,
                dial_state=cur_ds, # system slot, value를 담고 있는 dictionary => label로 사용
                prev_state=prev_ds,
                guid=guid
            ))

            # print("대화 히스토리", history)
            # print("이전 시스템 발화", prev_sys)
            # print("현재 사용자 발화", usr_utt)

            history = history + prev_sys
            prev_sys = next_sys_utt

            for slot, value in cur_ds.items():
                if value != '없음':
                    prev_ds[slot] = value

    return examples


if __name__ == "__main__" :
    dial_json = "./persona_data/train_dialog.json"
    create_example(dial_json)