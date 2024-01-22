import os
import json
import torch
import collections
import argparse
import logging
import numpy as np
import random
import re

import dataset_dst
from utils_dst import *

import pandas as pd
from kobart import get_kobart_tokenizer
from model import KoBART
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartForConditionalGeneration


tokens_SLOT = {
    "<unused0>": "노래 제목",
    "<unused1>": "가수",
    "<unused2>": "장르",
    "<unused3>": "작곡가",
    "<unused4>": "작사가",
    "<unused5>": "재생목록 종류",
    "<unused6>": "재생목록 제목",
    "<unused7>": "노래 추천 기준",
    "<unused8>": "개념어"
}

SLOTS = ["노래 제목", "가수", "장르", "작곡가", "작사가", "재생목록 종류", "재생목록 제목", "노래 추천 기준", "개념어"]



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        logging.info('No GPU available, using the CPU instead.')


def load_dataset(mode, tokenizer, args):

    # data_2023 폴더 내부의 파일
    path = os.path.join(args.dataset_path, "%s_dialog.json" %mode)    
    # examples.append(DSTExample(
    #             usr_utt=usr_utt,  user 발화
    #             sys_utt=prev_sys, 이전 turn system 발화
    #             history=history,  발화 history (이번 turn user 발화를 포함한 모든 발화)
    #             user_id=user_id, user id
    #             dial_state=cur_ds, system slot, value를 담고 있는 dictionary
    #             prev_state=prev_ds, 이전 turn system slot, value를 담고 있는 dictionary
    #             guid=guid 발화 id
    #         ))
    examples = dataset_dst.create_example(path, mode)
    print("Total %d number of examples for %s" %(len(examples), mode))
    # features.append(DSTFeature(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_ids,
    #         decoder_attention_mask=decoder_attn_mask,
    #         labels=labels, => example 에서의 dial_state
    #         dialog_state=example.dial_state,
    #         gold_seq=decoder_seq,
    #         guid=example.guid,
    #         index=num
    #     ))

    features = convert_examples_to_features(examples, tokenizer, args)
    # dialog_state, guid 빼고 다 넣음
    temp = [f.input_ids for f in features]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attn_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_dec_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
    all_dec_attn_mask = torch.tensor([f.decoder_attention_mask for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_index = torch.tensor([f.index for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attn_mask, all_dec_input_ids, all_dec_attn_mask, all_labels, all_index)

    return features, dataset


def train(model, train_dataset, train_features, args, tokenizer):

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.do_dev:
        dev_features, dev_dataset = load_dataset("dev", kobart_tokenizer, args)
        dev_loader = DataLoader(dev_dataset, batch_size=args.test_batch_size)
    else:
        dev_loader = None

    t_total = len(train_loader) // args.gradient_accumulation_steps * args.n_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_step, num_training_steps=t_total
    )

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_loader))
    logging.info("  Num Epochs = %d", args.n_epochs)
    logging.info(
        "  Total train batch size (w. parallel, accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("  Seed = %d", args.seed)

    model.zero_grad()
    train_iterator = trange(int(args.n_epochs), desc="Epoch")
    set_seed(args)

    global_step = 0.0
    tr_loss = 0.0

    for now_epoch in train_iterator:
        model.train()

        logging_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Train iteration")

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            result = model(input_ids=batch[0],\
                           attention_mask=batch[1],\
                           decoder_input_ids=batch[2],\
                           decoder_attention_mask=batch[3],\
                           labels=batch[4])

            loss = result.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            logging_loss += loss.item()
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        tr_loss += logging_loss

        logging.info('********** Train Result **********')
        logging.info('Epoch / Total Epoch : {} / {}'.format(now_epoch + 1, args.n_epochs))
        logging.info('Loss : {:.4f}'.format(logging_loss))

        # tb_writer.add_scalar("train_loss", logging_loss, global_step)

        # evaluation
        if args.do_dev:
            dev_loss = validate(dev_loader, model, args)
            logging.info("Dev Loss : {:.4f}".format(dev_loss))
        # tb_writer.add_scalar("dev_acc", eval_result["accuracy"], global_step)

        # save model
        if now_epoch >= 3:
            # best_acc = eval_result["accuracy"]
            save_path = os.path.join(args.output_dir, "checkpoint-%d" %now_epoch)
            os.makedirs(save_path, exist_ok=True)

            model.save_pretrained(save_path)

    logging.info('********** Total Train Result **********')
    logging.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

    # save model
    save_path = args.output_dir
    model.save_pretrained(save_path)

    save_path = os.path.join(args.train_cache, args.trained_weight)
    torch.save(model.state_dict(), save_path)
# end of train

def validate(dev_loader, model, args):

    logging.info("***** Running validation *****")
    logging.info("  Num examples = %d", len(dev_loader))

    dev_loss = 0.0
    epoch_iterator = tqdm(dev_loader, desc="Train iteration")

    for step, batch in enumerate(epoch_iterator):

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)

            result = model(input_ids=batch[0], \
                           attention_mask=batch[1], \
                           decoder_input_ids=batch[2], \
                           decoder_attention_mask=batch[3], \
                           labels=batch[4])

            dev_loss += result.loss.item()

    return dev_loss


def evaluate(test_dataset, test_features, model, args, tokenizer):
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    logging.info("***** Running Evaluation *****")
    logging.info("  Num examples = %d", len(test_loader))
    logging.info("  Batch size = %d", args.test_batch_size)

    model.eval()

    loss_fct = nn.CrossEntropyLoss()

    test_iterator = tqdm(test_loader, desc="Test Evaluating")

    # 생성 결과 저장 format
    # pred_list = []
    # gold_list = []
    gen_result = {}

    # 모델 시간 재기
    import time
    starttime = time.time()
    
    for step, batch in enumerate(test_iterator):

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            pred = []

            past = None
            input_ids = batch[0]
            attention_mask = batch[1]
            index = batch[5]

            gold_seq = test_features[index].gold_seq
            guid = test_features[index].guid

            first_len = len(input_ids)
            decoder_ids = torch.tensor([tokenizer.bos_token_id] * args.test_batch_size, \
                                       dtype=torch.long, device=args.device).unsqueeze(-1)

            while first_len + len(pred) < args.max_len:

                result = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_ids,
                               past_key_values=past)

                lm_logits = result.logits
                past = result.past_key_values
                # greedy
                new_token = torch.argmax(lm_logits[:, -1, :], dim=-1)

                # 답변 생성을 끝내는 경우
                if new_token.tolist() == [tokenizer.eos_token_id]:
                    break

                pred += new_token.tolist()
                decoder_ids = new_token.unsqueeze(0)
            # guid => 대화의 id
            if args.decoding == "ind_decoding" or args.decoding == "ind_decoding_prompt":
                gen_result.setdefault(guid, {})
                # gen_result[guid][slot][gold]
                #                      L[pred]
                input_ids = input_ids.squeeze()
                slot = test_features[index].slot_type
                
                gen_result[guid][slot] = {}
                gen_result[guid][slot]['gold'] = gold_seq

                if pred == []:
                    gen_result[guid][slot]['pred'] = ""
                else:
                    gen_result[guid][slot]['pred'] = pred
            # sequence
            else:
                gen_result.setdefault(guid, {})
                gen_result[guid]["gold"] = gold_seq

                if pred == []:
                    gen_result[guid]["pred"] = ""

                else:
                    gen_result[guid]["pred"] = pred

            if step < 10:
                logging.info(" Making Sample : {}".format(tokenizer.decode(pred, skip_special_tokens=True)))

    
    # 시간
    endtime = time.time()
    print((endtime - starttime) / 1000 / len(test_loader))
    # to get score
    # 여기서 계산함!
    # gen_result[guid]["gold"] or gen_result[guid]["pred"] 
    results = get_metric(gen_result, tokenizer, args.eval_concept, args.decoding)

    # save the generation result
    json.dump(gen_result, open(os.path.join(args.output_dir, "gen_result.json"), "w"),
              indent=4, ensure_ascii=False)

    return results
# end of evaluate function

def get_metric(gen_result, tokenizer, eval_concept=True, decoding="seqential"):
    if decoding == "sequential":
        if not eval_concept:
            jga = []
            slot_acc = []
            results = {}
            for key, value in gen_result.items():

                gold = value["gold"]
                pred = value["pred"]
                pred = tokenizer.decode(pred)

                try:
                    # re를 통해서 매칭된 부분의 튜플을 반환 (.*) 부분을 다 반환
                    pred_groups = re.search('<unused0>(.*)<unused1>(.*)<unused2>(.*)<unused3>(.*)<unused4>(.*)<unused5>(.*)<unused6>(.*)<unused7>(.*)<unused8>(.*)', pred).groups()
                except:
                    jga.append(0.0)
                    slot_acc.extend([0.0]*8)
                    continue

                gold_groups = re.search('<unused0>(.*)<unused1>(.*)<unused2>(.*)<unused3>(.*)<unused4>(.*)<unused5>(.*)<unused6>(.*)<unused7>(.*)<unused8>(.*)', gold).groups()

                assert len(pred_groups) == len(gold_groups) == len(SLOT_tokens)

                flag = True
                for p, g in zip(pred_groups, gold_groups):
                    if p.strip() != g.strip():
                        flag = False
                        slot_acc.append(0.0)
                    else:
                        slot_acc.append(1.0)


                if flag:
                    jga.append(1.0)
                else:
                    jga.append(0.0)

            # jga => joint goal metrics
            results["jga"] = np.mean(jga)
            results["slot_acc"] = np.mean(slot_acc)

            return results
        else:
            # concept 파트는 분리해서 평가하도록 코드 수정 
            concept_jga = []
            concept_slot_acc = []
            jga = []
            slot_acc = []
            results = {}
            for key, value in gen_result.items():

                gold = value["gold"]
                pred = value["pred"]
                pred = tokenizer.decode(pred)

                gold_groups = re.search('<unused0>(.*)<unused1>(.*)<unused2>(.*)<unused3>(.*)<unused4>(.*)<unused5>(.*)<unused6>(.*)<unused7>(.*)<unused8>(.*)', gold).groups()
                try:
                    pred_groups = re.search('<unused0>(.*)<unused1>(.*)<unused2>(.*)<unused3>(.*)<unused4>(.*)<unused5>(.*)<unused6>(.*)<unused7>(.*)<unused8>(.*)', pred).groups()
                except:
                    # concept이 포함된 대화에 대해서
                    if gold_groups[7] != '없음':
                        concept_jga.append(0.0)
                        concept_slot_acc.extend([0.0]*8)
                    # concept이 포함되지 않은 대화에 대해서
                    else:
                        jga.append(0.0)
                        slot_acc.extend([0.0]*8)
                    continue

                assert len(pred_groups) == len(gold_groups) == len(SLOT_tokens)
                
                # concept이 포함된 대화에 대해서
                if gold_groups[7] != '없음':
                    flag = True
                    for p, g in zip(pred_groups, gold_groups):
                        if p.strip() != g.strip():
                            flag = False
                            concept_slot_acc.append(0.0)
                        else:
                            concept_slot_acc.append(1.0)

                    if flag:
                        concept_jga.append(1.0)
                    else:
                        concept_jga.append(0.0)
                # concept이 포함되지 않은 대화에 대해서
                else:
                    flag = True
                    for p, g in zip(pred_groups, gold_groups):
                        if p.strip() != g.strip():
                            flag = False
                            slot_acc.append(0.0)
                        else:
                            slot_acc.append(1.0)

                    if flag:
                        jga.append(1.0)
                    else:
                        jga.append(0.0)

            results["jga"] = np.mean(jga)
            results["slot_acc"] = np.mean(slot_acc)
            results["concept_jga"] = np.mean(concept_jga)
            results["concept_slot_acc"] = np.mean(concept_jga)

            return results
    # ind_decoding 평가 코드 작성 -> gen_result 형식 바꿈, eval_concept 추가
    elif decoding == "ind_decoding" or decoding == "ind_decoding_prompt":
        if not eval_concept:
            jga = []
            slot_acc = []
            results = {}
            for guid, guid_dic in gen_result.items():
                # gen_result[guid][slot][gold]
                #                      L[pred]
                flag = True
                for slot in SLOTS:
                    try:
                        gold = guid_dic[slot]["gold"]
                        pred = guid_dic[slot]["pred"]
                        pred = tokenizer.decode(pred)
                    except:
                        breakpoint()
                    if gold.strip() != pred.strip():
                        flag = False
                        slot_acc.append(0.0)
                    else:
                        slot_acc.append(1.0)
                        
                if flag:
                    jga.append(1.0)
                else:
                    jga.append(0.0)
            # breakpoint()
            results["jga"] = np.mean(jga)
            results["slot_acc"] = np.mean(slot_acc)
        else:   
            concept_jga = []
            concept_slot_acc = []
            jga = []
            slot_acc = []
            results = {}
            for guid, guid_dic in gen_result.items():
                # gen_result[guid][slot][gold]
                #                      L[pred]
                # concept 이 포함되지 않은 대화
                if guid_dic["개념어"]["gold"] != "없음":
                    flag = True
                    for slot in SLOTS:
                        try:
                            gold = guid_dic[slot]["gold"]
                            pred = guid_dic[slot]["pred"]
                            pred = tokenizer.decode(pred)
                        except:
                            breakpoint()
                        if gold.strip() != pred.strip():
                            flag = False
                            concept_slot_acc.append(0.0)
                        else:
                            concept_slot_acc.append(1.0)
                            
                    if flag:
                        concept_jga.append(1.0)
                    else:
                        concept_jga.append(0.0)
                else:
                    flag = True
                    for slot in SLOTS:
                        try:
                            gold = guid_dic[slot]["gold"]
                            pred = guid_dic[slot]["pred"]
                            pred = tokenizer.decode(pred)
                        except:
                            breakpoint()
                        if gold.strip() != pred.strip():
                            flag = False
                            slot_acc.append(0.0)
                        else:
                            slot_acc.append(1.0)
                            
                    if flag:
                        jga.append(1.0)
                    else:
                        jga.append(0.0)
            
            results["jga"] = np.mean(jga)
            results["slot_acc"] = np.mean(slot_acc)
            results["concept_jga"] = np.mean(concept_jga)
            results["concept_slot_acc"] = np.mean(concept_slot_acc)
            
        return results
        


if __name__ == "__main__":
    # arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dataset_path", type=str, default="./persona_data", help="Path train data")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--learning_rate", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--warmup_step", type=int, default=0, help="step of linear warmup")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--output_dir", type=str, default="./model_cache",
                        help="Path, url or short name of the model")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=60, help="Random seed for initialization")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_dev", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--eval_concept", action='store_true')
    parser.add_argument("--decoding", choices=["sequential", "ind_decoding", "ind_decoding_prompt"])
    args = parser.parse_args()

    # logging console print
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=args.output_dir + '/log.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # load tokenizer
    kobart_tokenizer = get_kobart_tokenizer()
    # training
    if args.do_train:
        train_features, train_dataset = load_dataset("train", kobart_tokenizer, args)

        model = KoBART()
        model.to(args.device)
        
        train(model, train_dataset, train_features, args, kobart_tokenizer)

        # after training save model


    # evaluation
    if args.do_eval:
        test_features, test_dataset = load_dataset("test", kobart_tokenizer, args)
        config = BartConfig.from_pretrained(args.output_dir)
        model = BartForConditionalGeneration.from_pretrained(args.output_dir, config=config)
        model.to(args.device)
        eval_result = evaluate(test_dataset, test_features, model, args, kobart_tokenizer)
        logging.info('********** Eval Result **********')
        logging.info(eval_result)
