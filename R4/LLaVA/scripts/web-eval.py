import re

import numpy as np
from rouge import Rouge 

import torch
from torchvision.ops import box_iou
import glob
from tqdm import tqdm
import pandas as pd

def eval_web_caption(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_element_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_action_prediction(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        print(cur_pred)
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_action_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_bbox_ground(preds, golds, **kwargs):
    # print('preds[0]', preds[0])
    # print('golds[0]', golds[0])
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )


def eval_action_bbox_ground(preds, golds, **kwargs):
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )


def eval_webqa(preds, golds, **kwargs):
    f1_scores = []
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        try:
            if not pred:
                pred = " "
            cur_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(cur_f1)
        except:
            pass

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )

def eval_element_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

def eval_action_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

# ----------- Process Multi-choice -------------
# def parse_multi_choice_response(response: str, all_choices):
#     """
#     Parse the prediction from the generated response.
#     Return the predicted index e.g., A, B, C, D.
#     """
#     if len(response) == 1:
#         return response.upper()
#     elif not response:
#         return 'a'
#     elif re.match(r"[A-Z]\.", response):
#         return response[0]

#     for char in [',', '.', '!', '?', ';', ':', "'", '"']:
#         response = response.replace(char, "")
#     response = " " + response + " " # add space to avoid partial match

#     ans_with_brack = False
#     candidates = []
#     for choice in all_choices:  # e.g., (A) (B) (C) (D)
#         if f'({choice})' in response:
#             candidates.append(choice)
#             ans_with_brack = True

#     if len(candidates) == 0:
#         for choice in all_choices: # e.g., A B C D
#             if f' {choice} ' in response:
#                 candidates.append(choice)

#     if len(candidates) == 0:  # still not get answer
#         # pred_index = random.choice(all_choices)
#         pred_index = "z"
#     elif len(candidates) > 1:
#         start_indexes = []
#         if ans_with_brack: 
#             for can in candidates:
#                 index = response.rfind(f'({can})')
#                 start_indexes.append(index) # -1 will be ignored anyway
#             # start_indexes = [generated_response.index(f'({can})') for can in candidates]
#         else:
#             for can in candidates:
#                 index = response.rfind(f" {can} ")
#                 start_indexes.append(index)
#         # get the last one
#         pred_index = candidates[np.argmax(start_indexes)]
#     else: # if only one candidate, use it.
#         pred_index = candidates[0]

#     return pred_index

# def parse_multi_choice_response(response: str, all_choices):
#     """
#     Parse the prediction from the generated response.
#     Return the predicted index e.g., A, B, C, D.
#     """
#     # Isolate the part after "Answer:"
#     answer_part = response.split("Answer:")[-1].strip()
    
#     # print(answer_part)

#     # Step 1: Check for single-character responses directly
#     if len(answer_part) == 1:
#         return answer_part.upper()
#     elif not answer_part:
#         return 'a'  # default answer if response is empty
#     elif re.match(r"[A-Z]\.", answer_part):
#         return answer_part[0]

#     # Step 2: Remove punctuation for consistent matching
#     for char in [',', '.', '!', '?', ';', ':', "'", '"', '<\s>']:
#         answer_part = answer_part.replace(char, "")
#     answer_part = " " + answer_part + " "  # add space to avoid partial matches

#     ans_with_brack = False
#     candidates = []

#     # Step 3: Look for answers in the format (A), (B), etc.
#     for choice in all_choices:
#         if f'({choice})' in answer_part:
#             candidates.append(choice)
#             ans_with_brack = True

#     # Step 4: If not found, look for standalone answers (A, B, etc.)
#     if len(candidates) == 0:
#         for choice in all_choices:
#             if f' {choice} ' in answer_part:
#                 candidates.append(choice)

#     # Step 5: Determine final answer based on candidates found
#     if len(candidates) == 0:  # still no answer found
#         pred_index = "z"  # indicate that no valid answer was found
#     elif len(candidates) > 1:
#         # Choose the last occurrence of the choice if multiple candidates found
#         start_indexes = []
#         if ans_with_brack:
#             for can in candidates:
#                 index = answer_part.rfind(f'({can})')
#                 start_indexes.append(index)
#         else:
#             for can in candidates:
#                 index = answer_part.rfind(f" {can} ")
#                 start_indexes.append(index)
#         # Get the last occurring valid candidate
#         pred_index = candidates[np.argmax(start_indexes)]
#     else:  # only one candidate
#         pred_index = candidates[0]

#     return pred_index

def parse_multi_choice_response(response: str, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    # Isolate the part after "Answer:"
    answer_part = response.split("Answer:")[-1].strip()
    for char in [',', '.', '!', '?', ';', ':', "'", '"', '</s>']:
        answer_part = answer_part.replace(char, "")
    # answer_part = " " + answer_part + " "  # add space to avoid partial matches
    
    print(answer_part)

    # Step 1: Check for single-character responses directly
    if len(answer_part) == 1:
        return answer_part.upper()
    elif not answer_part:
        return 'a'  # default answer if response is empty
    elif re.match(r"[A-Z]\.", answer_part):
        return answer_part[0]

    # Step 2: Remove punctuation for consistent matching
    for char in [',', '.', '!', '?', ';', ':', "'", '"', '<\s>']:
        answer_part = answer_part.replace(char, "")
    answer_part = " " + answer_part + " "  # add space to avoid partial matches

    ans_with_brack = False
    candidates = []

    # Step 3: Look for answers in the format (A), (B), etc.
    for choice in all_choices:
        if f'({choice})' in answer_part:
            candidates.append(choice)
            ans_with_brack = True

    # Step 4: If not found, look for standalone answers (A, B, etc.)
    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in answer_part:
                candidates.append(choice)

    # Step 5: Determine final answer based on candidates found
    if len(candidates) == 0:  # still no answer found
        pred_index = "z"  # indicate that no valid answer was found
    elif len(candidates) > 1:
        # Choose the last occurrence of the choice if multiple candidates found
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                index = answer_part.rfind(f'({can})')
                start_indexes.append(index)
        else:
            for can in candidates:
                index = answer_part.rfind(f" {can} ")
                start_indexes.append(index)
        # Get the last occurring valid candidate
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # only one candidate
        pred_index = candidates[0]

    return pred_index

if __name__ == "__main__":
    # root_directory = "/home/abadagab/LLaVA/infer_results/"
    # subdirectories = ["action_ground", "action_prediction", "element_ground", "element_ocr",
    #                 "heading_ocr", "web_caption", "webqa"]
    
    # subdirectories = ["element_ground"]
    # csv_files = ["/home/abadagab/LLaVA/infer_results/test-00000-of-00001-action_ground.csv", "/home/abadagab/LLaVA/infer_results/test-00000-of-00001-action_prediction.csv", "/home/abadagab/LLaVA/infer_results/test-00000-of-00001-element_ground.csv", "/home/abadagab/LLaVA/infer_results/test-00000-of-00001-element_ocr.csv",
    #                 "/home/abadagab/LLaVA/infer_results/test-00000-of-00001-heading_ocr.csv", "/home/abadagab/LLaVA/infer_results/webqa.csv", '/home/abadagab/LLaVA/infer_results/test-00000-of-00001-web_caption.csv']

    # Search for parquet files in each subdirectory
    # for subdir in subdirectories:
    #     pattern = f"{root_directory}/{subdir}/*.csv"
    #     csv_files.extend(glob.glob(pattern, recursive=True))

    csv_files = ['/home/abadagab/LLaVA/infer_results_v2.1/test-00000-of-00001-action_prediction.csv', '/home/abadagab/LLaVA/infer_results_v2.1/test-00000-of-00001-element_ocr.csv']        
    # csv_files = ['/home/abadagab/LLaVA/infer_results_v2/test-00000-of-00001-heading_ocr.csv']        

    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        
        golds = df['answer'].tolist()
        preds = df['predicted_answer'].tolist()

        if "action_ground" in file:
            result = eval_action_ground(preds, golds)
        elif "action_prediction" in file:
            result = eval_action_prediction(preds, golds)
        elif "element_ground" in file:
            result = eval_element_ground(preds, golds)
        elif "element_ocr" in file:
            result = eval_element_ocr(preds, golds)
        elif "heading_ocr" in file:
            result = eval_heading_ocr(preds, golds)
        elif "web_caption" in file:
            result = eval_web_caption(preds, golds)
        elif "webqa" in file:
            result = eval_webqa(preds, golds)
        else:
            print(f"Unknown evaluation type for {file}")
            continue
        
        print(f"Results for {file}: {result}")