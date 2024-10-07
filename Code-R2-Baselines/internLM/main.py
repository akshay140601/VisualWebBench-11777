import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import os
import sys
sys.path.append('/home/saisravy/vbench/models/')
from utils import *
import pandas as pd
import requests
import io
import glob
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def inferencing_internLM(image_path, prompt, model_path = 'OpenGVLab/InternVL2-2B'):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    question = prompt
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    return(f'User: {question}\nAssistant: {response}')

def create_visualweb_prompt(x):
    WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

    HEADING_OCR_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""

    WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

    ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""
    ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

    ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

    ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""
    if x['task_type'] == 'web_caption':
        return WEB_CAPTION_PROMPT
    elif x['task_type'] == 'heading_ocr':
        return HEADING_OCR_PROMPT    
    elif x['task_type'] == 'webqa':
        return WEBQA_PROMPT.format(question= x['question'])
    elif x['task_type'] == 'element_ocr':
        return ELEMENT_OCR_PROMPT.format(bbox_ratio=x['bbox']) 
    elif x['task_type'] == 'element_ground':
        return ELEMENT_GROUND_PROMPT.format(element_desc=x['elem_desc'])
    elif x['task_type'] == 'action_prediction':
        return ACTION_PREDICTION_PROMPT.format(bbox_ratio=x['bbox'], choices_text=x['options'])
    elif x['task_type'] == 'action_ground':
        return ACTION_GROUND_PROMPT.format(instruction=x['instruction'])
    else :
        raise NotImplementedError(f"Task type {x['task_type']} not implemented.")

def make_prediction(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    predicted_answers = []
    for i in range(len(df)):
        prompt = create_visualweb_prompt(df.iloc[i])
        image  = df.iloc[i]['image']
        image_bytes = image['bytes']
        image_stream = io.BytesIO(image_bytes)

        answer = inferencing_internLM(image_path=image_stream, prompt=prompt)
        predicted_answers.append(answer)
    df['predicted_answer'] = predicted_answers 
    df.to_csv(f'{parquet_file_path[:-8]}.csv')  
    return df

# def evaluate_visualwebbench(self, model, accel):
#         pred_answers = [{'id': inputs['id'],
#                         'task_type': inputs['task_type'],
#                         'question': inputs['question'],
#                         'answer': inputs['answer'].tolist() if type(inputs['answer']) == np.ndarray else inputs['answer'],
#                         'prediction' : answer} for inputs, answer in zip(self.inputs, self.gen_answers)]
        
#         pred_pth = os.path.join(self.save_dir, f'{model}_visualwebbench_results.json')
        
#         json.dump(pred_answers, open(pred_pth, "w"))
        
#         task_type_answers = {}
        
#         for answer in pred_answers:
#             task_type = answer['task_type']
#             if task_type not in task_type_answers:
#                 task_type_answers[task_type] = []
#             task_type_answers[task_type].append(answer)
        
#         results = {}
        
#         for task_type, answers in task_type_answers.items():
#             preds, golds = zip(*[(answer['prediction'], answer['answer']) for answer in answers])

#             if task_type == 'web_caption':
#                 results[task_type] = eval_web_caption(preds, golds)
#             elif task_type == 'heading_ocr':
#                 results[task_type] = eval_heading_ocr(preds, golds)
#             elif task_type == 'element_ocr':
#                 results[task_type] = eval_element_ocr(preds, golds)
#             elif task_type == 'action_prediction':
#                 results[task_type] = eval_action_prediction(preds, golds)
#             elif task_type == 'element_ground':
#                 results[task_type] = eval_element_ground(preds, golds)
#             elif task_type == 'action_ground':
#                 results[task_type] = eval_action_ground(preds, golds)
#             elif task_type == 'webqa':
#                 results[task_type] = eval_webqa(preds, golds)
#             else:
#                 raise ValueError(f'{task_type} is not a valid task type.')
        
#         accel.print(results)
#         json.dump(results, open(os.path.join(self.save_dir, f'{model}_visualwebbench_scores.json'), "w"))

if __name__ == '__main__':
    root_directory = "visualwebbench/"
    pattern = f"{root_directory}**/*.parquet"
    parquet_files = glob.glob(pattern, recursive=True)
    for file in tqdm(parquet_files, desc="Processing Parquet files"):
        print(file)
        make_prediction(file)
