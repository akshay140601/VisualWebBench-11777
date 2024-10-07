import torch
from config import *
from PIL import Image
from utils.utils import *
import torch.nn.functional as F
from trol.load_trol import load_trol
from torchvision.transforms.functional import pil_to_tensor
import os
import torch
import argparse
import base64
from config import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from utils.utils import *
from datetime import timedelta
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trol.load_trol import load_trol
from eval.create_evaluator import Evaluator
from loader.create_eval_dataset import CreateEvalDataset
from accelerate import Accelerator, InitProcessGroupKwargs
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import os
import sys
#sys.path.append('/home/saisravy/vbench/models/')
from utils import *
import pandas as pd
import requests
import io
import glob
from tqdm import tqdm

def inferencing_trol(image_path, prompt):
    
    inputs = [{'image': image_path, 'question': prompt}]

    # Generate
    with torch.inference_mode():
        _inputs = model.eval_process(inputs=inputs,
                                    data='demo',
                                    tokenizer=tokenizer,
                                    device='cuda:0',
                                    img_token_number=1225)
        generate_ids = model.generate(**_inputs, max_new_tokens=256, use_cache=True)
        response = output_filtering(tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0], model)
    # print(response)
    
    return response

def make_prediction_trol(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    predicted_answers = []
    for i in tqdm(range(len(df))):
        prompt = create_visualweb_prompt(df.iloc[i])
        image  = df.iloc[i]['image']
        image_bytes = image['bytes']
        image_stream = io.BytesIO(image_bytes)
        image = pil_to_tensor(Image.open(image_stream).convert("RGB"))
        resized_img_tensor = F.interpolate(image.unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
        
        answer = inferencing_trol(image_path=resized_img_tensor, prompt=prompt)
        predicted_answers.append(answer)
    df['predicted_answer'] = predicted_answers 
    df.to_csv(f'{parquet_file_path[:-8]}.csv')
    return df

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


root_directory = "Evaluation_Dataset_Path/VisualWebBench/"
subdirectories = ["action_ground", "action_prediction", "element_ground", "element_ocr",
                  "heading_ocr", "web_caption", "webqa"]
parquet_files = []

# Search for parquet files in each subdirectory
for subdir in subdirectories:
    pattern = f"{root_directory}/{subdir}/*.parquet"
    parquet_files.extend(glob.glob(pattern, recursive=True))

print(parquet_files)

model, tokenizer = load_trol(link='TroL-1.8B')

# Move model parameters to GPU
for param in model.parameters():
    if not param.is_cuda:
        param.data = param.to('cuda:0')

# Process each Parquet file
for file in tqdm(parquet_files, desc="Processing Parquet files"):
    final_dataframe = make_prediction_trol(file)