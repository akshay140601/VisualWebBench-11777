# write the code for running llava on visual-web-bench dataset

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import requests
import torch
import pandas as pd
import io
import glob
from tqdm import tqdm
import os
import re
import warnings
import time
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
import sys
sys.path.append('/home/saisravy/VisualWebBench-11777/Code-R2-Baselines/LLaVA_model/llava/LLaVA')
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from torchvision.transforms.functional import pil_to_tensor

# from utils import (
#     load_image, 
#     aggregate_llm_attention, aggregate_vit_attention,
#     heterogenous_stack,
#     show_mask_on_image
# )

def append_bbox_to_image(image_bytes, bbox_ratio):
    """Appends the bounding box region to the bottom of the original image."""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    x1, y1, x2, y2 = bbox_ratio

    # Convert fractional coordinates to pixel values
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    # Crop the bounding box
    bbox_crop = image.crop((x1, y1, x2, y2))
    
    # Resize the bounding box crop to match the width of the original image
    bbox_resized = bbox_crop.resize((width, bbox_crop.height))

    # Create a new image to append the original image and resized bounding box
    appended_height = height + bbox_resized.height
    new_image = Image.new("RGB", (width, appended_height))
    new_image.paste(image, (0, 0))  # Paste original image
    new_image.paste(bbox_resized, (0, height))  # Paste resized bbox below

    return new_image

def format_choices_with_letters(choices):
    """
    Takes a list of choices and returns a formatted string with each choice
    prepended by its corresponding letter ('A:', 'B:', etc.).
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_choices = [f"{alphabet[i]}: {choice}" for i, choice in enumerate(choices)]
    return "\n".join(formatted_choices)

def create_visualweb_prompt_cot(x):
    ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

First, provide the reasoning behind your answer and then provide the ID of the chosen element in a single uppercase letter. Format of your answer is:
Thought: <thought>
Answer: <answer>
"""

    ACTION_PREDICTION_PROMPT = """You are given an image of a webpage, focusing on a rectangular region of interest. This image includes the full webpage view with the bounding box appended to the bottom for additional reference. 

Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:

{choices_text}

You should directly tell me your choice as a single uppercase letter, and do not output any explanation or any other contents."""


    ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

First, provide the reasoning behind your answer and then provide the ID of the chosen element in a single uppercase letter. Format of your answer is:
Thought: <thought>
Answer: <answer>
"""


    assert x['task_type'] != 'web_caption'
    assert x['task_type'] != 'heading_ocr'
    assert x['task_type'] != 'webqa'
    assert x['task_type'] != 'element_ocr'
      
    # elif x['task_type'] == 'webqa':
    #     return WEBQA_PROMPT.format(question= x['question'])
    # elif x['task_type'] == 'element_ocr':
    #     return ELEMENT_OCR_PROMPT.format(bbox_ratio=x['bbox']) 
    # elif x['task_type'] == 'element_ground':
    #     return ELEMENT_GROUND_PROMPT.format(element_desc=x['elem_desc'])
    if x['task_type'] == 'action_prediction':
        return ACTION_PREDICTION_PROMPT.format(bbox_ratio=x['bbox'], choices_text=x['options'])
    elif x['task_type'] == 'element_ground':
        return ELEMENT_GROUND_PROMPT.format(element_desc=x['elem_desc'])
    elif x['task_type'] == 'action_ground':
        return ACTION_GROUND_PROMPT.format(instruction=x['instruction'])
    else :
        raise NotImplementedError(f"Task type {x['task_type']} not implemented.")

def create_visualweb_prompt(x):
    WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

    HEADING_OCR_PROMPT = """You are given a screenshot from a webpage, with a heading. The heading is the main title of the webpage and contains the entire crux and meaning of the webpage. It could be the most prominent and largest text in the webpage. Perform OCR to extract the text from the heading exactly as it appears, reading left-to-right and top-to-bottom, as in a book.

                        Guidelines:
                        1. Transcribe text exactly as it appears, including numbers, symbols, hyperlinks, and text on challenging backgrounds. Do not rewrite or summarize (e.g., write "14,900,000" exactly, not "over 14.9 million").
                        2. For unfamiliar terms, acronyms, or specific site names, transcribe them exactly as seen, letter by letter, without guessing similar words or changing the content.
                        3. Do not hallucinate. Provide a direct transcription word-for-word without interpretation or generalization.
                        4. The main heading could be the largest text on the webpage, or the sentence describing what the webpage is about. 

You should directly tell me the text from the main heading, and do not output any explanation or any other contents.
"""

    WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

    ELEMENT_OCR_PROMPT = """You are given an image containing a webpage, focusing on a red rectangular region of interest. This image includes the full webpage view with the bounding box appended to the bottom for reference. Perform OCR to extract the text exactly as it appears, reading left-to-right and top-to-bottom, as in a book.

Guidelines:
1. Focus on the text within the red rectangular region of interest in the original portion of the image and provide a precise transcription. Do not include any text from the appended bounding box unless explicitly requested.
2. If any text within the red rectangular region is cropped (e.g., top or bottom of the region, or words cut vertically), do not guess or attempt to infer the missing parts. Only transcribe what is visible.
3. Transcribe text exactly as it appears, including numbers, symbols, hyperlinks, and text on challenging backgrounds. Do not rewrite or summarize (e.g., write "14,900,000" exactly, not "over 14.9 million").
4. For unfamiliar terms, acronyms, or specific site names, transcribe them exactly as seen, letter by letter, without guessing similar words or changing the content.
5. Do not hallucinate. Provide a direct transcription word-for-word without interpretation or generalization.
6. If explicitly requested, you may extract text from the appended bounding box portion, following the same transcription rules.

Provide the full OCR output for the red rectangular region, without omitting or altering any visible text."""

    
    ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

    ACTION_PREDICTION_PROMPT = """You are given a cropped screenshot of a webpage, specifically focusing on a rectangular region of interest. 
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:

{choices_text}

You should directly tell me your choice as a single uppercase letter, and do not output any explanation or any other contents.
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
        return ACTION_PREDICTION_PROMPT.format(bbox_ratio=x['bbox'], choices_text=format_choices_with_letters(x['options']))
    elif x['task_type'] == 'action_ground':
        return ACTION_GROUND_PROMPT.format(instruction=x['instruction'])
    else :
        raise NotImplementedError(f"Task type {x['task_type']} not implemented.")

def main():
    model_path = "/home/abadagab/LLaVA/llava/llava-ftmodel"

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda:0"
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, # model_base
        model_name, 
        load_8bit, 
        load_4bit, 
        device=device
    )

    root_directory = "/home/saisravy/vbench/visualwebbench"  
    pattern = os.path.join(root_directory, "**", "*.parquet")
    
    parquet_files = glob.glob(pattern, recursive=True)
    num=0
    # vwb_tasks = ['element_ground', 'webqa', 'heading_ocr', 'action_prediction', 'web_caption', 'action_ground', 'element_ocr']
    # parquet_files = [parquet_files[0], parquet_files[3], parquet_files[5]]
    parquet_files = [parquet_files[3], parquet_files[6]]
    # vwb_tasks = ['element_ground', 'webqa', 'heading_ocr', 'action_prediction', 'web_caption', 'action_ground', 'element_ocr']
    vwb_tasks = ['action_prediction', 'element_ocr']

    for file in tqdm(parquet_files, desc = "Processing Parquet Files"):
        print(file)
        df = pd.read_parquet(file)
        total_time = 0
        predicted_answers = []
        num+=1
        for i in tqdm(range(len(df)), desc="processing rows"):
            text_prompt = create_visualweb_prompt(df.iloc[i])
            image = df.iloc[i]['image']
            image_bytes = image['bytes']

            # Apply the appending scheme for OCR-related tasks
            if df.iloc[i]['task_type'] in ['element_ocr', 'action_prediction']:
                bbox_ratio = df.iloc[i]['bbox']
                modified_image = append_bbox_to_image(image_bytes, bbox_ratio)
                image_bytes = io.BytesIO()
                modified_image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()

            # modification for changing system prompt
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            conv = conv_templates[conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream)
            
            image_tensor, images = process_images([image], image_processor, model.config)
            image = images[0]
            image_size = image.size

            if type(image_tensor) is list:
                image_tensor = [image.to(device, dtype=torch.float16) for image in image_tensor]
                image_tensor = image_tensor.to(device, dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(device, dtype=torch.float16)

            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
            
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            prompt = prompt.replace("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",'')
            print(prompt)
            print("===")
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            start_time = time.time() 
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
            end_time = time.time()
            time_taken = end_time - start_time
            total_time += time_taken

            text = tokenizer.decode(outputs["sequences"][0]).strip()
            print("Predicted Answer", text)
            print("Correct Answer", df.iloc[i]['answer'])
            print("\n========\n")
            predicted_answers.append(text)
        df['predicted_answer'] = predicted_answers
        match = re.search(r'([^/]+)\.parquet$', file)
        name = match.group(1)
        # print(f'Code-R2-Baselines/LLaVA_model/predictions/{name}-cot-448-{num}.csv')
        # df.to_csv(f'Code-R2-Baselines/LLaVA_model/predictions/{name}-cot-448-{num}.csv')
        df.to_csv(f'/home/abadagab/LLaVA/infer_results_v2.1/{name}-{vwb_tasks[num-1]}.csv')

        avg_inference_time = total_time/len(df)
        # with open(f'Code-R2-Baselines/LLaVA_model/inference_time_LLaVA-7b-cot.txt', 'a') as txt_file:
        #     txt_file.write(f'The average inference time for {name}-{vwb_tasks[num]} is {avg_inference_time}. The file path is {file}\n')
            
if __name__ == '__main__':
    main()