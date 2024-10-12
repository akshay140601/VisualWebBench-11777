import torch
from config import *
from PIL import Image
from utils.utils import *
from model.load_model import load_model
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd
import io

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

# model selection
size = '7b' # [Select One] '0.5b' (transformers more recent version) | '1.8b' | '3.8b' (transformers==4.37.2) | '7b'

# User prompt
prompt_type="with_image" # Select one option "text_only", "with_image"
# img_path='figures/demo.png'
# question="Where is the cursor located?"

# loading model
model, tokenizer = load_model(size=size)

file_path = "/home/achuthchandrasekhar/Phantom/Phantom/Evaluation_Dataset_Path/VisualWebBench/element_ground/test-00000-of-00001.parquet"
    
df = pd.read_parquet(file_path)


# prompt type -> input prompt
if prompt_type == 'with_image':
    # Image Load
    question = create_visualweb_prompt(df.iloc[0])
    question_2 = create_visualweb_prompt(df.iloc[1])
    image  = df.iloc[0]['image']
    image_2 = df.iloc[1]['image']
    image_bytes = image['bytes']
    image_bytes_2 = image_2['bytes']
    image_stream = io.BytesIO(image_bytes)
    image_stream_2 = io.BytesIO(image_bytes_2)
    image = pil_to_tensor(Image.open(image_stream).convert("RGB"))
    image_2 = pil_to_tensor(Image.open(image_stream_2).convert("RGB"))
    resized_img_tensor = F.interpolate(image.unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
    resized_img_tensor_2 = F.interpolate(image_2.unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
    # image = pil_to_tensor(Image.open(img_path).convert("RGB"))
    #inputs = [{'image': image, 'question': question}]
    inputs = [{'image': resized_img_tensor, 'question': question}, {'image': resized_img_tensor_2, 'question': question_2}]

elif prompt_type=='text_only':

    question = create_visualweb_prompt(df.iloc[0])
    inputs = [{'question': question}]

# cpu -> gpu
for param in model.parameters():
    if not param.is_cuda:
        param.data = param.cuda()

# Generate
with torch.inference_mode():

    # Model
    _inputs = model.eval_process(inputs=inputs,
                                data='demo',
                                tokenizer=tokenizer,
                                device='cuda:0')
    generate_ids = model.generate(**_inputs, do_sample=False, max_new_tokens=256)
for ans in generate_ids:
    answer = tokenizer.batch_decode(ans, skip_special_tokens=True)[0]
    print(answer)


