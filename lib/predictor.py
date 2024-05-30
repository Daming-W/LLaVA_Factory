import argparse
import re
import requests
from PIL import Image
from io import BytesIO

import torch
from transformers import PreTrainedModel

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def su_inference(args, model_name, tokenizer, model, image_processor, image_file, query_dict):
    # used for scenario understanding
    disable_torch_init()
    # make an answer dict
    answer_dict= {key: None for key in query_dict}

    for sce,prompt in query_dict.items():
        qs = prompt[0]
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        text_processor = TextPreprocess(tokenizer, args.conv_mode)
        data_args = model.config
        image_processor = ImagePreprocess(image_processor, data_args)
        model.cuda()

        msg = Message()
        msg.add_message(qs)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()

        image_files = image_parser(args)
        images = load_images(image_files)[0]
        images_tensor = image_processor(images)
        images_tensor = images_tensor.unsqueeze(0).half().cuda()

        stop_str = text_processor.template.separator.apply()[1]
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        answer_dict[sce] = outputs
    
    return answer_dict