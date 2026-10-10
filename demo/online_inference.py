import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_online
from pathlib import Path
from config import *
from paths import resolve_output_dir
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody, draw_skeleton

def main(args):
    print(args)
    utils.set_seed(args.seed)

    # extract pose
    pose_data = pose_extraction(args.online_video)

    print(f"Creating dataset:")

    online_data = S2T_Dataset_online(args=args)
    print(online_data)
    online_data.rgb_data = args.online_video
    online_data.pose_data = pose_data

    online_sampler = torch.utils.data.SequentialSampler(online_data)
    online_dataloader = DataLoader(online_data,
                                batch_size=1,
                                collate_fn=online_data.collate_fn,
                                sampler=online_sampler,)

    print(f"Creating model:")
    model = Uni_Sign(
        args=args
    )
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu', weights_only=False)['model']

        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    else:
        raise NotImplementedError

    model.eval()
    model.to(torch.bfloat16)

    inference(online_dataloader, model)

def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, [W, H]

def pose_extraction(video_path):
    max_workers = 16
    wholebody = Wholebody(
        to_openpose=False,
        mode="lightweight",
        backend="onnxruntime",
        device="cuda"
    )

    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return

    vid_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vid_data.append(frame)
    cap.release()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in vid_data]
        for f in tqdm(futures, desc="Processing frames", total=len(vid_data)):
            results.append(f.result())

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

    return data

def inference(data_loader, model):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            stack_out = model(src_input, tgt_input)

            output = model.generate(stack_out,
                                    max_new_tokens=100,
                                    num_beams=4,
                                    )

            for i in range(len(output)):
                tgt_pres.append(output[i])

    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    print(f"Prediction result is: {tgt_pres[0]}")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        args.output_dir = resolve_output_dir(args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
