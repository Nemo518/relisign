import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
from config import rgb_dirs, pose_dirs

# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


def fit_box_to_image(center_x, center_y, box_hw, hw):
    H, W = hw
    max_side = min(H, W)
    if max_side <= 1:
        return np.array([0, 0, max(1, W), max(1, H)], dtype=np.int16)

    box_hw = int(round(box_hw / 2.0) * 2)
    box_hw = max(2, min(box_hw, max_side))
    half = box_hw / 2.0

    if W <= box_hw:
        center_x = W / 2.0
    else:
        center_x = float(np.clip(center_x, half, W - half))

    if H <= box_hw:
        center_y = H / 2.0
    else:
        center_y = float(np.clip(center_y, half, H - half))

    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x1 = min(max(0, x1), W - box_hw)
    y1 = min(max(0, y1), H - box_hw)
    x2 = x1 + box_hw
    y2 = y1 + box_hw
    return np.array([x1, y1, x2, y2], dtype=np.int16)


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2
    
    def compute_bbox(keypoints):
        keypoints = np.clip(keypoints, 0.0, 1.0)
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        
        return (max_x+min_x)/2, (max_y+min_y)/2, (max_x-min_x), (max_y-min_y)
    H,W = hw
    
    if left_keypoints is None:
        left_keypoints = np.zeros([1,21,2])
        
    if right_keypoints is None:
        right_keypoints = np.zeros([1,21,2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W*left_mean_x
    left_mean_y = H*left_mean_y
    
    left_diff_x = W*left_diff_x
    left_diff_y = H*left_diff_y
    
    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x,left_diff_y)
    
    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W*right_mean_x
    right_mean_y = H*right_mean_y
    
    right_diff_x = W*right_diff_x
    right_diff_y = H*right_diff_y
    
    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x,right_diff_y)
    
    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)
    if box_hw == 0:
        left_new_box = np.zeros((len(left_mean_x), 4), dtype=np.int16)
        right_new_box = np.zeros((len(right_mean_x), 4), dtype=np.int16)
        return left_new_box, right_new_box, box_hw

    box_hw = min(box_hw, min(H, W))

    left_new_box = np.stack(
        [fit_box_to_image(cx, cy, box_hw, hw) for cx, cy in zip(left_mean_x, left_mean_y)],
        axis=0,
    )
    right_new_box = np.stack(
        [fit_box_to_image(cx, cy, box_hw, hw) for cx, cy in zip(right_mean_x, right_mean_y)],
        axis=0,
    )

    return left_new_box, right_new_box, box_hw


def crop_with_padding(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    canvas = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(img.shape[1], x2)
    src_y2 = min(img.shape[0], y2)

    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return canvas

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = img[src_y1:src_y2, src_x1:src_x2, :]
    return canvas


def select_hand_indices(candidate_indices, candidate_scores, sample_mode, top_k_ratio=0.1):
    if len(candidate_indices) == 0:
        return None

    candidate_indices = np.asarray(candidate_indices)
    candidate_scores = np.asarray(candidate_scores)
    if not 0 < top_k_ratio <= 1:
        raise ValueError(f"top_k_ratio must be in (0, 1], got {top_k_ratio}")
    sample_size = max(1, int(np.ceil(top_k_ratio * len(candidate_indices))))
    if sample_size >= len(candidate_indices):
        return np.sort(candidate_indices)

    if sample_mode == "confidence_topk":
        chosen_order = np.argsort(-candidate_scores, kind="stable")[:sample_size]
        chosen_indices = candidate_indices[chosen_order]
    elif sample_mode == "confidence_weighted":
        weights = candidate_scores + 1e-5
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            probabilities = np.full(len(candidate_indices), 1.0 / len(candidate_indices))
        else:
            probabilities = weights / weights_sum
        chosen_indices = np.random.choice(
            candidate_indices.tolist(),
            size=sample_size,
            replace=False,
            p=probabilities,
        )
    elif sample_mode == "uniform":
        chosen_positions = build_uniform_indices(len(candidate_indices), sample_size)
        chosen_indices = candidate_indices[chosen_positions]
    elif sample_mode == "random":
        chosen_indices = np.random.choice(
            candidate_indices.tolist(),
            size=sample_size,
            replace=False,
        )
    else:
        raise ValueError(f"Unsupported rgb frame sampling mode: {sample_mode}")

    return np.sort(np.asarray(chosen_indices))


def build_uniform_indices(total_length, target_length):
    total_length = int(total_length)
    target_length = int(target_length)
    if total_length <= 0:
        return np.array([], dtype=np.int64)
    if target_length <= 0 or total_length <= target_length:
        return np.arange(total_length, dtype=np.int64)

    indices = np.linspace(0, total_length - 1, num=target_length)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, total_length - 1)
    indices = np.unique(indices)

    if len(indices) == target_length:
        return indices

    expanded = []
    seen = set()
    cursor = 0
    for value in indices.tolist():
        if value not in seen:
            expanded.append(value)
            seen.add(value)
    while len(expanded) < target_length and cursor < total_length:
        if cursor not in seen:
            expanded.append(cursor)
            seen.add(cursor)
        cursor += 1
    expanded = np.array(sorted(expanded[:target_length]), dtype=np.int64)
    return expanded


def build_rgb_meta(selected_scores, sampled_indices, filter_indices, total_frames):
    if sampled_indices is None or len(sampled_indices) == 0:
        return torch.zeros(1, 3, dtype=torch.float32)

    selection_density = len(sampled_indices) / max(1, len(filter_indices))
    hand_presence_ratio = len(filter_indices) / max(1, total_frames)

    meta = np.stack(
        [
            np.asarray(selected_scores, dtype=np.float32),
            np.full(len(sampled_indices), selection_density, dtype=np.float32),
            np.full(len(sampled_indices), hand_presence_ratio, dtype=np.float32),
        ],
        axis=-1,
    )
    return torch.tensor(meta, dtype=torch.float32)


def _fill_empty_hand_dict(support_rgb_dict, image_size=112):
    """Fill support_rgb_dict with empty hand data (pose-only fallback)."""
    support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
    support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
    support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
    support_rgb_dict['left_rgb_meta'] = torch.zeros(1, 3, dtype=torch.float32)
    support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
    support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
    support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)
    support_rgb_dict['right_rgb_meta'] = torch.zeros(1, 3, dtype=torch.float32)


def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform, sample_mode="confidence_topk", top_k_ratio=0.1):
    support_rgb_dict = {}

    normalized_skeletons = []
    normalized_confs = []
    for skeleton_frame, conf_frame in zip(skeletons, confs):
        skeleton_frame = np.asarray(skeleton_frame)
        conf_frame = np.asarray(conf_frame)

        # Some CE-CSL pose files contain multiple detected persons in a frame.
        # RGB-support expects a single person sequence, so keep the person with
        # the highest mean confidence for each frame.
        if conf_frame.ndim == 2 and conf_frame.shape[0] > 1:
            best_idx = int(np.argmax(conf_frame.mean(axis=1)))
            conf_frame = conf_frame[best_idx:best_idx + 1]
            skeleton_frame = skeleton_frame[best_idx:best_idx + 1]
        elif conf_frame.ndim == 1:
            conf_frame = conf_frame[None]
            skeleton_frame = skeleton_frame[None]

        normalized_skeletons.append(skeleton_frame)
        normalized_confs.append(conf_frame)

    confs = np.stack(normalized_confs, axis=0)
    skeletons = np.stack(normalized_skeletons, axis=0)

    # Prefer higher-confidence hand frames so RGB support sees cleaner crops.
    left_confs_filter = confs[:,0,91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
        left_rgb_meta = torch.zeros(1, 3, dtype=torch.float32)
    else:
        
        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:,0,[95,99,103,107,111]].min(-1)
        left_sampled_indices = select_hand_indices(left_confs_filter_indices, left_confs, sample_mode, top_k_ratio)
        left_index_lookup = {int(frame_idx): idx for idx, frame_idx in enumerate(left_confs_filter_indices)}
        left_selected_scores = np.asarray([left_confs[left_index_lookup[int(frame_idx)]] for frame_idx in left_sampled_indices], dtype=np.float32)
        left_rgb_meta = build_rgb_meta(left_selected_scores, left_sampled_indices, left_confs_filter_indices, len(confs))
        
        left_skeletons = skeletons[left_sampled_indices,0,91:112]

    right_confs_filter = confs[:,0,112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None
        right_rgb_meta = torch.zeros(1, 3, dtype=torch.float32)
        
    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:,0,[95+21,99+21,103+21,107+21,111+21]].min(-1)
        right_sampled_indices = select_hand_indices(right_confs_filter_indices, right_confs, sample_mode, top_k_ratio)
        right_index_lookup = {int(frame_idx): idx for idx, frame_idx in enumerate(right_confs_filter_indices)}
        right_selected_scores = np.asarray([right_confs[right_index_lookup[int(frame_idx)]] for frame_idx in right_sampled_indices], dtype=np.float32)
        right_rgb_meta = build_rgb_meta(right_selected_scores, right_sampled_indices, right_confs_filter_indices, len(confs))
        
        right_skeletons = skeletons[right_sampled_indices,0,112:133]
        
    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        _fill_empty_hand_dict(support_rgb_dict, image_size)
        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = [int(i) for i in np.asarray(tmp[sampled_indices]).tolist()]

    try:
        vr = VideoReader(full_path, num_threads=1, ctx=cpu(0))
        vr.seek(0)
        first_img = vr[sampled_indices_real[0]].asnumpy()
    except Exception:
        # Video corrupted or unsupported codec - return empty hand data (pose-only fallback)
        _fill_empty_hand_dict(support_rgb_dict, 112)
        return support_rgb_dict

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                        right_skeletons,
                                                        first_img.shape[:2])
    
    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        _fill_empty_hand_dict(support_rgb_dict, image_size)
        return support_rgb_dict

    factor = image_size / box_hw
    
    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)
            
        left_skeletons_norm = left_skeletons * first_img.shape[:2][::-1] - left_new_box[:, None, [0,1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0,1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)
        
        right_skeletons_norm = right_skeletons * first_img.shape[:2][::-1] - right_new_box[:, None, [0,1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0,1)
    left_idx = 0
    right_idx = 0

    for idx, frame_idx in enumerate(sampled_indices_real):
        try:
            img = first_img if idx == 0 else vr[frame_idx].asnumpy()
        except Exception:
            img = first_img  # fallback to first frame on decode error
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]

            img_draw = crop_with_padding(np.uint8(copy.deepcopy(img)), box)
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1
            
        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]

            img_draw = crop_with_padding(np.uint8(copy.deepcopy(img)), box)
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1

    del vr
    del first_img
   
    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])
        
    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)
    support_rgb_dict['left_rgb_meta'] = left_rgb_meta

    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)
    support_rgb_dict['right_rgb_meta'] = right_rgb_meta

    return support_rgb_dict


# use split rgb video for save time
def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    
    vr.seek(0)
    batch_image = [vr[int(frame_idx)].asnumpy() for frame_idx in np.asarray(tmp).tolist()]
    del vr

    return batch_image

# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch = [],[],[],[],[]
        
        for name_sample, pose_sample, text, gloss, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
                
        if self.rgb_support:
            support_rgb_dicts = {key:[] for key in batch[0][-1].keys()}
            for _, _, _, _, support_rgb_dict in batch:
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])
            
            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                meta_key = f'{part}_rgb_meta'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], 0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], 0)
                img_batch = torch.cat(support_rgb_dicts[rgb_key], 0)
                meta_batch = torch.cat(support_rgb_dicts[meta_key], 0)
                
                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[meta_key] = meta_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]

        elif self.args.dataset == "CE_CSL":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
            
        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "How2Sign" in self.args.dataset:
            if phase == 'dev':
                raise NotImplementedError("How2Sign dev set is not supported")
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "OpenASL" in self.args.dataset:
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        else:
            raise NotImplementedError

        self.list = list(self.raw_data.keys())

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])

        return name_sample,pose_sample,text, gloss, support_rgb_dict
    
    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = os.path.join(self.rgb_dir, path)
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                sample_mode=getattr(self.args, "rgb_frame_sampling", "confidence_topk"),
                top_k_ratio=getattr(self.args, "top_k_ratio", 0.1),
            )
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
       
        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
      
        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)
        
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            text = sample['text']
            name_sample = sample['video']
           
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
    
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break
           
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return name_sample, pose_sample, text, _, support_rgb_dict
    
    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        full_path = os.path.join(self.rgb_dir, rgb_name)
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_kp(skeletons, confs)
        
        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                sample_mode=getattr(self.args, "rgb_frame_sampling", "confidence_topk"),
                top_k_ratio=getattr(self.args, "top_k_ratio", 0.1),
            )

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_online(Base_Dataset):
    def __init__(self, args):
        super(S2T_Dataset_online, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length

        # place holder
        self.rgb_data = None
        self.pose_data = None

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = ''
        gloss = ''
        name_sample = 'online_data'

        pose_sample, support_rgb_dict = self.load_pose()

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self):
        pose = self.pose_data

        duration = len(pose['scores'])
        start = 0

        if duration > self.max_length:
            tmp = build_uniform_indices(duration, self.max_length).tolist()
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp) + start

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = self.rgb_data
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                sample_mode=getattr(self.args, "rgb_frame_sampling", "confidence_topk"),
                top_k_ratio=getattr(self.args, "top_k_ratio", 0.1),
            )

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'
