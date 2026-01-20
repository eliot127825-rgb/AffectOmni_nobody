import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import itertools
import pickle
from pathlib import Path

import argparse
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy
import av
import yaml
import math
import random

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None
    
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()


def emer_ov_mc(reference, hypothesis):
    list_a = reference.split(",")
    list_b = hypothesis.split(",")
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score

def reward_fn(output_ans, gt_ans, question_type):
    try:

        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)

        else:
            return 0.0
    except Exception as e:
        return 0.0

SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""

class MyDataset(Dataset):
    def __init__(self, data_path, processor):
        super(MyDataset, self).__init__()
        self.list_data_dict = []

        self.use_audio_in_video = True


        self.processor = processor

     
        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",

        }

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                #     data_root: xxxx/xx

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    if data.get("data_root", None):
                        for each in cur_data_dict:
                            if "path" in each:
                                each["path"] = os.path.join(data["data_root"], each["path"])
                            # 同时处理video字段，避免优先使用video时路径错误
                            if "video" in each and not os.path.isabs(each["video"]):
                                each["video"] = os.path.join(data["data_root"], each["video"])
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        self.mel_size = 128
        self.frames_upbound = 16


       
        self.data = self.list_data_dict

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        import traceback

        try:
            return self._get_item(i)
        except Exception as e:
            print(i)
            traceback.print_exc()


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                traceback.print_exc()
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

    def _get_item(self, index):
            data = self.data[index]
            
            if data["problem_type"] == 'multiple choice' or  data["problem_type"] == 'emer_ov_mc' :
                question = data['problem'] + " Options:\n"
                for op in data["options"]:
                    question += op + "\n"
            else:
                question = data['problem']

   
            video_path = data['video'] if "video" in data else data["path"]
            video_audio_avaliable = check_if_video_has_audio(video_path)


            # text_prompt = self.QUESTION_TEMPLATE.format(Question=question) + self.TYPE_TEMPLATE[data['problem_type']]
            text_prompt =  question + self.TYPE_TEMPLATE[data['problem_type']]
            if video_audio_avaliable:
                message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },
                        {
                            "type": "audio",
                            "audio": video_path
                        },
                        
                        {
                            "type": "text",
                            "text": f"Here is a {data['data_type']}, with the audio from the video.\n" + text_prompt
                        }
                    ]
                }]
            else:
                 message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },
                        
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]

            message.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })

            
            audios, images, videos = process_mm_info(message, use_audio_in_video=False)
            data_dict = {
                'images': images,
                'audios': audios,
                'videos': videos,
                'prompt': message,
                'solution': data["solution"],
                "problem_type": data["problem_type"],
                "raw_data": data
            }
            return data_dict
          
    def __len__(self):
        return len(self.data)
        
def collate_fn(examples):
    

    return examples

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        all_indices = list(range(total_size))
        
        interleaved_indices = [
            idx for idx in all_indices if idx % world_size == rank
        ]
    
        return interleaved_indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def save_checkpoint(rank, retained_samples, processed_indices, checkpoint_dir="checkpoints"):
    """保存checkpoint到磁盘"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"rank_{rank}_checkpoint.pkl")
    checkpoint_data = {
        'retained_samples': retained_samples,
        'processed_indices': processed_indices,
        'rank': rank
    }
    # 先保存到临时文件，再重命名（原子操作，避免保存过程中崩溃导致文件损坏）
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    os.rename(temp_path, checkpoint_path)
    print(f"[Rank {rank}] ✅ Checkpoint已保存: {len(processed_indices)}个batch已处理")

def load_checkpoint(rank, checkpoint_dir="checkpoints"):
    """从磁盘加载checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f"rank_{rank}_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"[Rank {rank}] 🔄 从checkpoint恢复: {len(checkpoint_data['processed_indices'])}个batch已完成")
            return checkpoint_data['retained_samples'], checkpoint_data['processed_indices']
        except Exception as e:
            print(f"[Rank {rank}] ⚠️  Checkpoint加载失败: {e}，从头开始")
            return [], set()
    else:
        print(f"[Rank {rank}] 📝 未找到checkpoint，从头开始")
        return [], set()

def main(args):
 

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))




    # 加载模型权重
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
    
    # Processor从基座模型加载（因为训练时只保存了模型权重）
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"
    processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL_PATH)

   
    model_name = args.model_path.split("/")[-1]
      
    # 使用指定的YAML配置文件来生成rollout数据（避免循环依赖）
    data_config = getattr(args, 'data_config', 'data_config/rollout_gen.yaml')
    dataset = MyDataset(data_config, processor)
    print(f"📂 使用数据配置: {data_config}")
    print(f"📊 数据集大小: {len(dataset)} 个样本")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # 加载checkpoint（如果存在）
    rank = torch.distributed.get_rank()
    retained_correct_samples, processed_indices = load_checkpoint(rank)
    
    # Checkpoint配置
    CHECKPOINT_INTERVAL = 20  # 每20个batch保存一次checkpoint（约40分钟）
    batch_counter = len(processed_indices)  # 已处理的batch数量
    
    for batch_idx, inputs in enumerate(tqdm(dataloader, desc=f"{rank} Processing batches", initial=batch_counter, total=len(dataloader))):
        # 跳过已处理的batch
        if batch_idx in processed_indices:
            continue

        images, videos, audios, prompts = [], [], [], []
        for each in inputs:
            prompts.append(each["prompt"])
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None
        
        text = processor.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )
        # print(text)
        model_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)

        
        model_inputs = model_inputs.to(model.device).to(model.dtype)
        NUM_SAMPLES_PER_INSTANCE = 8
        text_ids = []
        with torch.inference_mode():
            # 保持1024 tokens完整输出（Stage 2训练需要完整数据）
            # 每次只生成1个sequence，减少显存峰值占用，避免OOM
            for j in range(NUM_SAMPLES_PER_INSTANCE):
                text_ids.extend(model.generate(**model_inputs, use_audio_in_video=False, max_new_tokens=1024, num_return_sequences=1,
                do_sample=True, 
                temperature=0.9, 
                top_p=1.0,  
                ))
                # print(text_ids[0])
     

            for i, original_sample in enumerate(inputs):
                correct_predictions_count = 0
                correct_outputs_for_this_sample = [] 

             
                input_ids_length_for_this_sample = model_inputs['input_ids'][i].size(0)
                generated_ids_for_this_sample = text_ids[
                        i * NUM_SAMPLES_PER_INSTANCE : (i + 1) * NUM_SAMPLES_PER_INSTANCE
                    ]

                for j in range(NUM_SAMPLES_PER_INSTANCE):
                    try:
                        response = processor.decode(
                            generated_ids_for_this_sample[j][input_ids_length_for_this_sample:],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                    except Exception as e:
                        
                        print(generated_ids_for_this_sample)
                        print(len(text_ids))
                        print(e)
                    # print(response)
                    gt = inputs[i]["solution"]

                  
                    final_ans = extract_answer(response)# response #extract_answer(response)
                    gt_ans = extract_answer(gt)
                    problem_type =  inputs[i]["raw_data"]["problem_type"]
                    if final_ans == "":
                        final_ans = response

                    reward = reward_fn(final_ans, gt_ans, problem_type)

                    if reward>0.2:
                        correct_predictions_count += 1
                
                # 循环结束后计算准确率
                sample_accuracy = correct_predictions_count / NUM_SAMPLES_PER_INSTANCE

                # 打印样本评估结果
                print(f"[Rank {torch.distributed.get_rank()}] 样本评估: 正确数={correct_predictions_count}/{NUM_SAMPLES_PER_INSTANCE}, 准确率={sample_accuracy:.2f}")
                
                if 0 < sample_accuracy < 0.75 and correct_predictions_count > 0:
                    print(f"[Rank {torch.distributed.get_rank()}] ✅ 保留样本（准确率在0-75%之间）")
                    retained_correct_samples.append(inputs[i]["raw_data"])
                else:
                    if sample_accuracy == 0:
                        print(f"[Rank {torch.distributed.get_rank()}] ❌ 过滤：全部错误（准确率0%）")
                    elif sample_accuracy >= 0.75:
                        print(f"[Rank {rank}] ❌ 过滤：太简单（准确率{sample_accuracy*100:.0f}%）")
                    print(inputs[i]["raw_data"])
        
        # 记录已处理的batch
        processed_indices.add(batch_idx)
        batch_counter += 1
        
        # 定期保存checkpoint
        if batch_counter % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(rank, retained_correct_samples, processed_indices)

        

    # 最后保存一次checkpoint
    save_checkpoint(rank, retained_correct_samples, processed_indices)
    
    # 移除barrier：各GPU速度不均导致快的GPU等待慢的GPU超时
    # all_gather_object内部已有隐式同步，不需要额外barrier
    world_size = torch.distributed.get_world_size()
    print(f"[Rank {rank}] World size: {world_size}, 本地样本数: {len(retained_correct_samples)}")
   
    merged_sources = [None for _ in range(world_size)]

    print(f"[Rank {rank}] 开始all_gather（内含同步）...")
    torch.distributed.all_gather_object(merged_sources, retained_correct_samples)
    print(f"[Rank {rank}] all_gather完成！")


    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]


    if rank == 0:
        print(f"[Rank 0] 合并后总样本数: {len(merged_sources)}")
        print(f"[Rank 0] 保存到: data_config/{model_name}_r8.json")
        with open(f"data_config/{model_name}_r8.json", "w", encoding="utf-8") as f:
            json.dump(merged_sources, f, indent=2, ensure_ascii=False)
        print(f"[Rank 0] ✅ 文件保存成功！")
        
        # 清理所有checkpoint文件
        print(f"[Rank 0] 🧹 清理checkpoint文件...")
        import shutil
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
            print(f"[Rank 0] ✅ Checkpoint已清理")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument('--model-path', type=str, required=False, help="Path to the model")
    parser.add_argument('--data-config', type=str, default='data_config/rollout_gen.yaml', help="Path to the data config YAML file")
    args = parser.parse_args()

    main(args)

