import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from prompts import Prompt_Loader
from tqdm import tqdm
import torch.distributed as dist
import torch
import os
import argparse
import socket
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Program Generator with Distributed Support')
    parser.add_argument('--data_path', type=str, required=True,
                      help='数据集根目录路径')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['HOVER', 'FEVEROUS'],
                      help='数据集名称')
    parser.add_argument('--num_programs_per_example', type=int, default=1,
                      help='每个样本生成的程序数量')
    parser.add_argument('--num_eval_samples', type=int, default=-1,
                      help='评估样本数量，-1表示使用全部样本')
    parser.add_argument('--save_path', type=str, required=True,
                      help='结果保存路径')
    parser.add_argument('--model_name', type=str, default="codellama/CodeLlama-13b-hf",
                      help='使用的模型名称')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                      help='主节点地址')
    parser.add_argument('--master_port', type=str, default='29500',
                      help='主节点端口')
    parser.add_argument('--timeout', type=int, default=1800,
                      help='分布式训练超时时间(秒)')
    parser.add_argument('--local_rank', type=int, default=0,
                      help='本地进程序号')
    parser.add_argument('--use_4bit', action='store_true',
                      help='是否使用4-bit量化')
    parser.add_argument('--use_8bit', action='store_true',
                      help='是否使用8-bit量化')
    return parser.parse_args()

# 加载Prompt构造器
prompt_loader = Prompt_Loader()

def find_free_port():
    """查找可用的端口号"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_distributed(args):
    """设置分布式训练环境"""
    try:
        # 获取本地进程的rank
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = args.local_rank

        # 获取全局rank和world_size
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            rank = 0
            world_size = 1

        # 设置环境变量
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        # 设置当前设备
        if torch.cuda.is_available():
            # 直接使用local_rank作为GPU索引
            torch.cuda.set_device(local_rank)
            print(f"Process {rank} using GPU: {local_rank}")

        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{args.master_addr}:{args.master_port}',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=args.timeout)
        )
        
        return rank, world_size, local_rank
    except Exception as e:
        print(f"初始化分布式环境时出错: {str(e)}")
        raise e

def load_model(model_name, local_rank, args):
    """加载模型和tokenizer"""
    try:
        print(f"Loading model and tokenizer on GPU {local_rank} (local_rank {local_rank})...")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=f"cuda:{local_rank}",  # 直接指定使用对应的GPU
            torch_dtype=torch.float16,    # 使用半精度
            low_cpu_mem_usage=True        # 降低CPU内存使用
        )
        
        # 使用DDP包装模型
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            print(f"Model loaded successfully on GPU {local_rank}")
            
            # 打印当前GPU的内存使用情况
            allocated = torch.cuda.memory_allocated(local_rank) / (1024 * 1024 * 1024)
            reserved = torch.cuda.memory_reserved(local_rank) / (1024 * 1024 * 1024)
            print(f"GPU {local_rank} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return tokenizer, model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise e

def extract_first_program(data):
    """提取第一个 'New Question:' 后面紧跟的 'def program():' 块内容。"""
    lines = data.splitlines()
    result = []
    start_extracting = False

    for line in lines:
        if "New Question:" in line:
            start_extracting = True
            result = []
        if start_extracting:
            result.append(line)
            if line.strip() == "" and len(result) > 1:
                break

    return "\n".join(line for line in result if line.strip())

def process_file(args, rank, world_size, local_rank):
    """处理文件的分布式版本"""
    try:
        results = []
        
        # 构建输入文件路径
        input_file = os.path.join(args.data_path, args.dataset_name, "claims",
                                "gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl")
        
        # 确保保存路径存在
        if rank == 0:
            os.makedirs(args.save_path, exist_ok=True)
        output_file = os.path.join(args.save_path, "output_results.json")
        
        # 读取输入数据
        with open(input_file, "r", encoding="utf-8") as infile:
            data = [json.loads(line) for line in infile]
        
        # 如果指定了评估样本数量
        if args.num_eval_samples > 0:
            data = data[:args.num_eval_samples]
        
        # 根据rank分配数据
        chunk_size = len(data) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank != world_size - 1 else len(data)
        local_data = data[start_idx:end_idx]
        
        # 释放原始数据的内存
        del data
        torch.cuda.empty_cache()

        # 加载模型
        tokenizer, model = load_model(args.model_name, local_rank, args)

        # 处理分配的数据
        for idx, item in enumerate(tqdm(local_data, desc=f"Processing on rank {rank}")):
            try:
                global_idx = start_idx + idx
                claim = item.get("mutated", "")
                gold = item.get("gold", "")

                if not claim:
                    print(f"Warning: Empty claim detected at index {global_idx}, skipping...")
                    continue

                # 构造Prompt
                prompt = prompt_loader.prompt_construction(claim, args.dataset_name)

                # 使用上下文管理器来自动清理内存
                with torch.cuda.amp.autocast(), torch.no_grad():
                    # Tokenize输入
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
                    inputs = {k: v.to(f"cuda:{local_rank}") for k, v in inputs.items()}

                    # 生成多个程序
                    predicted_programs = []
                    for _ in range(args.num_programs_per_example):
                        outputs = model.module.generate(
                            **inputs,
                            max_length=1500,
                            temperature=0.2,
                            top_p=0.9,
                            do_sample=True,
                            num_return_sequences=1
                        )
                        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        program = extract_first_program(generated_code)
                        if program:
                            predicted_programs.append(program)
                        
                        # 清理生成的输出
                        del outputs
                        torch.cuda.empty_cache()

                    # 保存结果
                    results.append({
                        "idx": global_idx,
                        "claim": claim,
                        "gold": gold,
                        "predicted_programs": predicted_programs
                    })

                    # 清理本次迭代的内存
                    del inputs
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing item {global_idx}: {e}")
                continue

        # 收集所有进程的结果
        if world_size > 1:
            all_results = [None] * world_size
            dist.all_gather_object(all_results, results)
            if rank == 0:
                combined_results = []
                for r in all_results:
                    combined_results.extend(r)
                results = sorted(combined_results, key=lambda x: x["idx"])
        
        # 只在主进程中保存结果
        if rank == 0:
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(results, outfile, ensure_ascii=False, indent=4)
                
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        raise e

def main():
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 设置分布式环境
        rank, world_size, local_rank = setup_distributed(args)
        
        # 设置当前进程使用的GPU
        device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        
        # 设置随机种子
        torch.manual_seed(42 + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + rank)
        
        try:
            process_file(args, rank, world_size, local_rank)
            if rank == 0:
                print(f"Results saved to {os.path.join(args.save_path, 'output_results.json')}")
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            raise e
        finally:
            # 清理分布式环境
            if dist.is_initialized():
                dist.destroy_process_group()
                
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
