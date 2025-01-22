import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import Prompt_Loader
from tqdm import tqdm
import torch.distributed as dist
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP

# 加载Prompt构造器
prompt_loader = Prompt_Loader()

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        return rank, world_size
    return 0, 1

def load_model(model_name, rank):
    """加载模型和tokenizer"""
    print(f"Loading model and tokenizer on rank {rank}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 根据rank选择不同的GPU
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": rank} if torch.cuda.is_available() else "auto"
    )
    
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    
    return tokenizer, model

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

def process_file(input_file, output_file, rank, world_size, dataset_name="HOVER"):
    """处理文件的分布式版本"""
    results = []
    
    # 读取输入数据
    with open(input_file, "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile]
    
    # 根据rank分配数据
    chunk_size = len(data) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else len(data)
    local_data = data[start_idx:end_idx]

    # 加载模型
    model_name = "codellama/CodeLlama-13b-hf"
    tokenizer, model = load_model(model_name, rank)

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
            prompt = prompt_loader.prompt_construction(claim, dataset_name)

            # Tokenize输入
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            if torch.cuda.is_available():
                inputs = {k: v.cuda(rank) for k, v in inputs.items()}

            # 模型生成
            outputs = model.module.generate(
                **inputs,
                max_length=1500,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )

            # 解码生成结果
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = extract_first_program(generated_code)

            # 保存结果（使用新的数据格式）
            results.append({
                "idx": global_idx,
                "claim": claim,
                "gold": gold,
                "predicted_programs": [generated_code] if generated_code else []
            })

        except Exception as e:
            print(f"Error processing item {global_idx}: {e}")

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

def main():
    input_file = "./datasets/HOVER/claims/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl"
    output_file = "./results/programs/output_results.json"
    
    # 设置分布式环境
    rank, world_size = setup_distributed()
    
    try:
        process_file(input_file, output_file, rank, world_size, dataset_name="HOVER")
        if rank == 0:
            print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Unexpected error on rank {rank}: {e}")
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
