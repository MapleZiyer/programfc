import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import Prompt_Loader
from tqdm import tqdm

# 加载Prompt构造器
prompt_loader = Prompt_Loader()

# 模型及tokenizer加载
model_name = "codellama/CodeLlama-13b-hf"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 数据文件路径
input_file_path = "./datasets/HOVER/claims/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl"  # 替换为你的输入文件路径
output_file_path = "./results/programs/output_results.json"  # 替换为你的输出文件路径

def process_file(input_file, output_file, dataset_name="HOVER"):
    """
    从输入文件读取信息，生成prompt，调用模型，保存结果。
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param dataset_name: 数据集名称 (HOVER 或 FEVEROUS)
    """
    results = []

    # 逐行处理输入文件
    with open(input_file, "r", encoding="utf-8") as infile:
        # 使用tqdm显示进度条
        for line in tqdm(infile, desc="Processing claims"):
            try:
                # 解析JSON数据
                data = json.loads(line.strip())
                claim = data.get("mutated", "")

                # 构造Prompt
                prompt = prompt_loader.prompt_construction(claim, dataset_name)

                # Tokenize输入
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # 模型生成
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

                # 解码生成结果
                generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 保存结果
                results.append({
                    "claim": claim,
                    "generated_code": generated_code
                })

            except Exception as e:
                print(f"Error processing line: {e}")

    # 保存结果到输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

# 运行主程序
process_file(input_file_path, output_file_path, dataset_name="HOVER")
print(f"Results saved to {output_file_path}")
