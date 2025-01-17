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

def extract_first_program(data):
    """
    提取第一个 'New Question:' 后面紧跟的 'def program():' 块内容。
    """
    lines = data.splitlines()
    result = []
    start_extracting = False

    for line in lines:
        if "New Question:" in line:
            start_extracting = True  # 开始提取
            result = []  # 清空，确保只保留第一个块
        if start_extracting:
            result.append(line)
            # 检查是否到了块结束的地方
            if line.strip() == "" and len(result) > 1:  # 空行标志结束
                break

    # 只返回内容部分，去掉多余的空行
    return "\n".join(line for line in result if line.strip())


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
        lines = infile.readlines()

        # 使用tqdm显示进度条
        for line in tqdm(lines, desc="Processing claims"):
            try:
                # 解析JSON数据
                data = json.loads(line)
                claim = data.get("mutated", "")

                # 检查claim是否为空
                if not claim:
                    print("Warning: Empty claim detected, skipping...")
                    continue

                # 构造Prompt
                prompt = prompt_loader.prompt_construction(claim, dataset_name)
                print(f"Model loaded on device: {model.device}")

                # Tokenize输入
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)

                print("Input:")
                print(prompt)

                # 模型生成
                outputs = model.generate(
                    **inputs,
                    max_length=1500,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1
                )

                # 解码生成结果
                generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_code = extract_latest_program(generated_code)
                print("Output:")
                print(generated_code)

                # 保存结果
                results.append({
                    "claim": claim,
                    "generated_code": generated_code
                })

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}, skipping line...")
            except Exception as e:
                print(f"Error processing line: {e}")

    # 保存结果到输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

# 主程序入口
if __name__ == "__main__":
    try:
        process_file(input_file_path, output_file_path, dataset_name="HOVER")
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")
