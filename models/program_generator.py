import argparse
import os
import json
from tqdm import tqdm

from prompts import Prompt_Loader
from utils import OpenAIModel
from transformers import AutoTokenizer, AutoModelForCausalLM

i = 0

class CodeLlamaModel:
    def __init__(self, model_name, max_new_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts, temperature=0.7):
        # 批量生成推理程序
        return [self.generate(prompt, temperature) for prompt in prompts]

class Reasoning_Program_Generator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example

        self.model = CodeLlamaModel(args.model_name, args.max_new_tokens)
        self.prompt_loader = Prompt_Loader()

    def update_results(self, sample, generated_text):
        program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]
        global i
        self.result_dict[i]['predicted_programs'].append(program_list)
        i += 1

    def batch_generate_programs(self, batch_size = 10):
        # create output_dir
        self.result_dict = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataset
        with open(os.path.join(self.data_path, self.dataset_name, 'claims',
                               'gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl'), 'r') as f:
            raw_dataset = [json.loads(line.strip()) for line in f if line.strip()]

        raw_dataset = raw_dataset if self.args.num_eval_samples < 0 else raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} dev set.")

        # generate programs
        temperature = 1e-5 if self.num_programs_per_example == 1 else 0.7
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]

        # initialize empty results
        result_dict = {}
        for idx, sample in enumerate(raw_dataset):
            result = {'idx': idx,
                        'claim': sample['mutated'],
                        'gold': sample['original'],
                        'predicted_programs': []}
            result_dict[idx] = result
        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            print(f"Generating programs for iteration {iteration + 1}...")
            # for each chunk
            for chunk in tqdm(dataset_chunks):
                # create prompt
                full_prompts = [self.prompt_loader.prompt_construction(example['mutated'], self.dataset_name) for example in chunk]
                try:
                    batch_outputs = self.model.batch_generate(full_prompts, temperature)
                    for sample, output in zip(chunk, batch_outputs):
                        self.update_results(sample['idx'], sample, output)
                except Exception as e:
                    print('Error in generating reasoning programs:', e)

        print(f"Generated {len(result_dict)} examples.")
        # create outputs
        for key in result_dict:
            outputs.append(result_dict[key])
        sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

        # save outputs
        with open(os.path.join(self.save_path, f'{self.dataset_name}_N={self.num_programs_per_example}_{self.model_name}_programs.json'), 'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='HOVER', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default = './results/programs', type=str)
    parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-13b-hf')
    parser.add_argument('--stop_words', type=str, default='# The claim is')
    parser.add_argument('--max_new_tokens', type=int, default=2300)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generator = Reasoning_Program_Generator(args)
    generator.batch_generate_programs()