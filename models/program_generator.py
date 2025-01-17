import argparse
import os
import json
from tqdm import tqdm
import torch
from prompts import Prompt_Loader
from utils import OpenAIModel
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            attention_mask=inputs["attention_mask"]
        )
        torch.cuda.empty_cache()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts, temperature=0.7):
        return [self.generate(prompt, temperature) for prompt in prompts]

class ReasoningProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example

        self.model = CodeLlamaModel(args.model_name, args.max_new_tokens)
        self.prompt_loader = PromptLoader()
        self.result_dict = {}

    def update_results(self, idx, generated_text):
        program_list = [line.strip() for line in generated_text.split('\n')]
        self.result_dict[idx]['predicted_programs'].append(program_list)

    def load_dataset(self):
        dataset_file = os.path.join(self.data_path, self.dataset_name, 'claims',
                                    'gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl')
        with open(dataset_file, 'r') as f:
            raw_dataset = [json.loads(line.strip()) for line in f if line.strip()]

        if self.args.num_eval_samples > 0:
            raw_dataset = raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name}.")
        return raw_dataset

    def batch_generate_programs(self, batch_size=10):
        os.makedirs(self.save_path, exist_ok=True)

        raw_dataset = self.load_dataset()
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]

        self.result_dict = {
            idx: {
                'idx': idx,
                'claim': sample['mutated'],
                'gold': sample['original'],
                'predicted_programs': []
            } for idx, sample in enumerate(raw_dataset)
        }

        temperature = 0.1 if self.num_programs_per_example == 1 else 0.7

        for iteration in range(self.num_programs_per_example):
            print(f"Generating programs for iteration {iteration + 1}...")
            for chunk in tqdm(dataset_chunks):
                full_prompts = [self.prompt_loader.construct_prompt(example['mutated'], self.dataset_name) for example in chunk]
                try:
                    batch_outputs = self.model.batch_generate(full_prompts, temperature)
                    for example, output in zip(chunk, batch_outputs):
                        self.update_results(example['idx'], output)
                except Exception as e:
                    print(f"Error in generating reasoning programs: {e}")
                torch.cuda.empty_cache()

        sorted_outputs = sorted(self.result_dict.values(), key=lambda x: x['idx'])
        output_file = os.path.join(self.save_path, f'{self.dataset_name}_N={self.num_programs_per_example}_{self.args.model_name}_programs.json')
        with open(output_file, 'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)
        print(f"Saved generated programs to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='HOVER', type=str)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default='./results/programs', type=str)
    parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-13b-hf')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generator = ReasoningProgramGenerator(args)
    generator.batch_generate_programs()
