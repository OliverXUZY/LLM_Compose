import torch
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM


@dataclass
class EvalResult:
    prompt: str
    solution: str
    answer: str
    accuracy: bool


class Evaluator:
    def __init__(self, pretrained, device = "cuda") -> None:
        '''
        pretrained (str):
            The HuggingFace Hub model ID name or the path to a pre-trained
            model to load. This is effectively the `pretrained_model_name_or_path`
            argument of `from_pretrained` in the HuggingFace `transformers` API.
        '''
        self.device = device
        self.model_id = pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained,torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        self.model.eval()

    def get_model_id(self):
        return self.model_id

    def eval(self, prompt, answer, seq_len: int = 2) -> EvalResult:
        prompt = prompt.strip()
        solution = self.prompt_to_solution(prompt, seq_len = seq_len)
        result = EvalResult(prompt, solution, answer, answer in solution)
        return result
    

    def prompt_to_solution(self, prompt, seq_len: int = 2):
        # Encode the input prompt
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        solution = ""
        # print("pro: ", prompt)
        prompt_len = len(prompt)

        # print("tokens: ", tokens)

        for _ in range(seq_len):
            # print("zhuoyan 1")
            # Directly use the model's forward pass
            # print(tokens.shape)
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, return_dict=True)
            # print("zhuoyan 2")
            logits = outputs.logits

            # Select the next token with the highest probability
            new_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

            # Append the generated token to the tokens tensor
            tokens = torch.cat((tokens, new_token), dim=1)

            # print(new_token.shape)
            # print(new_token.squeeze().tolist())

            # Decode and append the generated token to the solution string
            new_word = self.tokenizer.decode(new_token.squeeze().tolist())
            if "\n" in new_word:
                break
            # print(solution)

        solution += self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        solution = solution[prompt_len:]

            

        # print("solu: ", solution)
        return solution
    
    
    
