from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print("tokenizer loaded")
model = LlamaForCausalLM.from_pretrained(model_name)

# Your prompt text
prompt_text = "* is a function for capitalizing letters, () is another function for swapping the position of 2 words.\n\ninput: (european maraca)\noutput: maraca european\n\ninput: (tan bottle)\noutput: bottle tan\n\ninput: (chain-link stupa)\noutput: stupa chain-link\n\ninput: (guenon oystercatcher)\noutput: oystercatcher guenon\n\ninput: (wooden nautilus)\noutput: nautilus wooden\n\ninput: (kit maypole)\noutput: maypole kit\n\ninput: (warthog newt)\noutput: newt warthog\n\ninput: (sandbar kelpie)\noutput: kelpie sandbar\n\ninput: (bubble black)\noutput: black bubble\n\ninput: (carolina screen)\noutput: screen carolina\n\ninput: (pie sports)\noutput:"
input_len = len(prompt_text)
prompt_tokens = tokenizer(prompt_text, return_tensors='pt')
prompt_length = prompt_tokens.input_ids.shape[1]

# Method 1: Using the generate function
generated_tokens_1 = model.generate(
    input_ids=prompt_tokens.input_ids,
    max_new_tokens=5,
    do_sample=False
)

# Decode and print the output
generated_text_1 = tokenizer.decode(generated_tokens_1[0], skip_special_tokens=True)
print("Method 1 Output:", generated_text_1[input_len:])

# Method 2: Manual token-by-token generation
with torch.no_grad():
    tokens = prompt_tokens.input_ids
    for _ in range(5):  # Generating up to 10 new tokens
        outputs = model(input_ids=tokens)
        logits = outputs.logits
        # Pick the most likely next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # print(next_token.shape)
        # new_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        # print(new_token.shape)
        # assert False

        # Append the generated token to the sequence
        tokens = torch.cat([tokens, next_token], dim=1)

# Decode and print the output
generated_text_2 = tokenizer.decode(tokens[0], skip_special_tokens=True)
print("Method 2 Output:", generated_text_2[input_len:])

