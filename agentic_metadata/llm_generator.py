from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     device_map=None
# ).to("cuda")

# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=0
# )

# def generate_insight(prompt: str) -> str:
#     response = generator(
#         prompt,
#         do_sample=True,
#         max_new_tokens=200,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         eos_token_id=tokenizer.eos_token_id
#     )

#     return response[0]["generated_text"];


# if __name__ == "__main__":
#     test_prompt = "You are a data scientist. Analyze this metadata: age, region, income. What trends can you see? Give short answers. Be precise and analytical."
#     print(generate_insight(test_prompt))

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16
# ).to("cuda")

# def generate_insight(prompt: str, max_tokens: int = 512) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id  # Prevents warnings
#     )

#     generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return generated[len(prompt):].strip()  # remove original prompt


# if __name__ == "__main__":
#     prompt = (
#         "You are a data scientist. Analyze this metadata: age, region, income. Give 3 short points and stop generating"
#     )
#     print(generate_insight(prompt))


def generate_insight(message, df, model ,tokenizer ,max_tokens: int = 512):
    chat_prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = decoded[len(chat_prompt):].strip()

    print(result)
    return result;

def evaluate_insight(message, df, model ,tokenizer ,insight, max_tokens: int = 512):
    chat_prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(chat_prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    # print("==================outputs==================")
    # print(outputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("==================decoded==================")
    # print(decoded)
    result = decoded[len(chat_prompt):].strip()

    print(result)
    return result;



