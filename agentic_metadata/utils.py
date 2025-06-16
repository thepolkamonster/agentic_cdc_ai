import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def get_prompt_message(role: str, df = None, insight = None, path="agentic_metadata\prompts\prompts.json"):
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    role_prompts = prompts[role]
    # Replace placeholders if provided
    system_msg = role_prompts["system"]
    user_msg = role_prompts["user"]
    if df is not None and insight is not None:
        df = df.to_markdown()
        user_msg = user_msg.format(df=df, insight=insight)
    elif df is not None:
        df = df.to_markdown()
        user_msg = user_msg.format(df=df)

    message = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    return message


def get_clean_csv(cav_path):
    df = pd.read_csv("agentic_metadata\metadata.csv").head(5)
    df.dropna(axis = 1, inplace = True)
    df = df.round(2)

    return df

def load_generator_model_and_tokenizer(MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True
    ).to("cuda")
    #print(model)

    return model, tokenizer

def load_evaluator_model_and_tokenizer(MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True,
        load_in_4bit=True
    ).to("cuda")
    # print(model)

    return model, tokenizer

# if __name__ == "__main__":
#     get_prompt_messages("epidemiologist")
