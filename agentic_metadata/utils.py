import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from git import Repo, GitCommandError
from datetime import datetime

LOG_FILE = "agentic_git_flow_log.txt"
DEFAULT_REMOTE = "origin"
DEFAULT_MAIN_BRANCH = "master"

# def start_git_flow(repo_path=".", branch_prefix="insight"):
#     repo = Repo(repo_path, search_parent_directories=True)
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     branch_name = f"{branch_prefix}-{timestamp}"
#     repo.git.checkout('-b', branch_name)
#     return repo, branch_name


# def end_git_flow(repo, branch_name, commit_message="Auto commit from agentic system"):
#     repo.git.add(A=True)
#     repo.index.commit(commit_message)

#     # Checkout main and merge
#     repo.git.checkout("master")
#     repo.git.merge(branch_name)

#     # Optionally delete the branch
#     repo.git.branch('-d', branch_name)


def start_git_flow(repo_path=".", branch_prefix="query", position="Data Scientist", initial_prompt="N/A"):
    repo = Repo(repo_path, search_parent_directories=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"{branch_prefix}-{timestamp}"
    
    # Create and checkout new branch
    repo.git.checkout('-b', branch_name)

    # Push branch to remote
    try:
        repo.git.push('--set-upstream', DEFAULT_REMOTE, branch_name)
    except GitCommandError as e:
        print(f"[WARN] Could not push to remote: {e}")

    # Log the session start
    with open(LOG_FILE, "a") as f:
        f.write(f"\n---\n[{timestamp}] Position: {position}\nInitial Prompt: {initial_prompt}\nBranch: {branch_name}\n")

    return repo, branch_name

def end_git_flow(repo, branch_name, commit_message="Auto commit from agentic system"):
    # Stage and commit all changes
    
    repo.git.add(A=True)
    repo.index.commit(commit_message)

    # Push commit
    try:
        repo.git.push()
    except GitCommandError as e:
        print(f"[WARN] Could not push commit: {e}")

    # Merge into main and push
    repo.git.checkout(DEFAULT_MAIN_BRANCH)
    repo.git.merge(branch_name)
    try:
        repo.git.push()
    except GitCommandError as e:
        print(f"[WARN] Could not push {DEFAULT_MAIN_BRANCH}: {e}")

    # Delete local and remote branch
    repo.git.branch('-d', branch_name)
    try:
        repo.git.push(DEFAULT_REMOTE, '--delete', branch_name)
    except GitCommandError:
        pass  # remote branch deletion is optional


def get_prompt_message(role: str, df = None, insight = None,suggestion=None,  path="agentic_metadata\prompts\prompts.json"):
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
    
    if suggestion is not None:
        user_msg += f"\nSuggested improvement: Please focus and improve on this on this: {suggestion}"

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
        trust_remote_code=True,
        load_in_4bit=True
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
