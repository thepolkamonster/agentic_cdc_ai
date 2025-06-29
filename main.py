from agentic_metadata.utils import get_prompt_message, get_clean_csv, load_generator_model_and_tokenizer, load_evaluator_model_and_tokenizer, start_git_flow, end_git_flow
from agentic_metadata.llm_generator import generate_insight, evaluate_insight

df = get_clean_csv("agentic_metadata\metadata.csv")
role = "data_scientist"
generation_message = get_prompt_message(role, df)

git_commit_message = generation_message[0]
print(git_commit_message)



suggestion = "NA"

GENERATOR_MODEL_ID = "openchat/openchat-3.5-1210"
EVALUATOR_MODEL_ID = "openchat/openchat-3.5-1210"
generator, generator_tokenizer = load_generator_model_and_tokenizer(GENERATOR_MODEL_ID)
insight = generate_insight(generation_message, df, generator, generator_tokenizer)

print("================Evaluation=================")
evaluator, evaluator_tokenizer = load_evaluator_model_and_tokenizer(EVALUATOR_MODEL_ID)
evaluation_message = get_prompt_message("data_scientist_evaluator", df=df, insight=insight)
print("==========Evaluation message===========")
# print(evaluation_message)
rating, suggestion = evaluate_insight(evaluation_message, df, evaluator, evaluator_tokenizer, insight)

counter = 2

while(rating < 9 and counter > 0):
    content = generation_message[0]["content"]
    git_commit_message = content + f"\nSuggestion: {suggestion}"
    generation_message = get_prompt_message("data_scientist", df, insight=insight, suggestion=suggestion)
    insight = generate_insight(generation_message, df, generator, generator_tokenizer)
    evaluation_message = get_prompt_message("data_scientist_evaluator", df=df, insight=insight)
    rating, suggestion = evaluate_insight(evaluation_message, df, evaluator, evaluator_tokenizer, insight)
    counter -= 1

repo, branch = start_git_flow(position = role, initial_prompt = git_commit_message)

end_git_flow(repo, branch)
