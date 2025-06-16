from agentic_metadata.utils import get_prompt_message, get_clean_csv, load_generator_model_and_tokenizer, load_evaluator_model_and_tokenizer
from agentic_metadata.llm_generator import generate_insight, evaluate_insight

df = get_clean_csv("agentic_metadata\metadata.csv")
generation_message = get_prompt_message("data_scientist", df)


GENERATOR_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EVALUATOR_MODEL_ID = "openchat/openchat-3.5-1210"
generator, generator_tokenizer = load_generator_model_and_tokenizer(GENERATOR_MODEL_ID)
insight = generate_insight(generation_message, df, generator, generator_tokenizer)

print("================Evaluation=================")
evaluator, evaluator_tokenizer = load_evaluator_model_and_tokenizer(EVALUATOR_MODEL_ID)
evaluation_message = get_prompt_message("data_scientist_evaluator", df=df, insight=insight)
print("==========Evaluation message===========")
# print(evaluation_message)
print("=======================================")
evaluation = evaluate_insight(evaluation_message, df, evaluator, evaluator_tokenizer, insight)