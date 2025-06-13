from agentic_metadata.utils import get_prompt_message, get_clean_csv, load_model_and_tokenizer
from agentic_metadata.llm_generator import generate_insight, evaluate_insight

df = get_clean_csv("agentic_metadata\metadata.csv")
generation_message = get_prompt_message("data_scientist", df)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
generator, tokenizer = load_model_and_tokenizer(MODEL_ID)
insight = generate_insight(generation_message, df, generator, tokenizer)

print("================Evaluation=================")

evaluation_message = get_prompt_message("data_scientist_evaluator", df=df, insight=insight)
print("==========Evaluation message===========")
print(evaluation_message)
print("========================================")
evaluation = evaluate_insight(evaluation_message, df, generator, tokenizer, insight)