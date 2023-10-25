dataset=cora
# InferenceLLM
# python inference.py --model_name InferenceLLM --dataset $dataset --seed 0
# python inference.py --model_name InferenceLLM --dataset $dataset --seed 1
# python inference.py --model_name InferenceLLM --dataset $dataset --seed 2
# python inference.py --model_name InferenceLLM --dataset $dataset --seed 3

# GraphLLM
# python train.py --model_name GraphLLM --dataset $dataset --seed 0
# python train.py --model_name GraphLLM --dataset $dataset --seed 1
# python train.py --model_name GraphLLM --dataset $dataset --seed 2
# python train.py --model_name GraphLLM --dataset $dataset --seed 3

# Prompt Tuning
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 0 
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 1
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 2
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 3

# lora 
python train.py --model_name llm --dataset $dataset  --seed 0
python train.py --model_name llm --dataset $dataset  --seed 1
python train.py --model_name llm --dataset $dataset  --seed 2
python train.py --model_name llm --dataset $dataset  --seed 3

# dataset=pubmed
# # InferenceLLM
# python inference.py --model_name InferenceLLM --dataset $dataset  --seed 0
# python inference.py --model_name InferenceLLM --dataset $dataset  --seed 1
# python inference.py --model_name InferenceLLM --dataset $dataset  --seed 2
# python inference.py --model_name InferenceLLM --dataset $dataset  --seed 3

# # GraphLLM
# python train.py --model_name GraphLLM --dataset $dataset  --seed 0
# python train.py --model_name GraphLLM --dataset $dataset  --seed 1
# python train.py --model_name GraphLLM --dataset $dataset  --seed 2
# python train.py --model_name GraphLLM --dataset $dataset  --seed 3

# # Prompt Tuning
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 0 
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 1
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 2
# python train.py --model_name pt_llm --llm_prompt_type text --llm_num_virtual_tokens 10 --dataset $dataset  --seed 3