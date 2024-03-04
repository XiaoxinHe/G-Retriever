for seed in 0 1 2 3 
do
# 1) inference only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat --seed $seed
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat --seed $seed
python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat --seed $seed

# 2) frozen llm + prompt tuning
# a) query
python train.py --dataset expla_graphs --model_name pt_llm --max_txt_len 0 --seed $seed
python train.py --dataset scene_graphs_baseline --model_name pt_llm --max_txt_len 0 --seed $seed
python train.py --dataset webqsp_baseline --model_name pt_llm --max_txt_len 0 --seed $seed

# b) query + textual graph
python train.py --dataset expla_graphs --model_name pt_llm --seed $seed
python train.py --dataset scene_graphs --model_name pt_llm --seed $seed
python train.py --dataset webqsp --model_name pt_llm --seed $seed

# c) g-retriever
python train.py --dataset expla_graphs --model_name graph_llm --seed $seed
python train.py --dataset scene_graphs --model_name graph_llm --seed $seed
python train.py --dataset webqsp --model_name graph_llm --seed $seed

# 3) tuned llm
# a) finetuning with lora
python train.py --dataset expla_graphs --model_name llm --llm_frozen False --seed $seed
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False --seed $seed
python train.py --dataset webqsp_baseline --model_name llm --llm_frozen False --seed $seed

# b) g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --seed $seed
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False --seed $seed
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --seed $seed
done