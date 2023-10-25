# GraphPromptTuning

## Environment setup
```
conda create --name gpt python=3.9 -y
conda activate gpt
source activate gpt

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric

```

## Training
Replace path to the llm checkpoints in the [`src/model/__init__.py`](https://github.com/XiaoxinHe/GraphPromptTuning/blob/main/src/model/__init__.py), then run
```
python train.py --dataset cora --model_name graph_llm --llm_model_name 7b --gnn_model_name gat --seed 0
```