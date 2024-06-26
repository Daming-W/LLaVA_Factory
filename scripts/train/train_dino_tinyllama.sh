DATA_PATH=/root/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727/blip_laion_cc_sbu_558k.json #pretrain annotation file path
# FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/root/llava_data #pretrain image dir
# FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset #finetune image dir

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0 # llm path in huggingface
VT_VERSION=facebook/dinov2-giant #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=llama #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
