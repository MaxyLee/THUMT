ROOT=/data1/private/mxy/projects/mmt

PRETRAIN_DATA=wmt18+cc-en-de-big

EXP=$ROOT/exp/THUMT
CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-77.pt
# CKPT=$EXP/pretrain/cc-en-de/eval/model-48.pt

VOCAB_DIR=$ROOT/data/wmt18/de-en
DATA_DIR=$ROOT/data/thumt/en-de
# DATA_DIR=$ROOT/data/mmt/conceptual_captions

# pretrain
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --input $DATA_DIR/wmt18+cc_train.32k.en.shuf $DATA_DIR/wmt18+cc_train.32k.de.shuf \
#   --output $EXP/pretrain/wmt18+cc-en-de-big \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/wmt18+cc_dev.32k.en \
#   --references $DATA_DIR/wmt18+cc_dev.de \
#   --model transformer \
#   --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2 \
#   --hparam_set big

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw
# finetune
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --input $DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
#   --output $EXP/finetune/$PRETRAIN_DATA \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/val.32k.en \
#   --references $DATA_DIR/val.de \
#   --checkpoint $CKPT \
#   --model transformer \
#   --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
#   --hparam_set big

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw
# prefix tuning
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
#   --input $DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
#   --output $EXP/prefix-tuning/$PRETRAIN_DATA \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/val.32k.en \
#   --references $DATA_DIR/val.de \
#   --checkpoint $CKPT \
#   --model prefix_transformer \
#   --parameters=prefix_length=64,learning_rate=7e-4,batch_size=4096,device_list=[0],update_cycle=2,eval_steps=20 \
#   --hparam_set base

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1
# visual prefix tuning
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --input $DATA_DIR/raw/train.32k.en.shuf $DATA_DIR/raw/train.32k.de.shuf \
  --img_input $DATA_DIR/image_splits $DATA_DIR/clip-ViT-B_32.pkl \
  --output $EXP/visual-prefix-tuning/${PRETRAIN_DATA}-mlp \
  --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
  --validation $DATA_DIR/raw/val.32k.en \
  --references $DATA_DIR/raw/val.de \
  --checkpoint $CKPT \
  --model visual_prefix_transformer \
  --parameters=max_length=64,mapping_type=mlp,batch_size=128,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
  --hparam_set base