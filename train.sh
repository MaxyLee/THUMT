ROOT=/data1/private/mxy/projects/mmt

EXP=$ROOT/exp/THUMT
CKPT=$EXP/pretrain/cc-en-de/eval/model-48.pt

VOCAB_DIR=$ROOT/data/wmt18/de-en
DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw
# DATA_DIR=$ROOT/data/thumt/en-de
# DATA_DIR=$ROOT/data/mmt/conceptual_captions

# finetune
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --input $DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
#   --output $EXP/finetune/cc-en-de \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/val.32k.en \
#   --references $DATA_DIR/val.de \
#   --checkpoint $CKPT \
#   --model transformer \
#   --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
#   --hparam_set base
  
# prefix tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --input $DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
  --output $EXP/prefix-tuning/cc-en-de \
  --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
  --validation $DATA_DIR/val.32k.en \
  --references $DATA_DIR/val.de \
  --checkpoint $CKPT \
  --model prefix_transformer \
  --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
  --hparam_set base