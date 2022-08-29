ROOT=/data1/private/mxy/projects/mmt

MODEL_SIZE=big
VOCAB=32k

if [ "$VOCAB" = "32k" ]; then
  VOCAB_DIR=$ROOT/data/wmt18/de-en
else
  VOCAB_DIR=$ROOT/data/mmt/conceptual_captions
fi

if [ "$MODEL_SIZE" = "big" ]; then
  # PRETRAIN_DATA=wmt18-en-de-big
  PRETRAIN_DATA=wmt18+cc-en-de-big
  # PRETRAIN_DATA=cc-en-de-big
else
  PRETRAIN_DATA=cc-en-de-base
fi

EXP=$ROOT/exp/THUMT
if [ "$MODEL_SIZE" = "big" ]; then
# CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-48.pt
  CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-77.pt
  # CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-68.pt
else
  CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-98.pt
fi

# pretrain
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --input $DATA_DIR/corpus.tc.32k.en.shuf $DATA_DIR/corpus.tc.32k.de.shuf \
#   --output $EXP/pretrain/wmt18-en-de-big \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/newstest2016.tc.32k.en \
#   --references $DATA_DIR/newstest2016.tc.de \
#   --model transformer \
#   --parameters=train_steps=500000,warmup_steps=20000,batch_size=4096,device_list=[0,1,2,3],update_cycle=2 \
#   --hparam_set big

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1
# finetune
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --input $DATA_DIR/raw/train.32k.en.shuf $DATA_DIR/raw/train.32k.de.shuf \
#   --img_input $DATA_DIR/image_splits $CLIP_FEATURES \
#   --output $EXP/finetune/$PRETRAIN_DATA-m30ktest \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/raw/val.32k.en \
#   --references $DATA_DIR/raw/val.de \
#   --checkpoint $CKPT \
#   --model transformer \
#   --parameters=max_length=64,batch_size=128,device_list=[0],update_cycle=8,eval_steps=20 \
#   --hparam_set big

MASKC_DATA_DIR=$ROOT/code/fairseq_mmt/data/multi30k-en-de.maskc
## on masked data
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
#   --input $MASKC_DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
#   --output $EXP/maskc/finetune/$PRETRAIN_DATA \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $MASKC_DATA_DIR/valid.32k.en \
#   --references $MASKC_DATA_DIR/valid.de \
#   --checkpoint $CKPT \
#   --model transformer \
#   --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
#   --hparam_set big

# prefix tuning
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --input $DATA_DIR/train.32k.en.shuf $DATA_DIR/train.32k.de.shuf \
#   --output $EXP/prefix-tuning/$PRETRAIN_DATA \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --validation $DATA_DIR/val.32k.en \
#   --references $DATA_DIR/val.de \
#   --checkpoint $CKPT \
#   --model prefix_transformer \
#   --parameters=prefix_length=64,learning_rate=7e-4,batch_size=4096,device_list=[0,1,2,3],update_cycle=2,eval_steps=20 \
#   --hparam_set big

IMG_PATH=/data2/share/data/flickr30k-entities/flickr30k-images
# CLIP_FEATURES=$DATA_DIR/clip-ViT-B_32.pkl
CLIP_FEATURES=$DATA_DIR/vit-all.pkl

# visual prefix tuning
# CUDA_VISIBLE_DEVICES=7 python train.py \
#   --input $DATA_DIR/raw/train.$VOCAB.en.shuf $DATA_DIR/raw/train.$VOCAB.de.shuf \
#   --img_input $DATA_DIR/image_splits $CLIP_FEATURES \
#   --output $EXP/visual-prefix-tuning_v6-2/${PRETRAIN_DATA}-3 \
#   --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
#   --validation $DATA_DIR/raw/val.$VOCAB.en \
#   --references $DATA_DIR/raw/val.de \
#   --checkpoint $CKPT \
#   --model visual_prefix_transformer_v6 \
#   --parameters=prefix_only=True,visual_prefix_length=30,learning_rate=7e-4,max_length=64,batch_size=128,device_list=[0],update_cycle=16,eval_steps=20 \
#   --hparam_set $MODEL_SIZE

## few-shot
DEVICES=2
FS_NUM=200
BATCH_SIZE=100
TRAIN_STEPS=500
WARMUP_STEPS=50
EVAL_STEPS=2
FS_IDS=(
  '4'
)

for FS_ID in ${FS_IDS[@]}; do
  FS_NAME=train-fs${FS_NUM}-${FS_ID}
  OUTPATH=$EXP/visual-prefix-tuning_fs-3/$FS_NAME/${PRETRAIN_DATA}

  echo training on $FS_NAME

  # CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
  #   --input $DATA_DIR/raw/$FS_NAME.$VOCAB.en $DATA_DIR/raw/$FS_NAME.$VOCAB.de \
  #   --fewshot_name $FS_NAME \
  #   --img_input $DATA_DIR/image_splits $CLIP_FEATURES \
  #   --output $OUTPATH \
  #   --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
  #   --validation $DATA_DIR/raw/val.$VOCAB.en \
  #   --references $DATA_DIR/raw/val.de \
  #   --checkpoint $CKPT \
  #   --model visual_prefix_transformer_v6 \
  #   --parameters=prefix_only=True,learning_rate=5e-4,learning_rate_schedule=constant,max_length=64,batch_size=$BATCH_SIZE,device_list=[0],eval_steps=$EVAL_STEPS,train_steps=$TRAIN_STEPS,warmup_steps=$WARMUP_STEPS \
  #   --hparam_set $MODEL_SIZE

  # # eval
  PREFIX_PATH=$OUTPATH/eval

  # # average checkpoint
  # python thumt/scripts/average_checkpoints.py --path $PREFIX_PATH --output $PREFIX_PATH/average

  PREFIX_CKPT=$PREFIX_PATH/average/average-0.pt

  NAME=vpt_fs

  TESTNAMES="test_2016_flickr test_2017_flickr test_2018_flickr test_2017_mscoco"

  for TESTNAME in $TESTNAMES; do
    INPUT=$DATA_DIR/raw/$TESTNAME.$VOCAB.en
    OUTPUT=$DATA_DIR/raw/$TESTNAME-$NAME.trans

    if [ "$TESTNAME" = "test_2016_flickr" ]; then
      VIT_FEATURES=$DATA_DIR/vit-all.pkl
      IMG_PATH=$F30K_IMG_PATH
    else
      VIT_FEATURES=$DATA_DIR/vit-${TESTNAME}.pkl
      if [ "$TESTNAME" = "test_2017_mscoco" ]; then
        IMG_PATH=$COCO_IMG_PATH
      elif [ "$TESTNAME" = "test_2017_flickr" ]; then
        IMG_PATH=$M30K_IMG_PATH/test2017
      else
        IMG_PATH=$M30K_IMG_PATH/test2018
      fi
    fi
    # visual
    CUDA_VISIBLE_DEVICES=$DEVICES python translate.py \
      --models visual_prefix_transformer_v6 \
      --input $INPUT \
      --img_input $DATA_DIR/image_splits $VIT_FEATURES \
      --output $OUTPUT \
      --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
      --checkpoints $CKPT \
      --prefix=$PREFIX_CKPT \
      --parameters=prefix_only=True,device_list=[0],decode_alpha=1.2,beam_size=4,max_length=64 \
      --hparam_set $MODEL_SIZE

    sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

    perl multi-bleu.perl -lc $DATA_DIR/raw/$TESTNAME.de < $OUTPUT.norm > $PREFIX_PATH/average/perl_evalResult_${TESTNAME}
  done
done

# DEVICES=0
# BATCH_SIZE=50
# TRAIN_STEPS=250
# EVAL_STEPS=2
# # domain
# for FS_ID in ${FS_IDS[@]}; do
#   FS_NAME=train-dog50-${FS_ID}
#   OUTPATH=$EXP/visual-finetune_dog/$FS_NAME/${PRETRAIN_DATA}

#   echo training on $FS_NAME

#   CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
#     --input $DATA_DIR/raw/$FS_NAME.$VOCAB.en $DATA_DIR/raw/$FS_NAME.$VOCAB.de \
#     --fewshot_name $FS_NAME \
#     --img_input $DATA_DIR/image_splits $CLIP_FEATURES \
#     --output $OUTPATH \
#     --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
#     --validation $DATA_DIR/raw/val.dog.$VOCAB.en \
#     --references $DATA_DIR/raw/val.dog.de \
#     --checkpoint $CKPT \
#     --model visual_prefix_transformer_v6 \
#     --parameters=prefix_only=False,learning_rate=3e-4,learning_rate_schedule=constant,max_length=64,batch_size=$BATCH_SIZE,device_list=[0],eval_steps=$EVAL_STEPS,train_steps=$TRAIN_STEPS \
#     --hparam_set $MODEL_SIZE

#   # eval
#   PREFIX_PATH=$OUTPATH/eval

#   # average checkpoint
#   python thumt/scripts/average_checkpoints.py --path $PREFIX_PATH --output $PREFIX_PATH/average

#   PREFIX_CKPT=$PREFIX_PATH/average/average-0.pt

#   NAME=vpt_fs

#   TESTNAMES="test.dog cat"

#   for TESTNAME in $TESTNAMES; do
#     INPUT=$DATA_DIR/raw/$TESTNAME.$VOCAB.en
#     OUTPUT=$DATA_DIR/raw/$TESTNAME-$NAME.trans

#     VIT_FEATURES=$DATA_DIR/vit-all.pkl

#     CUDA_VISIBLE_DEVICES=$DEVICES python translate.py \
#       --models visual_prefix_transformer_v6 \
#       --input $INPUT \
#       --img_input $DATA_DIR/image_splits $VIT_FEATURES \
#       --output $OUTPUT \
#       --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
#       --checkpoints $CKPT \
#       --prefix=$PREFIX_CKPT \
#       --parameters=prefix_only=True,device_list=[0],decode_alpha=1.2,beam_size=4,max_length=64 \
#       --hparam_set $MODEL_SIZE

#     sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

#     perl multi-bleu.perl -lc $DATA_DIR/raw/$TESTNAME.de < $OUTPUT.norm > $PREFIX_PATH/average/perl_evalResult_${TESTNAME}
#   done
# done