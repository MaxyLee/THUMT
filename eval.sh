ROOT=/data1/private/mxy/projects/mmt
EXP=$ROOT/exp/THUMT

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

if [ "$MODEL_SIZE" = "big" ]; then
  # CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-48.pt
  CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-77.pt
  # CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-68.pt
else
  CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval/model-98.pt
fi

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1
F30K_IMG_PATH=/data2/share/data/flickr30k-entities/flickr30k-images
M30K_IMG_PATH=/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1
COCO_IMG_PATH=/data2/share/data/coco2014

PREFIX_PATH=$EXP/visual-prefix-tuning_v6-2/$PRETRAIN_DATA-3/eval
PREFIX_CKPT=$PREFIX_PATH/model-36.pt

NAME=vpt_v6

TESTNAMES="test_2016_flickr test_2017_flickr test_2018_flickr test_2017_mscoco"

for TESTNAME in $TESTNAMES; do
  INPUT=$DATA_DIR/raw/$TESTNAME.$VOCAB.en
  OUTPUT=$DATA_DIR/raw/$TESTNAME-$NAME.trans

  if [ "$TESTNAME" = "test_2016_flickr" ]; then
    # CLIP_FEATURES=$DATA_DIR/clip-ViT-B_32.pkl
    VIT_FEATURES=$DATA_DIR/vit-all.pkl
    # VIT_FEATURES=$DATA_DIR/flickr-timm-vit_base_patch16_384.pkl
    IMG_PATH=$F30K_IMG_PATH
  else
    # CLIP_FEATURES=$DATA_DIR/${TESTNAME}-ViT-B_32.pkl
    VIT_FEATURES=$DATA_DIR/vit-${TESTNAME}.pkl
    # VIT_FEATURES=$DATA_DIR/${TESTNAME}-timm-vit_base_patch16_384.pkl
    if [ "$TESTNAME" = "test_2017_mscoco" ]; then
      IMG_PATH=$COCO_IMG_PATH
    elif [ "$TESTNAME" = "test_2017_flickr" ]; then
      IMG_PATH=$M30K_IMG_PATH/test2017
    else
      IMG_PATH=$M30K_IMG_PATH/test2018
    fi
  fi
  # visual
  CUDA_VISIBLE_DEVICES=2 python translate.py \
    --models visual_prefix_transformer_v6 \
    --input $INPUT \
    --img_input $DATA_DIR/image_splits $VIT_FEATURES \
    --output $OUTPUT \
    --vocabulary $VOCAB_DIR/vocab.$VOCAB.en.txt $VOCAB_DIR/vocab.$VOCAB.de.txt \
    --checkpoints $CKPT \
    --prefix=$PREFIX_CKPT \
    --parameters=prefix_only=True,visual_prefix_length=30,device_list=[0],decode_alpha=1.2,beam_size=4,max_length=64 \
    --hparam_set $MODEL_SIZE

  sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

  perl multi-bleu.perl -lc $DATA_DIR/raw/$TESTNAME.de < $OUTPUT.norm > $PREFIX_PATH/perl_evalResult_${TESTNAME}_${NAME}

  sacrebleu $DATA_DIR/raw/$TESTNAME.de -i $OUTPUT.norm -w 4 > $PREFIX_PATH/sacrebleu_${TESTNAME}_${NAME}.json
done

for TESTNAME in $TESTNAMES; do
  echo $TESTNAME
  cat $PREFIX_PATH/perl_evalResult_${TESTNAME}_${NAME}
  # cat $DATA_DIR/raw/bleu_${TESTNAME}_${NAME}.json
done