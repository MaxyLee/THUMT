ROOT=/data1/private/mxy/projects/mmt

PRETRAIN_DATA=wmt18+cc-en-de-big
# PRETRAIN_DATA=cc-en-de

EXP=$ROOT/exp/THUMT
# CKPT=$EXP/finetune/$PRETRAIN_DATA/eval
CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval
# PREFIX_CKPT=$EXP/prefix-tuning/$PRETRAIN_DATA/eval/model-79.pt

mapping_type=mlp
PREFIX_CKPT=$EXP/visual-prefix-tuning_v3/$PRETRAIN_DATA-$mapping_type-2/eval/model-61.pt

VOCAB_DIR=$ROOT/data/wmt18/de-en
DATA_DIR=$ROOT/data/multi30k-dataset/data/task1
F30K_IMG_PATH=/data2/share/data/flickr30k-entities/flickr30k-images
M30K_IMG_PATH=/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1
COCO_IMG_PATH=/data2/share/data/coco2014

NAME=vpt_v3

TESTNAMES="test_2016_flickr test_2017_flickr test_2018_flickr test_2017_mscoco"

for TESTNAME in $TESTNAMES; do
  INPUT=$DATA_DIR/raw/$TESTNAME.32k.en
  OUTPUT=$DATA_DIR/raw/$TESTNAME-$NAME.trans

  if [ "$TESTNAME" = "test_2016_flickr" ]; then
    CLIP_FEATURES=$DATA_DIR/clip-ViT-B_32.pkl
    IMG_PATH=$F30K_IMG_PATH
  else
    CLIP_FEATURES=$DATA_DIR/${TESTNAME}-ViT-B_32.pkl
    if [ "$TESTNAME" = "test_2017_mscoco" ]; then
      IMG_PATH=$COCO_IMG_PATH
    elif [ "$TESTNAME" = "test_2017_flickr" ]; then
      IMG_PATH=$M30K_IMG_PATH/test2017
    else
      IMG_PATH=$M30K_IMG_PATH/test2018
    fi
  fi
  # normal
  # CUDA_VISIBLE_DEVICES=2 python translate.py \
  #   --models transformer \
  #   --input $INPUT \
  #   --output $OUTPUT \
  #   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
  #   --checkpoints $CKPT \
  #   --parameters=device_list=[0],decode_alpha=1.2,beam_size=4 \
  #   --hparam_set big
  # visual
  CUDA_VISIBLE_DEVICES=1 python translate.py \
    --models visual_prefix_transformer_v3 \
    --input $INPUT \
    --img_input $DATA_DIR/image_splits $CLIP_FEATURES \
    --output $OUTPUT \
    --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
    --checkpoints $CKPT \
    --prefix=$PREFIX_CKPT \
    --parameters=device_list=[0],decode_alpha=1.2,beam_size=4,max_length=64,mapping_type=$mapping_type \
    --hparam_set base

  sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

  perl multi-bleu.perl -lc $DATA_DIR/raw/$TESTNAME.de < $OUTPUT.norm > $DATA_DIR/raw/evalResult_${TESTNAME}_${NAME}
done

for TESTNAME in $TESTNAMES; do
  echo $TESTNAME
  cat $DATA_DIR/raw/evalResult_${TESTNAME}_${NAME}
done