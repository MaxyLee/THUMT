ROOT=/data1/private/mxy/projects/mmt

PRETRAIN_DATA=wmt18+cc-en-de-big

EXP=$ROOT/exp/THUMT
# CKPT=$EXP/finetune/$PRETRAIN_DATA/eval
CKPT=$EXP/pretrain/$PRETRAIN_DATA/eval
# PREFIX_CKPT=$EXP/prefix-tuning/$PRETRAIN_DATA/eval/model-79.pt
PREFIX_CKPT=$EXP/visual-prefix-tuning_v2/$PRETRAIN_DATA-mlp/eval/model-64.pt

VOCAB_DIR=$ROOT/data/wmt18/de-en
DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw

TESTNAME=test_2017_mscoco
INPUT=$DATA_DIR/$TESTNAME.32k.en
OUTPUT=$DATA_DIR/$TESTNAME.trans
# INPUT=$VOCAB_DIR/$TESTNAME.tc.32k.en
# OUTPUT=$VOCAB_DIR/$TESTNAME.trans

# normal
# CUDA_VISIBLE_DEVICES=0 python translate.py \
#   --models prefix_transformer \
#   --input $INPUT \
#   --output $OUTPUT \
#   --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
#   --checkpoints $CKPT \
#   --prefix=$PREFIX_CKPT \
#   --parameters=device_list=[0],decode_alpha=1.2,beam_size=4 \
#   --hparam_set big

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1
F30K_IMG_PATH=/data2/share/data/flickr30k-entities/flickr30k-images
M30K_IMG_PATH=/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1
COCO_IMG_PATH=/data2/share/data/coco2014
mapping_type=mlp

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
# visual
CUDA_VISIBLE_DEVICES=4 python translate.py \
  --models visual_prefix_transformer_v2 \
  --input $INPUT \
  --img_input $DATA_DIR/image_splits $IMG_PATH \
  --output $OUTPUT \
  --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
  --checkpoints $CKPT \
  --prefix=$PREFIX_CKPT \
  --parameters=device_list=[0],decode_alpha=1.2,beam_size=4,max_length=64,mapping_type=$mapping_type \
  --hparam_set big

sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw
perl multi-bleu.perl -lc $DATA_DIR/$TESTNAME.de < $OUTPUT.norm > $DATA_DIR/evalResult
# import sacrebleu

cat $DATA_DIR/evalResult