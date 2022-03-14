ROOT=/data1/private/mxy/projects/mmt

EXP=$ROOT/exp/THUMT
# CKPT=$EXP/finetune/cc-en-de/eval
CKPT=$EXP/pretrain/cc-en-de/eval
PREFIX_CKPT=$EXP/prefix-tuning/cc-en-de/eval/model-142.pt

VOCAB_DIR=$ROOT/data/wmt18/de-en
DATA_DIR=$ROOT/data/multi30k-dataset/data/task1/raw

TESTNAME=test_2016_flickr
INPUT=$DATA_DIR/$TESTNAME.32k.en
OUTPUT=$DATA_DIR/$TESTNAME.trans
# INPUT=$VOCAB_DIR/$TESTNAME.tc.32k.en
# OUTPUT=$VOCAB_DIR/$TESTNAME.trans

CUDA_VISIBLE_DEVICES=4 python translate.py \
  --models prefix_transformer \
  --input $INPUT \
  --output $OUTPUT \
  --vocabulary $VOCAB_DIR/vocab.32k.en.txt $VOCAB_DIR/vocab.32k.de.txt \
  --checkpoints $CKPT \
  --prefix=$PREFIX_CKPT \
  --parameters=device_list=[0],decode_alpha=1.2,beam_size=4

sed -r 's/(@@ )|(@@ ?$)//g' < $OUTPUT > $OUTPUT.norm

perl multi-bleu.perl -lc $DATA_DIR/$TESTNAME.de < $OUTPUT.norm > $DATA_DIR/evalResult
# import sacrebleu

cat $DATA_DIR/evalResult