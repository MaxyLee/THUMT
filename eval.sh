DATA_DIR=/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/raw

perl multi-bleu.perl -lc $DATA_DIR/test_2017_flickr.de < $DATA_DIR/test_2017_flickr.trans.norm > $DATA_DIR/evalResult

cat $DATA_DIR/evalResult