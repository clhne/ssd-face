#!/bin/bash
DATASET=Face
if [ ! -d data/${DATASET} ]; then
    python create_lmdb.py
fi
TOOLS=~/Detection/ssd/build/tools/caffe
latest=voc/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
echo "Resuming from "${latest}
if test -z $latest; then
	echo "No checkpoints found!"
fi
if [ ! -d snapshot ]; then
    mkdir -p snapshot
fi
# train
${TOOLS} train -solver="${DATASET}/solver_train.prototxt" -weights=$latest

latest=$(ls -t snapshot/*.caffemodel | head -n 1)
echo "Testing "${latest}
# test
${TOOLS} train -solver="${DATASET}/solver_test.prototxt" --weights=$latest
