#!/bin/bash

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

nbpe=100
bpemode=unigram
dict=../asr1/data/lang_char/${bpemode}${nbpe}_units.txt
bpemodel=../asr1/data/lang_char/${bpemode}${nbpe}

dumpdir=dump
do_delta=false

slu_model=bert-base-nli-stsb-mean-tokens
slu_model_type=sbert

for dset in train dev test; do
    feat_dir=${dumpdir}/${dset}/delta${do_delta}

    text2slu.py \
        --model_name_or_path ${slu_model} \
        --model_type ${slu_model_type} data/${dset}/text \
        ark,scp:${feat_dir}/slu_${slu_model}.ark,${feat_dir}/slu_${slu_model}.scp

    data2json.sh \
        --feat ${feat_dir}/feats.scp \
        --slu ${feat_dir}/slu_${slu_model}.scp --bpecode ${bpemodel}.model \
        data/${dset} ${dict} > ${feat_dir}/data_${slu_model}.json
done

asr=2
slu=0
w=700000
e=100
lr=30
loss=l1_loss
expdir=exp/sbert__tune_${asr}_${slu}_warmup_${w}_epoch_${e}_lr_${lr}_loss_${loss}

${cuda_cmd} ${expdir}/train.log \
    asr_train.py \
    --config conf/tune.yaml \
    --backend pytorch
    --outdir ${expdir} \
    --debugmode 1 \
    --dict ${dict} \
    --minibatches 0 \
    --verbose 0 \
    --train-json dump/train/deltafalse/data_${slu_model}.json \
    --valid-json dump/dev/deltafalse/data_${slu_model}.json \
    --asr-model ../asr1/exp/train/results/model.val7.avg.best \
    --slu-loss ${loss} \
    --slu-model ${slu_model} \
    --slu-tune-weights "asr${asr}+slu${slu}" \
    --early-stop-criterion validation/main/loss \
    --opt noam \
    --num-save-attention 0 \
    --epochs ${e} \
    --transformer-lr ${lr} \
    --transformer-warmup-steps ${w}

for dset in test; do
    ${cuda_cmd} ${expdir}/recog_${dset}.log \
        asr_recog.py \
        --backend pytorch \
        --config conf/decode.yaml \
        --ngpu 1 \
        --batchsize 16 \
        --debugmode 1 \
        --verbose 1 \
        --recog-json dump/${dset}/deltafalse/data_unigram100.json \
        --result-label ${expdir}/${dset}.json \
        --model ${expdir}/results/model.loss.best

    score_slu.py \
        dump/${dset}/deltafalse/data_${slu_model}.json \
        ${expdir}/${dset}.json
done
