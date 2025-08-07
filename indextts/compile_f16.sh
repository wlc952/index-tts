#!/bin/bash
set -ex
models=""
name="indextts"
chip="bm1684x"
out_model=$name.bmodel
mode="f16"

seq_length=256
hidden_size=1280
head_dim=64
num_blocks=24

onnx_dir=$PWD/tmp/onnx
folder='tmp/'$name'_'$chip'_'$mode
out_model=$name'_'$chip'_'$mode'_seq'${seq_length}'.bmodel'

# Convert block
outdir=${folder}/block
mkdir -p $outdir
pushd $outdir

process_block()
{
    i=$1

    model_transform.py \
        --model_name block_$i \
        --model_def ${onnx_dir}/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        --quantize W4F16 --q_group_size 64 \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ${onnx_dir}/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        --quantize W4F16 --q_group_size 64 \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel
}

# Process each block
for ((i=0; i<$num_blocks; i++)); do
    process_block $i 
    models="${models}${outdir}/block_${i}.bmodel ${outdir}/block_cache_${i}.bmodel "
done

popd
echo $models

# convert embedding
outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir


model_transform.py \
    --model_name mel_embedding \
    --model_def ${onnx_dir}/mel_embedding.onnx \
    --mlir mel_embedding.mlir

model_deploy.py \
    --mlir mel_embedding.mlir \
    --quantize F16 \
    --quant_output \
    --chip ${chip} \
    --model mel_embedding.bmodel

model_transform.py \
    --model_name text_embedding \
    --model_def ${onnx_dir}/text_embedding.onnx \
    --mlir text_embedding.mlir

model_deploy.py \
    --mlir text_embedding.mlir \
    --quantize F16 \
    --quant_output \
    --chip ${chip} \
    --addr_mode io_alone \
    --model text_embedding.bmodel

model_transform.py \
    --model_name text_embedding_cache \
    --model_def ${onnx_dir}/text_embedding_cache.onnx \
    --mlir text_embedding_cache.mlir

model_deploy.py \
    --mlir text_embedding_cache.mlir \
    --quantize F16 \
    --quant_output \
    --chip ${chip} \
    --addr_mode io_alone \
    --model text_embedding_cache.bmodel

model_transform.py \
    --model_name conds_encoder \
    --model_def ${onnx_dir}/conds_encoder.onnx \
    --mlir conds_encoder.mlir

model_deploy.py \
    --mlir conds_encoder.mlir \
    --quantize F32 \
    --chip ${chip} \
    --addr_mode io_alone \
    --model conds_encoder.bmodel

models=$models' '$outdir'/mel_embedding.bmodel '$outdir'/text_embedding.bmodel '$outdir'/text_embedding_cache.bmodel '${outdir}'/conds_encoder.bmodel '

popd
echo $models

# convert lm_head
outdir=${folder}/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ${onnx_dir}/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --chip ${chip} \
    --model lm_head.bmodel

model_transform.py \
    --model_name penalty_sample_head \
    --model_def ${onnx_dir}/penalty_sample_head.onnx \
    --mlir penalty_sample_head.mlir

model_deploy.py \
    --mlir penalty_sample_head.mlir \
    --chip ${chip} \
    --model penalty_sample_head.bmodel

model_transform.py \
    --model_name ln_f \
    --model_def ${onnx_dir}/ln_f.onnx \
    --mlir ln_f.mlir

model_deploy.py \
    --mlir ln_f.mlir \
    --quantize F16 \
    --quant_input \
    --chip ${chip} \
    --model ln_f.bmodel

model_transform.py \
    --model_name ln_f2 \
    --model_def ${onnx_dir}/ln_f2.onnx \
    --mlir ln_f2.mlir

model_deploy.py \
    --mlir ln_f2.mlir \
    --quantize F16 \
    --quant_input \
    --chip ${chip} \
    --model ln_f2.bmodel

model_transform.py \
    --model_name final_norm \
    --model_def ${onnx_dir}/final_norm.onnx \
    --mlir final_norm.mlir

model_deploy.py \
    --mlir final_norm.mlir \
    --chip ${chip} \
    --model final_norm.bmodel


models=${models}${outdir}'/lm_head.bmodel '${outdir}'/penalty_sample_head.bmodel '${outdir}'/ln_f.bmodel '${outdir}'/ln_f2.bmodel '${outdir}'/final_norm.bmodel '
popd
echo $models

# convert bigvgan
outdir=${folder}/bigvgan
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name speaker_encoder \
    --model_def ${onnx_dir}/bigvgan_speaker_encoder.onnx \
    --mlir speaker_encoder.mlir

model_deploy.py \
    --mlir speaker_encoder.mlir \
    --quantize F16 \
    --chip ${chip} \
    --addr_mode io_alone \
    --model speaker_encoder.bmodel

model_transform.py \
    --model_name bigvgan \
    --model_def ${onnx_dir}/bigvgan_filter_approximated.onnx \
    --mlir bigvgan.mlir

model_deploy.py \
    --mlir bigvgan.mlir \
    --quantize F16 \
    --chip ${chip} \
    --model bigvgan.bmodel

models=${models}${outdir}'/speaker_encoder.bmodel '${outdir}'/bigvgan.bmodel '
popd
echo $models
# combine all models
model_tool --combine $models -o $out_model