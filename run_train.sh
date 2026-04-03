

TORCH_DISTRIBUTED_DEBUG=INFO \
CUDA_VISIBLE_DEVICES="1,2,3" \
python -m torch.distributed.launch --nproc_per_node 3 --master_port='29502' train.py