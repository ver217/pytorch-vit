# Run with single node multi GPU

```shell
torchrun --standalone --nnodes=1 --nproc_per_node=gpu train.py
```