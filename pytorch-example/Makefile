nvtop:
	nvidia-smi |head -n 15
PORT=4000
#docker pull pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
jupyter-GPU-pytorch:
	docker run --gpus all -it --rm --user 0 --group-add users -p ${PORT}:8888 -v "${PWD}":/home/jovyan/work pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
