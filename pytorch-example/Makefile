.PHONY: dmrg
all: dmrg
nvtop:
	nvidia-smi |head -n 15

#CUDA_VISIBLE_DEVICES=1,2 python myscript.py
py	=../env/bin/python
dmrg:
	CUDA_VISIBLE_DEVICES=6 $(py) dmrg-ising-state.py

#sync figs to weilei's macbook 
rsync:
	rsync -rP root@10.200.69.64:/root/weilei/neural-network-example/pytorch-example/fig/ fig


#following content not used
#the port is not open on gpu nodes

PORT=4000
#docker pull pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
#quay.io/jupyter/base-notebook
#jupyter:
#	docker run -it --rm --user 0 --group-add users -p ${PORT}:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/base-notebook

VERSION=jupyter/datascience-notebook:python-3.9.13
PORT=4000
UID=0
jupyter:
	docker run --gpus all -it --rm --user ${UID} --group-add users -p $(PORT):8888 -e JUPYTER_TOKEN=passwd -v "${PWD}":/home/jovyan/work $(VERSION)
#	docker run --gpus all -it --rm --user ${UID} --group-add users -p $(PORT):8888 -e JUPYTER_TOKEN=passwd -v "${PWD}":/home/jovyan/work $(VERSION)

