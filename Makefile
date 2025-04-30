NODE ?= tikgpu04
GRES ?= gpu:1
CPUS ?= 8
MEM ?= 80G
TIME ?= 2:00:00
ORIGIN ?= tikgpu04

conda:
	@echo conda activate moabb

cpu:
	@echo srun --nodelist=$(NODE) --gres=gpu:0 --cpus-per-task=$(CPUS) --mem=$(MEM) --time=$(TIME) --pty bash -i

gpu:
	@echo srun --nodelist=$(NODE) --gres=$(GRES) --cpus-per-task=$(CPUS) --mem=$(MEM) --time=$(TIME) --pty bash -i

rsync:
	@echo rsync -a --inplace /scratch_net/$(ORIGIN)/jbuerki/ /scratch/jbuerki/ -v --progress

info:
	@echo scontrol show node=$(NODE)