PARTITION=$1
CONFIG=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
srun --mpi=pmi2 -p ${PARTITION} --ntasks=${GPUS} --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} \
python -u main.py --config=$CONFIG --distributed #--eval_only
