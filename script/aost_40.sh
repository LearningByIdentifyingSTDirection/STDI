EXP='../AVL_data/aost_40/'
if [ $1 = "train" ];then
	PTH="none"
	BS=256
else
	PTH="${EXP}save_${2}.pth"
	BS=256
fi
echo ${PTH}

python -m AOST.main --task 'aost' --stage ${1} --lr 0.01 --experiment_folder ${EXP} \
--resume_path ${PTH} --batch_size ${BS} --crop_size 40