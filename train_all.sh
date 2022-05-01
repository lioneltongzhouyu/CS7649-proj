for i in 1 2 3
do
   echo "[Iteration Num] i = $i"
#    python run_bc_gail.py --pretrain_model 10 --cuda False --seed $i
#   echo ">> run_bc_gail, model 0"
#   python run_bc_gail.py --pretrain_model 0 --cuda False --seed $i
#   echo ">> run_bc_gail, model 1"
#   python run_bc_gail.py --pretrain_model 1 --cuda False --seed $i
#   echo ">> run_bc_bc_gail"
#   python run_bc_bc_gail.py --pretrain_model 0 --cuda False --seed $i
#
    python run_rl_gail.py  --cuda False --seed $i
#   echo ">> run_gail_gail, model 0"
#   python run_gail_gail.py --pretrain_model 0 --cuda False --seed $i
#   echo ">> run_gail_gail, model 1"
#   python run_gail_gail.py --pretrain_model 1 --cuda False --seed $i
#   echo ">> run_gail_gail_gail"
#   python run_gail_gail_gail.py --pretrain_model 0 --cuda False --seed $i
#    python basic.py --pretrain_model 10 --cuda False --seed $i
done