for i in 1 2 3
do
   echo "[Iteration Num] i = $i"

#   python run_bc_gail.py --pretrain_model 0 --cuda False --seed $i
#   python plot.py --algorithm bc_bc_gail --dataset AntLeg2-v0 --number $i
#   python plot.py --algorithm bc_gail --dataset Antorigin-v0 --number $i
#   python plot.py --algorithm bc_gail --dataset AntLeg2-v0 --number $i
#   python plot.py --algorithm bc_gail --dataset AntLeg3-v0 --number $i
    python plot.py --algorithm rl_gail --dataset Antorigin-v0 --number $i
#   python plot.py --algorithm gail_gail --dataset AntLeg2-v0 --number $i
#   python plot.py --algorithm gail_gail --dataset AntLeg3-v0 --number $i
#   python plot.py --algorithm gail_gail_gail --dataset AntLeg2-v0 --number $i
#   python plot.py --algorithm basic_gail --dataset Antorigin-v0 --number $i
done