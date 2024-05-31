# Schnet
CUDA_VISIBLE_DEVICES='0' python main.py --mode train --config-yml configs/ani-1x/schnet/schnet.yml > ani_sch.log

# spinconv
CUDA_VISIBLE_DEVICES='0' python main.py --mode train --config-yml configs/ani-1x/spinconv/spinconv_force.yml > ani_spn.log

# eSCN
CUDA_VISIBLE_DEVICES='0' python main.py --mode train --config-yml configs/ani-1x/escn/eSCN-L6-M2-Lay12-All-MD.yml > ani_esc.log

# equiformer v2
CUDA_VISIBLE_DEVICES='0' python main.py --mode train \
--config-yml configs/ani-1x/equiformer_v2/equiformer_v2_N8_L4_M2_31M.yml > ani_eqf.log

# PaiNN
CUDA_VISIBLE_DEVICES='0' python main.py --mode train --config-yml configs/ani-1x/painn/painn_h512.yml > ani_pai.log