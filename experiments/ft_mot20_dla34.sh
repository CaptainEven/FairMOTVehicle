cd src
python train.py mot --exp_id ft_mot20_dla34 --gpus 0,1 --batch_size 8 --load_model '../models/all_dla34.pth' --data_cfg '../src/lib/cfg/mot20.json' --num_epochs 20 --lr_step '15' --K 500
cd ..