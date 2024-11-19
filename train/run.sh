# CUDA_VISIBLE_DEVICES=1 python main.py --data_dir kitti --inWidth 1248 --inHeight 384 --savedir ./kitti --cached_data_file kitti.p

# CUDA_VISIBLE_DEVICES=1 python test.py --data_dir kitti --inWidth 1248 --inHeight 384 \
# --savedir ./kitti --cached_data_file kitti.p

# CUDA_VISIBLE_DEVICES=1 python main.py --scaleIn 1 --decoder True --pretrained kitti_enc_2_8_0/model_300.pth --batch_size 16 \
# --inWidth 1248 --inHeight 384 --savedir ./kitti --cached_data_file kitti.p --data_dir kitti

# CUDA_VISIBLE_DEVICES=2 python test.py --scaleIn 1 --decoder True \
# --data_dir kitti --inWidth 1248 --inHeight 384 --savedir ./kitti --cached_data_file kitti.p


# CUDA_VISIBLE_DEVICES=0 python main.py --id 4 --inWidth 1248 --inHeight 384 \
# --savedir ./kitti --cached_data_file kitti.p --data_dir kitti

# CUDA_VISIBLE_DEVICES=0 python test.py --id 4 --data_dir kitti --inWidth 1248 --inHeight 384 \
# --savedir ./kitti --cached_data_file kitti.p


# CUDA_VISIBLE_DEVICES=1 python main.py --data_dir horse --inWidth 384 --inHeight 256 \
# --savedir ./horse --cached_data_file horse.p

# CUDA_VISIBLE_DEVICES=1 python test.py --data_dir horse --inWidth 384 --inHeight 256 \
# --savedir ./horse --cached_data_file horse.p

# CUDA_VISIBLE_DEVICES=2 python main.py --scaleIn 1 --decoder True --pretrained horse_enc_2_8_0/model_300.pth --batch_size 16 \
# --inWidth 384 --inHeight 256 --savedir ./horse --cached_data_file horse.p --data_dir horse

CUDA_VISIBLE_DEVICES=2 python test.py --scaleIn 1 --decoder True \
--data_dir horse --inWidth 384 --inHeight 256 --savedir ./horse --cached_data_file horse.p

