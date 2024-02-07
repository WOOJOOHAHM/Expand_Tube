# python /home/lab/hahmwj/Expand_Tube/train.py --model RN50 --dataset hmdb --batch_size 64 --epochs 2 --fast_dev_run True --classifier mean
# python /home/lab/hahmwj/Expand_Tube/train.py --model RN50 --dataset hmdb --batch_size 64 --epochs 2 --fast_dev_run True --classifier span
# python /home/lab/hahmwj/Expand_Tube/train.py --model RN50 --dataset hmdb --batch_size 64 --epochs 2 --fast_dev_run True --classifier difference

python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset hmdb --batch_size 32 --epochs 100  --classifier mean
python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset hmdb --batch_size 32 --epochs 100  --classifier span2
python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset hmdb --batch_size 32 --epochs 100  --classifier difference

python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset ucf --batch_size 32 --epochs 100  --classifier mean
python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset ucf --batch_size 32 --epochs 100  --classifier span2
python /home/lab/hahmwj/Expand_Tube/train.py --model ViT-B/16 --dataset ucf --batch_size 32 --epochs 100  --classifier difference