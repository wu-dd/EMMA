# trian_victim
for seed in 3407 3408 3409
do
  for ds in cifar10 cifar100 cub200
  do
    if [ "$ds" = "cub200" ]; then
      bs=64 ep=50 lr=1e-2 wd=5e-4
    fi
    if [ "$ds" = "cifar10" ]; then
      bs=256 ep=200 lr=1e-1 wd=5e-4
    fi
    if [ "$ds" = "cifar100" ]; then
      bs=256 ep=200 lr=1e-1 wd=5e-4
    fi
    python train_teacher.py --dataset ${ds} --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
  done
done
# python train_victim.py --dataset cifar10 --batch_size 256 --epochs 200 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_victim.py --dataset cifar100 --batch_size 256 --epochs 200 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_victim.py --dataset cub200 --batch_size 64 --epochs 50 --lr 1e-2 --wd 5e-4 --seed 3407


# train surrogate 
for scenario in da kl
do
  for eval_data in cifar10 cifar100 cub200
  do
    if [ "$eval_data" = "cub200" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=caltech256
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cub200
      fi
      bs=64 lr=1e-2 wd=5e-4 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cifar10" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar100
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar10
      fi
      bs=256 lr=1e-1 wd=5e-4 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cifar100" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar10
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar100
      fi
      bs=256 lr=1e-1 wd=5e-4 ep=50 seed=3407
    fi
    python train_surrogate.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data} --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
  done
done
# python train_surrogate.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_surrogate.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_surrogate.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_surrogate.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407
# python train_surrogate.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 50 --lr 1e-2 --wd 5e-4 --seed 3407
# python train_surrogate.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 50 --lr 1e-2 --wd 5e-4 --seed 3407


# train emma
for eval_data in cifar10 cifar100 cub200
do
  for scenario in da kl
  do
    if [ "$eval_data" = "cifar10" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar100
        lamda_list=$(seq 0.0005 0.0005 0.03)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar10
        lamda_list=$(seq 0.001 0.001 0.04)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=0.3 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cifar100" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar10
        lamda_list=$(seq 0.0001 0.0005 0.02)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar100
        lamda_list=$(seq 0.0001 0.0005 0.05)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cub200" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=caltech256
        lamda_list=$(seq 0.0001 0.0005 0.01)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cub200
        lamda_list=$(seq 0.0001 0.0005 0.01)
      fi
      bs=16 ep=50 lr=1e-2 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    for lamda in ${lamda_list}
    do
      for st_epoch in 0 10 20 30 40
      do
         python PoiMat-1.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data} --lamda ${lamda} --lr_gamma ${lr_gamma} --st_epoch ${st_epoch} \
                --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
      done
    done
  done
done
# python EMMA.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20 
# python EMMA.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20 
# python EMMA.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
# python EMMA.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
# python EMMA.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 50 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
# python EMMA.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 50 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20


# get queries
for scenario in da kl
do
  for eval_data in cifar10 cifar100 cub200
  do
    if [ "$eval_data" = "cifar10" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar100
        lamda_list=$(seq 0.0005 0.0005 0.03)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar10
        lamda_list=$(seq 0.001 0.001 0.04)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=0.3 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cifar100" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar10
        lamda_list=$(seq 0.0001 0.0005 0.02)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar100
        lamda_list=$(seq 0.0001 0.0005 0.05)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cub200" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=caltech256
        lamda_list=$(seq 0.0001 0.0005 0.01)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cub200
        lamda_list=$(seq 0.0001 0.0005 0.01)
      fi
      bs=16 ep=50 lr=1e-2 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    for lamda in ${lamda_list}
    do
     for st_epoch in 20
     do
       python get_queries.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data} --lamda ${lamda} --lr_gamma ${lr_gamma}  --st_epoch ${st_epoch} \
       --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
       wait
       nohup python get_queries.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data}  --eval_perturbations --lamda ${lamda} --lr_gamma ${lr_gamma}  --st_epoch ${st_epoch} \
       --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
       wait
     done
    done
  done
done

# python get_queries.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
# python get_queries.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
# python get_queries.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
# python get_queries.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
# python get_queries.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
# python get_queries.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5

# python get_queries.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
# python get_queries.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
# python get_queries.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations  --eval_perturbations
# python get_queries.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
# python get_queries.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
# python get_queries.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations



# eval attacker
for scenario in da kl
do
  for eval_data in cub200
  do
    if [ "$eval_data" = "cifar10" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar100
        lamda_list=$(seq 0.0005 0.0005 0.03)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar10
        lamda_list=$(seq 0.001 0.001 0.04)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=0.3 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cifar100" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=cifar10
        lamda_list=$(seq 0.0001 0.0005 0.02)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cifar100
        lamda_list=$(seq 0.0001 0.0005 0.05)
      fi
      bs=256 ep=200 lr=1e-1 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    if [ "$eval_data" = "cub200" ]; then
      if [ "$scenario" == "kl" ]; then
        transfer_data=caltech256
        lamda_list=$(seq 0.0001 0.0005 0.01)
      elif [ "$scenario" == "da" ]; then
        transfer_data=imagenet_cub200
        lamda_list=$(seq 0.0001 0.0005 0.01)
      fi
      bs=16 ep=50 lr=1e-2 wd=5e-4 lr_gamma=1.0 ep=50 seed=3407
    fi
    for lamda in ${lamda_list}
    do
     for st_epoch in 0 10 20 30 40
     do
       python eval_attacker.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data} --defense ${defense} --lamda ${lamda} --lr_gamma ${lr_gamma}  --st_epoch ${st_epoch} \
       --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
       wait
       nohup python eval_victim.py --eval_dataset ${eval_data} --transfer_dataset ${transfer_data} --defense ${defense} --lamda ${lamda} --lr_gamma ${lr_gamma}  --st_epoch ${st_epoch} \
       --batch_size ${bs} --epochs ${ep} --lr ${lr} --wd ${wd} --seed ${seed}
       wait
     done
    done
  done
done


CUDA_VISIBLE_DEVICES=3 python eval_attacker.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
CUDA_VISIBLE_DEVICES=3 python eval_attacker.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
CUDA_VISIBLE_DEVICES=3 python eval_attacker.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
CUDA_VISIBLE_DEVICES=3 python eval_attacker.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
python eval_attacker.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5
python eval_attacker.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5

CUDA_VISIBLE_DEVICES=3 python eval_victim.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
CUDA_VISIBLE_DEVICES=3 python eval_victim.py --eval_dataset cifar10 --transfer_dataset imagenet_cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
CUDA_VISIBLE_DEVICES=3 python eval_victim.py --eval_dataset cifar100 --transfer_dataset cifar10 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations  --eval_perturbations
CUDA_VISIBLE_DEVICES=3 python eval_victim.py --eval_dataset cifar100 --transfer_dataset imagenet_cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
python eval_victim.py --eval_dataset cub200 --transfer_dataset caltech256 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations
python eval_victim.py --eval_dataset cub200 --transfer_dataset imagenet_cub200 --batch_size 64 --epochs 5 --lr 1e-2 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 5  --eval_perturbations