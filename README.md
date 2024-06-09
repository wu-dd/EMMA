# EMMA

This is the implementation of our CVPR'24 paper (**E**fficient **M**odel Stealing Defense with Noise Transition **Ma**trix).

Requirements: Python 3.8.12, numpy 1.22.3, torch 1.12.1, torchvision 0.13.1.



### **Experiment Pipeline Overview**

Our experiment pipeline comprises three fundamental steps:

1. **Victim Model Initialization:**

   - We begin by initializing victim models through training on each evaluation dataset, including CIFAR10, CIFAR100, and CUB200.

2. **Bi-level Optimization for Model Stealing Defense:**

   - Surrogate Model Initialization:

      In this phase, surrogate models are initialized by distilling from the victim models using the query dataset. This process is executed in two scenarios:

     - *Knowledge Limited Scenarios:* Utilizing datasets such as CIFAR100, CIFAR10, and CALTECH256.
     - *Distribution-aware Scenarios:* Incorporating datasets like ImageNet-CIFAR10, ImageNet-CIFAR100, and ImageNet-CUB200. Early stopping mechanisms are employed during this distillation process.

   - **Transition Matrix Optimization:** This step involves optimizing the transition matrix under various defense budget constraints.

3. **Validation of Model Stealing Attack:**

   - **Generation of Perturbed Posteriors:** Perturbed posteriors are generated for each query dataset specifically targeting the victim model.
   - **Attacker Model Training:** We then train the attacker model using the collected query-response dataset.

*Note:* Distribution-aware datasets can be accessed [here](https://drive.google.com/drive/folders/1pApfOeDAYPyICsG4YsQuQlaPAKieGNXo?usp=sharing).



## Defense Demonstration: CIFAR100 → CIFAR10 Model Stealing Attack Scenario

Here's how we demonstrate the defense against a CIFAR100 → CIFAR10 model stealing attack:

**Step 1:** Initialize victim models using the CIFAR10 dataset.

```bash
cd train_victim
python train_victim.py --dataset cifar10 --batch_size 256 --epochs 200 --lr 1e-1 --wd 5e-4 --seed 3407
```

**Step 2.1:** Initialize surrogate models by distilling from the victim models on the CIFAR100 dataset, incorporating early stopping mechanisms.

~~~bash
cd ..
cp -r train_victim/victim_models train_surrogate/
cd train_surrogate/
python train_surrogate.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407
~~~

**Step 2.2:** Optimize the transition matrix under various defense budgets.

```bash
cd ..
cp -r train_victim/victim_models train_matrix/
cp -r train_surrogate/surrogate_models train_matrix/
cd train_matrix/
python EMMA.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
```

**Step 3.1:** Generate perturbed posteriors targeting the CIFAR10 model.

```bash
cd ..
cp -r train_victim/victim_models generate_query/
cp -r train_matrix/matrix generate_query/
cd generate_query/
python get_queries.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
python get_queries.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 5 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20  --eval_perturbations
```

**Step 3.2:** Train the attacker model utilizing the collected query-response dataset.

```bash
cd ..
cp -r generate_query/perturbations train_attacker/
cd train_attacker/

python eval_attacker.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
python eval_victim.py --eval_dataset cifar10 --transfer_dataset cifar100 --batch_size 256 --epochs 50 --lr 1e-1 --wd 5e-4 --seed 3407 --lamda 0.001 --lr_gamma 0.3 --st_epoch 20
```

If you have any further questions, please feel free to send an e-mail to: [dongdongwu@seu.edu.cn](mailto:dongdongwu@seu.edu.cn). Have fun!