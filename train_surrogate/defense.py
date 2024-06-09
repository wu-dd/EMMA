import torch
import numpy as np
import tqdm


def generate_perturbations(dataset, teacher, student, method, epsilons=[0.1],batch_size=64, num_workers=4, **kwargs):
    shuffle_indices = np.arange(len(dataset))
    np.random.shuffle(shuffle_indices)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, shuffle_indices),
                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    teacher_pred_perturbed = []
    for (bx, by) in tqdm.tqdm(loader, mininterval=1.0):
        bx = bx.cuda()
        out = method(bx, teacher, student, epsilons=epsilons,  **kwargs).cpu().detach()
        teacher_pred_perturbed.append(out)
    teacher_pred_perturbed = torch.cat(teacher_pred_perturbed, dim=0)

    assert len(teacher_pred_perturbed) == len(shuffle_indices), 'sanity check; can remove later'
    unshuffle_indices = np.zeros(len(dataset))
    for i, p in enumerate(shuffle_indices):
        unshuffle_indices[p] = i
    teacher_pred_perturbed = teacher_pred_perturbed[unshuffle_indices]

    return teacher_pred_perturbed


def method_no_perturbation(bx, teacher, student=None, epsilons=None):
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)

    return teacher_pred.detach()