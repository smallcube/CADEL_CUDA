import torch
import torch.nn.functional as F


def ensemble_loss(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)

    index1 = target.view(pred.shape[0], 1).long()
    p1 = pt.gather(1, index1).clone().detach()

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)

    modulating_factor1 = torch.mean(weight1, 1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum().item()
        if num_in_bin>0:
            w1[inds] = base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight + gamma*num_in_bin/batch_size 
            
        #weights = weights.view(-1, 1)**gamma
        loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2