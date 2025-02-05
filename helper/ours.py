import torch
import torch.nn.functional as F

from train import update_freq
from utils import correct_num

def IPASD(feats, logits, input_indices, train_target, proto_feature_pre, criterion_cls, num_classes, coslogits, prelogits, epoch, args):
    if epoch < args.epochs * update_freq:
        smooth_weight = args.weight_div * (epoch + 1) / args.epochs
    else:
        smooth_weight = args.weight_div * (args.epochs * update_freq) / args.epochs
    con_weight = args.weight_con
    con_temp = args.con_temp    # Contrastive learning temperature
    con_label = train_target

    all_feat = feats.float().cpu()
    all_label = train_target.cpu().detach().numpy()
    proto_matrix = torch.eye(num_classes)
    label_one_hot = proto_matrix[all_label]
    input_indices = input_indices.cpu()
    loss_con = torch.tensor(0.).cuda()
    if epoch > 0:
        # InfoNCE loss
        proto_feat = proto_feature_pre.float().detach()
        cos_dist = F.cosine_similarity(all_feat.unsqueeze(1), proto_feat.unsqueeze(0),dim=-1)
        cos_logit = torch.div(cos_dist, con_temp).cuda().detach()
        loss_con = criterion_cls(cos_logit, con_label)
    else:
        coslogits[input_indices] = label_one_hot
        prelogits[input_indices] = label_one_hot


    cos_top1, _ = correct_num(coslogits[input_indices], train_target.cpu(), topk=(1, 5))
    pre_top1, _ = correct_num(prelogits[input_indices], train_target.cpu(), topk=(1, 5))

    w_cos = cos_top1 / (cos_top1 + pre_top1)
    w_pre = pre_top1 / (cos_top1 + pre_top1)

    pre_prediction = w_cos * coslogits[input_indices] + w_pre * prelogits[input_indices]

    soft_prediction = (1 - smooth_weight) * label_one_hot + smooth_weight * pre_prediction
    soft_prediction = soft_prediction.cuda()
    log_prob = F.log_softmax(logits, dim=1).cuda()
    loss_smooth = - torch.sum(log_prob * soft_prediction) / train_target.size(0)
    loss = loss_smooth + con_weight * loss_con

    if epoch > 0:
        coslogits[input_indices] = F.softmax(cos_logit, dim=1).cpu().detach()
    prelogits[input_indices] = F.softmax(logits, dim=1).cpu().detach()

    return loss, loss_smooth, loss_con, coslogits, prelogits
