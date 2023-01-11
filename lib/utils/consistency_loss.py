import torch


def semantic_consistency(args, outputs, gt_label, attr_len, feature, fc_weights, memory_q):
    losses = torch.zeros(attr_len, requires_grad=True).cuda()
    criterion_mse = torch.nn.MSELoss()
    b, c, h, w = feature.shape
    count = 0
    for a in range(attr_len):
        all = []
        quality = []
        for i, label in enumerate(gt_label):
            if label[a] == 1:
                cam = fc_weights[a].matmul(feature[i].reshape(c, h * w)).reshape((h, w))
                semantic = torch.einsum('hw, chw -> c', cam, feature[i]) / (h * w)
                all.append(semantic)
                if outputs[i][a] > args.thres:
                    quality.append(semantic)

        if not quality == []:
            count += 1
            quality = torch.stack(quality)
            quality = torch.mean(quality, 0)
            all = torch.stack(all)
            all = torch.mean(all, 0)
            all = all / torch.norm(all, 2)
            if torch.mean(memory_q[a]) == 0:
                losses[a] += criterion_mse(all, quality/torch.norm(quality, 2))
                memory_q[a] = quality.detach().clone()
            else:
                pre_q = memory_q[a] / torch.norm(memory_q[a])
                quality = quality / torch.norm(quality, 2)
                quality = pre_q * (1 - 0.9) + 0.9 * quality
                losses[a] += criterion_mse(all, quality)
                memory_q[a] = quality.detach().clone()

    if count == 0:
        return torch.tensor([0.0], requires_grad=True).cuda(), memory_q
    else:
        return losses.mean(), memory_q




def spatial_consistency(args, outputs, gt_label, attr_len, feature, fc_weights, memory_q):
    losses = torch.zeros(attr_len, requires_grad=True).cuda()
    criterion_mse = torch.nn.MSELoss()
    b, c, h, w = feature.shape
    count = 0
    for a in range(attr_len):
        all = []
        quality = []
        for i, label in enumerate(gt_label):
            if label[a] == 1:
                cam = fc_weights[a].matmul(feature[i].reshape(c, h * w))
                all.append(cam)
                if outputs[i][a] > args.thres:
                    quality.append(cam)

        if not quality == []:
            count += 1
            quality = torch.stack(quality)
            quality = torch.mean(quality, 0)
            all = torch.stack(all)
            all = torch.mean(all, 0)
            all = all / torch.norm(all, 2)
            if torch.mean(memory_q[a]) == 0:
                losses[a] += criterion_mse(all, quality/torch.norm(quality, 2))
                memory_q[a] = quality.detach().clone()
            else:
                pre_q = memory_q[a] / torch.norm(memory_q[a])
                quality = quality / torch.norm(quality, 2)
                quality = pre_q * (1 - 0.9) + 0.9 * quality
                losses[a] += criterion_mse(all, quality)
                memory_q[a] = quality.detach().clone()

    if count == 0:
        return torch.tensor([0.0], requires_grad=True).cuda(), memory_q
    else:
        return losses.mean(), memory_q