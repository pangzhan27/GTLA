import os
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy
import logging
import wandb
from copy import deepcopy
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
################## Helper function
def get_frame_num_per_cls(list_file, gt_path, actions_dict):
    file_ptr = open(list_file, 'r')
    list_of_examples = file_ptr.read().split('\n')[:-1]
    file_ptr.close()

    num_per_cls = np.zeros(len(actions_dict))
    for vid in list_of_examples:
        file_ptr = open(gt_path + vid, 'r')
        contents = file_ptr.read().split('\n')[:-1]
        for c in contents:
            num_per_cls[actions_dict[c]] += 1
    return num_per_cls

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, logpath, logfile, level='info',fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt)
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(format_str)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(format_str)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

################## Metrics
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def overlap_f1(P, Y, overlap=.1, bg_class=["background"]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score(P[i], Y[i], overlap, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / float(TP + FP + 1e-8)
    recall = TP / float(TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall+1e-16)
    F1 = np.nan_to_num(F1)
    return F1 * 100

def accuracy(P, Y):
    total = 0.
    correct = 0
    for i in range(len(P)):
        total += len(Y[i])
        correct += (P[i] == Y[i]).sum()
    return torch.Tensor([100 * correct / total])

def levenstein_(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], 'float')
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(P, Y, norm=True, bg_class=["background"]):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_, _, _ = get_labels_start_end_time(P, bg_class)
        Y_, _, _ = get_labels_start_end_time(Y, bg_class)
        return levenstein_(P_, Y_, norm)

# balanced metric
def f_score_ana(recognized, ground_truth, overlap, actions_dict, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = np.zeros(len(actions_dict))
    fp = np.zeros(len(actions_dict))
    fn = np.zeros(len(actions_dict))
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp[actions_dict[p_label[j]]] += 1
            hits[idx] = 1
        else:
            fp[actions_dict[p_label[j]]] += 1
    for j in range(len(y_label)):
        if hits[j] == 0:
            fn[actions_dict[y_label[j]]] += 1

    return tp, fp, fn


def overlap_f1_macro(P, Y, overlap=.1, bg_class=["background"]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score_ana(P[i], Y[i], overlap, actions_dict, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return F1 * 100


def b_accuracy(P, Y):
    total = np.zeros(len(actions_dict))
    correct = np.zeros(len(actions_dict))
    cover = np.zeros(len(actions_dict))
    for i in range(len(P)):
        num = min(len(P[i]), len(Y[i]))
        for j in range(num):
            if P[i][j] == Y[i][j]:
                correct[actions_dict[Y[i][j]]] += 1
            total[actions_dict[Y[i][j]]] += 1
            cover[actions_dict[P[i][j]]] += 1

    avg_acc = 100 * correct / (total + 1e-8)
    avg_prec = 100 * correct / (cover + 1e-8)
    return avg_acc, avg_prec


################## Dataloader
class BatchGenerator(torch.utils.data.Dataset):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.read_data(vid_list_file)
    
    def __len__(self):
        return len(self.list_of_examples)
    
    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def getitem(self, index):
        vid = self.list_of_examples[index]

        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        input = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]

        batch_input_tensor = torch.from_numpy(input).float()
        batch_target_tensor = torch.from_numpy(target).long()
        mask = torch.ones(self.num_classes, np.shape(input)[1])

        return batch_input_tensor, batch_target_tensor, mask, vid

    def __getitem__(self, index):
        return self.getitem(index)
  

################## Model (change loss)
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes))
                                     if s != num_stages-2 else
                                     copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes+args.nbins))
                                     for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = [out]
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs.append(out)
        return outputs

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


def cross_entropy(pred, label, weight=None, avg_factor=None):
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        loss = loss * weight.float()
    
    if avg_factor is None:
        loss = loss.mean()
    else:
        loss = loss.sum() / avg_factor
    return loss

################## Trainer (change loss)
class Trainer:
    def __init__(self, model, log, sample_rate, **kwargs):
        set_seed(seed)
        self.model = model(**kwargs)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_split = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = kwargs.get('num_classes', 0)
        self.sample_rate = sample_rate
        assert self.num_classes > 0, "wrong class numbers"
        self.log = log
        self.log.logger.info('Model Size: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.nbins = int(args.nbins)
        self.others_sample_ratio = int(args.ratio)
        self.loss_bins = []
        for i in range(self.nbins):
            self.loss_bins.append(cross_entropy)
        self.creat_index()

    def creat_index(self):
        cast_info = {}
        fg_splits ={}
        for indx, ky in enumerate(['frequent', 'common', 'rare']):
            fg_splits[indx] = []
            for j in group_dict[ky]:
                cast_info[actions_dict[j]] = indx
                fg_splits[indx].append(actions_dict[j])

        self.fg_splits = []
        for j in range(len(fg_splits)):
            self.fg_splits.append(torch.from_numpy(np.array(fg_splits[j])).long().to(device))

        binlabel_count = [1] * self.nbins
        label2binlabel = np.zeros((self.nbins, self.num_classes), dtype='int')
        for cid, group in cast_info.items():
            label2binlabel[group, cid] = binlabel_count[group]
            binlabel_count[group] += 1
        self.label2binlabel = torch.from_numpy(label2binlabel).long().to(device)

        # start and length
        pred_slice = np.zeros((self.nbins, 2), dtype='int')
        start_idx = 0
        for i, bincount in enumerate(binlabel_count):
            pred_slice[i, 0] = start_idx
            pred_slice[i, 1] = bincount
            start_idx += bincount

        self.pred_slice = torch.from_numpy(pred_slice).long().to(device)

    def _sample_others(self, label):
        fg = torch.where(label > 0, torch.ones_like(label), torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.others_sample_ratio)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num,), replace=False)
            sample_idx = torch.from_numpy(sample_idx).to(device)
            fg[sample_idx] = 1
            weight = fg

        return weight

    def _remap_labels(self, labels):
        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels]

            weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _slice_preds(self, cls_score):
        new_preds = []
        num_bins = self.pred_slice.shape[0]
        for i in range(num_bins):
            start = self.pred_slice[i, 0]
            length = self.pred_slice[i, 1]
            sliced_pred = cls_score.narrow(1, start, length)
            new_preds.append(sliced_pred)
        return new_preds

    def suppression_loss(self, cls_score, labels):
        '''
        :param cls_logits: shape TxC
        :param labels: shape T
        :return: softmax cross entropy based on modified logits
        '''
        total_loss = 0
        new_labels, new_weights, new_avgfactors = self._remap_labels(labels)
        new_preds = self._slice_preds(cls_score)
        pos_losses = []
        for i in range(self.nbins):
            total_loss += self.loss_bins[i](new_preds[i], new_labels[i], new_weights[i], avg_factor=new_avgfactors[i])
        
        return total_loss/self.nbins

    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. Decay the score of normal classes
        with the score of fg.
        From v1.
        '''

        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        fg_score = new_scores
        fg_merge = torch.zeros((num_proposals, self.num_classes)).to(device)

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i][:, 1:]
        return fg_merge
    
    
    def train(self, save_dir, num_epochs, batch_size, learning_rate, device, actions_dict,
              batch_gen_tst=None):

        self.model.train()
        self.model.to(device)
        resume_epoch = 0
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_score = -10000
        if args.resume > 0:
            state = torch.load("./models/" + args.dataset + "/split_" + args.split +  TYPE + "/epoch-last" + ".model",map_location = device)
            resume_epoch = state['epoch'] + 1
            self.model.load_state_dict(state['net'])
            best_score = state['score']

        for epoch in range(resume_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            ce_loss, smooth_loss = [0, 0, 0, 0], [0, 0, 0, 0]
            for _, items in enumerate(train_loader):
                batch_input, batch_target, mask, vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                mask_new = torch.ones((mask.shape[0], mask.shape[1] + self.nbins, mask.shape[2])).to(device)
                mask_ind = self.pred_slice[:,0]
                mask_new[0, mask_ind, :] = 0

                optimizer.zero_grad()
                activity = vids[0].split('.txt')[0].split('_')[-1]

                ps = self.model(batch_input, mask)
                loss = 0
                for i, p in enumerate(ps):
                    if i != len(ps)-1:
                        s_ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),batch_target.view(-1))
                        s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                    else:
                        s_ce_loss = self.suppression_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins),batch_target.view(-1))                       
                        s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask_new[:, :, 1:])
                    loss += s_ce_loss
                    loss += s_smooth_loss
                    ce_loss[i] += s_ce_loss.item()
                    smooth_loss[i] += s_smooth_loss.item()

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                merge_results = self._merge_score(ps[-1].transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins))
                _, predicted = torch.max(merge_results, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            nums = len(batch_gen.list_of_examples)
            pr_str = "[epoch %d]: loss = %f, ce1 = %f, ce2 = %f, ce3 = %f, ce4 = %f, " \
                     "sm1 = %f, sm2 = %f, sm3 = %f, sm4 = %f,  acc = %f" % \
                     (epoch + 1, epoch_loss/nums, np.round(ce_loss[0]/nums,3), np.round(ce_loss[1]/nums,3),
                      np.round(ce_loss[2] / nums, 3), np.round(ce_loss[3]/nums,3), np.round(smooth_loss[0]/nums,3),
                      np.round(smooth_loss[1] / nums, 3), np.round(smooth_loss[2]/nums,3), np.round(smooth_loss[3]/nums,3),
                      float(correct) / total)
            self.log.logger.info(pr_str)

            test_score, test_log, CM_tst = self.test(epoch, actions_dict, device)
            if test_score > best_score:
                best_score = test_score
                best_save = {'net': self.model.state_dict(), 'epoch': epoch, 'score': best_score}
                torch.save(best_save, save_dir + "/epoch-best" + ".model")
                self.log.logger.info("Save for the best model")
            
            last_save = {'net': self.model.state_dict(), 'epoch': epoch, 'score': best_score}
            torch.save(last_save, save_dir + "/epoch-last" + ".model")

            log_dict = {'epoch': epoch}
            log_dict.update(test_log)

            train_log, CM = self.train_eval()
            log_dict.update(train_log)

            if args.is_wandb:
                wandb.log(log_dict)


    def train_eval(self):
        self.model.eval()
        total_frames, confusion_matrix = 0, np.zeros((num_classes, num_classes))
        preds, labels = [], []
        epoch_loss = 0
        loss_d, conf_d = {}, {}
        ce_loss, smooth_loss = [0, 0, 0, 0], [0, 0, 0, 0]
        for i, items in enumerate(train_ad_loader):
            batch_input, batch_target, mask, vids = items
            batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
            mask_new = torch.ones((mask.shape[0], mask.shape[1] + self.nbins, mask.shape[2])).to(device)
            mask_ind = self.pred_slice[:,0]
            mask_new[0, mask_ind, :] = 0

            total_frames += len(batch_target.view(-1))
            activity = vids[0].split('.txt')[0].split('_')[-1]

            ps = self.model(batch_input, mask)

            merge_results = self._merge_score(ps[-1].transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins))
            _, predicted = torch.max(merge_results, 1)
            pred1s, lbls = predicted.view(-1, 1).detach().cpu().numpy(), batch_target.view(-1, 1).cpu().numpy()
            np.add.at(confusion_matrix, (lbls, pred1s), 1)

            # loss
            loss = 0
            for i, p in enumerate(ps):
                if i != len(ps)-1:
                    s_ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),batch_target.view(-1))
                    s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                else:
                    s_ce_loss = self.suppression_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins),
                                                      batch_target.view(-1))                       
                    s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask_new[:, :, 1:])
                
                if i == len(ps) - 1:
                    s_ce_loss_split = -torch.log(merge_results)
                    s_ce_loss_split = s_ce_loss_split[torch.arange(len(batch_target.view(-1))), batch_target.view(-1)]
                    confidence = merge_results[torch.arange(len(batch_target.view(-1))), batch_target.view(-1)]
                    current_l = np.unique(batch_target.view(-1).cpu().numpy())
                    for k in current_l:
                        if k not in loss_d:
                            loss_d[k] = []
                            conf_d[k] = []
                        tmpt_loss = s_ce_loss_split[batch_target.view(-1) == k]
                        tmpt_conf = confidence[batch_target.view(-1) == k]
                        loss_d[k].append(torch.mean(tmpt_loss).item())
                        conf_d[k].append(torch.mean(tmpt_conf).item())

                loss += s_ce_loss
                loss += s_smooth_loss
                ce_loss[i] += s_ce_loss.item()
                smooth_loss[i] += s_smooth_loss.item()

            epoch_loss += loss.item()

            predicted = predicted.squeeze().cpu()
            batch_target = batch_target.squeeze().cpu()
            predicted_word, label_word = [], []
            for i in range(len(predicted)):
                predicted_word += [label_dict[predicted[i].item()]] * self.sample_rate
                label_word += [label_dict[batch_target[i].item()]] * self.sample_rate

            preds.append(np.array(predicted_word))
            labels.append(np.array(label_word))

        assert np.sum(confusion_matrix) == total_frames
        confusion_matrix /= total_frames

        results = {}
        # action: standard metrics
        results['f1_10'] = overlap_f1_macro(preds, labels, overlap=0.1)
        results['f1_25'] = overlap_f1_macro(preds, labels, overlap=0.25)
        results['f1_50'] = overlap_f1_macro(preds, labels, overlap=0.50)
        results['f_rec'], results['f_prec'] = b_accuracy(preds, labels)

        results1 = {}
        # action: standard metrics
        results1['f1_10'] = overlap_f1(preds, labels, overlap=0.1)
        results1['f1_25'] = overlap_f1(preds, labels, overlap=0.25)
        results1['f1_50'] = overlap_f1(preds, labels, overlap=0.50)
        results1['f_acc'] = accuracy(preds, labels).item()
        results1['edit'] = edit_score(preds, labels)

        self.model.train()


        head_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in freq_list))])
        com_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in common_list))])
        tail_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in rare_list))])
        head_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in freq_list))])
        com_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in common_list))])
        tail_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in rare_list))])
        head_recall = np.mean([results['f_rec'][i] for i in freq_list if i in loss_d])
        com_recall = np.mean([results['f_rec'][i] for i in common_list if i in loss_d])
        tail_recall = np.mean([results['f_rec'][i] for i in rare_list if i in loss_d])
        head_prec = np.mean([results['f_prec'][i] for i in freq_list if i in loss_d])
        com_prec = np.mean([results['f_prec'][i] for i in common_list if i in loss_d])
        tail_prec = np.mean([results['f_prec'][i] for i in rare_list if i in loss_d])
        f1 = 2 * results['f_rec'] * results['f_prec'] / (results['f_rec'] + results['f_prec'] + 1e-8)
        head_f1 = np.mean([f1[i] for i in freq_list if i in loss_d])
        com_f1 = np.mean([f1[i] for i in common_list if i in loss_d])
        tail_f1 = np.mean([f1[i] for i in rare_list if i in loss_d])


        nums = len(batch_gen_ad.list_of_examples)
        log_dict = {'train_loss': ce_loss[3] / nums, 
                    'train_head_loss': head_loss, 'train_com_loss': com_loss, 'train_tail_loss': tail_loss,
                    'train_head_conf': head_conf, 'train_com_conf': com_conf, 'train_tail_conf': tail_conf,
                    'train_mean_rec': np.mean([results['f_rec'][i] for i in range(num_classes) if i in loss_d]),
                    'train_mean_prec': np.mean([results['f_prec'][i] for i in range(num_classes) if i in loss_d]),
                    'train_head_rec': head_recall, 'train_com_rec': com_recall, 'train_rare_rec': tail_recall,
                    'train_head_prec': head_prec, 'train_com_prec': com_prec, 'train_rare_prec': tail_prec,
                    'train_head_f1': head_f1, 'train_com_f1': com_f1, 'train_rare_f1': tail_f1,
                    'train_glb_acc': results1['f_acc'], 'train_edit': results1['edit'],
                    'train_glb_f1_10': results1['f1_10'], 'train_glb_f1_25': results1['f1_25'], 'train_glb_f1_50': results1['f1_50'],
                    'train_cls_f1_10': np.mean([results['f1_10'][i] for i in range(num_classes) if i in loss_d]), 
                    'train_cls_f1_25': np.mean([results['f1_25'][i] for i in range(num_classes) if i in loss_d]),
                    'train_cls_f1_50': np.mean([results['f1_50'][i] for i in range(num_classes) if i in loss_d]),}
        return log_dict, confusion_matrix

    def test(self, epoch, actions_dict, device):
        self.model.eval()
        total_frames, confusion_matrix = 0, np.zeros((num_classes, num_classes))
        preds = []
        labels = []
        epoch_loss = 0
        ce_loss, smooth_loss = [0, 0, 0, 0], [0, 0, 0, 0]
        loss_d, conf_d = {}, {}
        with torch.no_grad():
            for i, items in enumerate(test_loader):
                batch_input, batch_target, mask, vids = items
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                mask_new = torch.ones((mask.shape[0], mask.shape[1] + self.nbins, mask.shape[2])).to(device)
                mask_ind = self.pred_slice[:,0]
                mask_new[0, mask_ind, :] = 0
                total_frames += len(batch_target.view(-1))
                activity = vids[0].split('.txt')[0].split('_')[-1]

                ps = self.model(batch_input, mask)
                merge_results = self._merge_score(ps[-1].transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins))
                _, predicted = torch.max(merge_results, 1)
                pred1s, lbls = predicted.view(-1, 1).detach().cpu().numpy(), batch_target.view(-1, 1).cpu().numpy()
                #pred_activity = self.decide_activity(pred1s, activity)
                np.add.at(confusion_matrix, (lbls, pred1s), 1)
                

                loss = 0
                for i, p in enumerate(ps):
                    if i != len(ps)-1:
                        s_ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),batch_target.view(-1))
                        s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                    else:
                        s_ce_loss = self.suppression_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins),
                                                      batch_target.view(-1))                       
                        s_smooth_loss = 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask_new[:, :, 1:])

                    if i == len(ps) - 1:
                        s_ce_loss_split = -torch.log(merge_results)
                        s_ce_loss_split = s_ce_loss_split[torch.arange(len(batch_target.view(-1))), batch_target.view(-1)]
                        confidence = merge_results[torch.arange(len(batch_target.view(-1))), batch_target.view(-1)]
                        current_l = np.unique(batch_target.view(-1).cpu().numpy())
                        for k in current_l:
                            if k not in loss_d:
                                loss_d[k] = []
                                conf_d[k] = []
                            tmpt_loss = s_ce_loss_split[batch_target.view(-1) == k]
                            tmpt_conf = confidence[batch_target.view(-1) == k]
                            loss_d[k].append(torch.mean(tmpt_loss).item())
                            conf_d[k].append(torch.mean(tmpt_conf).item())

                    loss += s_ce_loss
                    loss += s_smooth_loss
                    ce_loss[i] += s_ce_loss.item()
                    smooth_loss[i] += s_smooth_loss.item()

                epoch_loss += loss.item()

                predicted = predicted.squeeze().cpu()
                batch_target = batch_target.squeeze().cpu()
                predicted_word, label_word = [], []
                for i in range(len(predicted)):
                    predicted_word += [label_dict[predicted[i].item()]] * self.sample_rate
                    label_word += [label_dict[batch_target[i].item()]] * self.sample_rate

                preds.append(np.array(predicted_word))
                labels.append(np.array(label_word))

            assert np.sum(confusion_matrix) == total_frames
            confusion_matrix /= total_frames
            
            nums = len(batch_gen_tst.list_of_examples)
            pr_str = "***[epoch %d]***: loss = %f, ce1 = %f, ce2 = %f, ce3 = %f, ce4 = %f, " \
                     "sm1 = %f, sm2 = %f, sm3 = %f, sm4 = %f" % \
                     (epoch + 1, epoch_loss / nums, np.round(ce_loss[0] / nums, 3), np.round(ce_loss[1] / nums, 3),
                      np.round(ce_loss[2] / nums, 3), np.round(ce_loss[3] / nums, 3),
                      np.round(smooth_loss[0] / nums, 3),
                      np.round(smooth_loss[1] / nums, 3), np.round(smooth_loss[2] / nums, 3),
                      np.round(smooth_loss[3] / nums, 3))
            self.log.logger.info(pr_str)
            results1 = {}
            # action: standard metrics
            results1['f1_10'] = overlap_f1(preds, labels, overlap=0.1)
            results1['f1_25'] = overlap_f1(preds, labels, overlap=0.25)
            results1['f1_50'] = overlap_f1(preds, labels, overlap=0.50)
            results1['f_acc'] = accuracy(preds, labels).item()
            results1['edit'] = edit_score(preds, labels)

            results = {}
            # action: standard metrics
            results['f1_10_s'] = overlap_f1_macro(preds, labels, overlap=0.1)
            results['f1_25_s'] = overlap_f1_macro(preds, labels, overlap=0.25)
            results['f1_50_s'] = overlap_f1_macro(preds, labels, overlap=0.50)
            results['f_rec'], results['f_prec'] = b_accuracy(preds, labels)

            test_not_appear = np.array([i for i in range(num_classes) if i not in loss_d])
            valid_num = num_classes - len(test_not_appear)
            results['f_acc'] = np.sum(results['f_rec']) / valid_num
            results['f1_10'] = np.sum(results['f1_10_s']) / valid_num
            results['f1_25'] = np.sum(results['f1_25_s']) / valid_num
            results['f1_50'] = np.sum(results['f1_50_s']) / valid_num

            results1['total_score'] = (results1['f1_10'] + results1['f1_25'] + results1['f1_50'])/3.0 + results1['f_acc'] +  results1['edit'] + (results['f1_10']  + results['f1_25']  + results['f1_50'])/3.0  + results['f_acc']

            self.log.logger.info(
                "---[epoch %d]---: tst edit = %f, f1_10 = %f, f1_25 = %f, f1_50 = %f, acc = %f, total = %f "
                % (epoch + 1, results1['edit'], results1['f1_10'], results1['f1_25'], results1['f1_50'], results1['f_acc'], results1['total_score']))

            self.log.logger.info( 
                "                  balanced acc = %f, f1_10 = %f, f1_25 = %f, f1_50 = %f" % (results['f_acc'], results['f1_10'],  results['f1_25'], results['f1_50']))

        self.model.train()

        head_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in freq_list))])
        com_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in common_list))])
        tail_loss = np.mean([np.mean(loss_d[i]) for i in range(num_classes) if ((i in loss_d) and (i in rare_list))])
        head_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in freq_list))])
        com_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in common_list))])
        tail_conf = np.mean([np.mean(conf_d[i]) for i in range(num_classes) if ((i in conf_d) and (i in rare_list))])
        head_recall = np.mean([results['f_rec'][i] for i in freq_list if i in loss_d])
        com_recall = np.mean([results['f_rec'][i] for i in common_list if i in loss_d])
        tail_recall = np.mean([results['f_rec'][i] for i in rare_list if i in loss_d])
        head_prec = np.mean([results['f_prec'][i] for i in freq_list if i in loss_d])
        com_prec = np.mean([results['f_prec'][i] for i in common_list if i in loss_d])
        tail_prec = np.mean([results['f_prec'][i] for i in rare_list if i in loss_d])
        f1 = 2 * results['f_rec'] * results['f_prec'] / (results['f_rec'] + results['f_prec'] + 1e-8)
        head_f1 = np.mean([f1[i] for i in freq_list if i in loss_d])
        com_f1 = np.mean([f1[i] for i in common_list if i in loss_d])
        tail_f1 = np.mean([f1[i] for i in rare_list if i in loss_d])

        exist_rec = np.array([results['f_rec'][i] for i in range(num_classes) if i in loss_d])
        exist_prec = np.array([results['f_prec'][i] for i in range(num_classes) if i in loss_d])


        log_dict = {'test_loss': ce_loss[3] / nums, 
                    'test_head_loss': head_loss, 'test_com_loss': com_loss, 'test_tail_loss': tail_loss,
                    'test_head_conf': head_conf, 'test_com_conf': com_conf, 'test_tail_conf': tail_conf,
                    'test_mean_rec': np.mean(exist_rec), 'test_mean_prec': np.mean(exist_prec),
                    'test_head_rec': head_recall, 'test_com_rec': com_recall, 'test_rare_rec': tail_recall,
                    'test_head_prec': head_prec, 'test_com_prec': com_prec, 'test_rare_prec': tail_prec,
                    'test_head_f1': head_f1, 'test_com_f1': com_f1, 'test_rare_f1': tail_f1,
                    'test_glb_acc': results1['f_acc'], 'test_edit': results1['edit'],
                    'test_glb_f1_10': results1['f1_10'], 'test_glb_f1_25': results1['f1_25'],
                    'test_glb_f1_50': results1['f1_50'],
                    'test_cls_f1_10': results['f1_10'],  'test_cls_f1_25': results['f1_25'], 'test_cls_f1_50': results['f1_50']}

        return results1['total_score'], log_dict,  confusion_matrix

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, actions_dict, sample_rate, device):

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            state = torch.load(model_dir + "/epoch-best" + ".model", map_location = device)
            self.model.load_state_dict(state['net'])

            import time

            time_start = time.time()
            for i, items in enumerate(test_loader):
                batch_input, batch_target, mask, vids = items
                vid = vids[0]
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                merge_results = self._merge_score(predictions[-1].transpose(2, 1).contiguous().view(-1, self.num_classes + self.nbins))
                _, predicted = torch.max(merge_results, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--seed', default='42')
parser.add_argument('--split', default='1')
parser.add_argument('--nbins', default='3', type=int)
parser.add_argument('--ratio', default='4', type=int)
parser.add_argument('--is_wandb', action='store_true', help='To log results on wandb')
parser.add_argument('--resume', default=0, type=int, help='do we resume form lastest saved model')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = int(args.seed)

TYPE = '/mstcn_bags_nbins{}_ratio{}_{}'.format(args.nbins, args.ratio, args.seed)

logpath = "results/" + args.dataset + "/split_{}/".format(args.split) + TYPE
logfile = logpath + '/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
log = Logger(logpath, logfile, fmt="[%(asctime)s - %(levelname)s]: %(message)s")
log.logger.info('########################## MS-TCN #####################################')

log.logger.info("Training for MS-TCN with Bags")
log.logger.info(args)


num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "../../../data/" + args.dataset + "/splits/train.split" + args.split + ".bundle"
vid_list_file_tst = "../../../data/" + args.dataset + "/splits/test.split" + args.split + ".bundle"
features_path = "../../../data/" + args.dataset + "/features/"
gt_path = "../../../data/" + args.dataset + "/groundTruth/"

mapping_file = "../../../data/" + args.dataset + "/mapping.txt"

model_dir = "./models/" + args.dataset + "/split_" + args.split + TYPE
results_dir = "./results/" + args.dataset + "/split_" + args.split + TYPE

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict, label_dict = dict(), dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
    label_dict[int(a.split()[0])] = a.split()[1]

num_classes = len(actions_dict)

import json

with open('../data/breakfast_frame_bin3.json', 'r') as f:
    group_dict = json.load(f)

freq_list = np.array([actions_dict[i] for i in group_dict['frequent']])
common_list = np.array([actions_dict[i] for i in group_dict['common']])
rare_list = np.array([actions_dict[i] for i in group_dict['rare']])

trainer = Trainer(MultiStageModel, log, sample_rate, num_stages = num_stages, num_layers = num_layers, num_f_maps = num_f_maps,
                  dim = features_dim, num_classes =num_classes)
if args.action == "train":
    if args.is_wandb:
        wandb.init(project='new_mstcn', entity='pang_neurips',
               name='mstcn_bags_nbins{}_ratio{}-split{}_{}'.format(args.nbins, args.ratio, args.split, args.seed),  resume = (args.resume >0))
    
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
    batch_gen_ad = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file_tst)

    train_loader = torch.utils.data.DataLoader(dataset=batch_gen, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
    train_ad_loader = torch.utils.data.DataLoader(dataset=batch_gen_ad, batch_size=1, shuffle=True, pin_memory=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=batch_gen_tst, batch_size=1, shuffle=True, pin_memory=False, num_workers=2)

    trainer.train(model_dir, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device,
                  actions_dict =actions_dict, batch_gen_tst=batch_gen_tst)
    if args.is_wandb:
        wandb.finish()

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file_tst)
    test_loader = torch.utils.data.DataLoader(dataset=batch_gen_tst, batch_size=1, shuffle=True, pin_memory=False, num_workers=2)

    if not os.path.exists(os.path.join(results_dir,'prediction')):
        os.makedirs(os.path.join(results_dir,'prediction'))
    trainer.predict(model_dir, os.path.join(results_dir,'prediction'), features_path, batch_gen_tst, actions_dict, sample_rate, device)

