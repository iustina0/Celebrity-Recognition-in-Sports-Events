from __future__ import  absolute_import
import os
import matplotlib.pyplot as plt
import ipdb
from tqdm import tqdm

from utils.config import opt
from data.dataset import TrainDataset, TestDataset
from model.fasterRCNN import FasterRCNN
from torch.utils.data import DataLoader
from model.trainer import FasterRCNNTrainer
from utils.array_util import scalar
from utils.eval_tool import eval_detection_voc
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(dataloader, faster_rcnn, test_num=1000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    print('load data')
    train_dataset = TrainDataset(opt)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)

    test_set = TestDataset(opt)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=opt.test_num_workers, shuffle=False, pin_memory=True)
    faster_rcnn = FasterRCNN()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    best_map = 0
    lr_ = opt.lr

    loss_values = {}
    total_loss = []
    test_map = []

    for k in trainer.get_meter_data().keys():
        if k != 'total_loss':
            loss_values[k] = []


    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                for k, v in trainer.get_meter_data().items():
                    if v is not None and k != 'total_loss':
                        loss_values[k].append(v)
                    elif v is not None and k == 'total_loss':
                        total_loss.append(v)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        test_map.append(eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break

    plt.figure()
    plt.plot(test_map, label='test_map')
    plt.legend()
    plt.savefig('./misc/test_map_plot.png')

    plt.figure()
    plt.plot(total_loss, label='total_loss')
    plt.legend()
    plt.savefig('./misc/total_loss_plot.png')

    plt.figure()
    for loss_name, values in loss_values.items():
        plt.plot(values, label=loss_name)
    plt.legend()
    plt.savefig('./misc/loss_values_plot.png')


if __name__ == '__main__':
    import fire

    fire.Fire()
