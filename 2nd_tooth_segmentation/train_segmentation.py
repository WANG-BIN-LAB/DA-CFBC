import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('networks')))
from single_tooth_segmentation.networks.DAUNet_2nd import daunet_tiny as DAUNet_3rd
from utils.losses import dice_loss
from dataloaders.singeToothLoader import singeToothLoader, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, LabelCrop, DataScale

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/../', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='Main_Results', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "/home/data2//DA-CFBC/models/" + args.exp + "/"
log_path = "/home/data2//DA-CFBC/models/" + args.exp + "/" + 'log'
pth_path = "/home/data2//DA-CFBC/models/" + args.exp + "/" + 'pth'
copy_src_path = "/home/data2//DA-CFBC/single_tooth_segmentation/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (48, 48, 64)
num_classes = 3

seg_criterion = torch.nn.BCELoss()

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree(copy_src_path, snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"log8080128_2clsf.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    net = DAUNet_3rd()
    net.cuda()

    db_train = singeToothLoader(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                           #LabelCrop(),
                           #RandomRotFlip(),
                           ToTensor()
                       ]))
    db_test = singeToothLoader(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           ToTensor()
                       ]))
                       
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(log_path)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch, bd_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['boundary']
            volume_batch, label_batch, boundary_batch = volume_batch.cuda(), label_batch.cuda(),bd_batch.cuda()
            case_name = sampled_batch['case_name']
            print('the shape:', volume_batch.shape)
            outputs_seg,outputs_bd = net(volume_batch)

            loss_seg = F.cross_entropy(outputs_seg, label_batch)
            outputs_soft = F.softmax(outputs_seg, dim=1)
            outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)
            dice_losses = []
            for i in range(num_classes):
                target_i = (label_batch == i).float() 
                dice_loss_i = dice_loss(outputs_soft[:, i, :, :, :], target_i)
                dice_losses.append(dice_loss_i)
            loss_seg_dice = torch.mean(torch.stack(dice_losses))

            loss_bd = F.cross_entropy(outputs_bd, boundary_batch)
            outputs_bd_soft = F.softmax(outputs_bd, dim=1)
            outputs_bd = torch.argmax(outputs_bd_soft, dim=1, keepdim=True)
            dice_losses = []
            for i in range(num_classes):
                target_i = (boundary_batch == i).float()  
                dice_loss_i = dice_loss(outputs_bd_soft[:, i, :, :, :], target_i)
                dice_losses.append(dice_loss_i)
            loss_bd_dice = torch.mean(torch.stack(dice_losses))
            loss = 0.65*(0.6*loss_seg_dice + 0.4*loss_seg) + 0.35*(0.6*loss_bd_dice+0.4*loss_bd )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_bd', loss_bd, iter_num)
            writer.add_scalar('loss/loss_bd_dice', loss_bd_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('dice/dice_seg', 1-loss_seg_dice.item(), iter_num)
            writer.add_scalar('dice/dice_bd', 1-loss_bd_dice.item(), iter_num)
            dice_mean = ((1-loss_seg_dice.item())+(1-loss_bd_dice.item()))/2
            writer.add_scalar('dice/dice_mean', dice_mean, iter_num)
            logging.info('iteration %d : loss : %f, %f' % (iter_num, loss.item(), loss_seg_dice.item()))
            logging.info('iteration %d : dice : %f' % (iter_num, dice_mean))

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(pth_path, 'iter_single_2cls' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
            if iter_num % 1000 == 0:
                net.eval()
                test_loss = 0
                iter_test = 0
                for i_batch, sampled_batch in enumerate(testloader):
                    volume_batch, label_batch, bd_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['boundary']
                    volume_batch, label_batch, boundary_batch = volume_batch.cuda(), label_batch.cuda(),bd_batch.cuda()
                    with torch.no_grad():
                        outputs_seg,outputs_bd = net(volume_batch)
                        
                        loss_seg = F.cross_entropy(outputs_seg, label_batch)
                        outputs_soft = F.softmax(outputs_seg, dim=1)
                        outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)
                        dice_losses = []
                        for i in range(num_classes):
                            target_i = (label_batch == i).float()  
                            dice_loss_i = dice_loss(outputs_soft[:, i, :, :, :], target_i)
                            dice_losses.append(dice_loss_i)
                        loss_seg_dice = torch.mean(torch.stack(dice_losses))

                        loss_bd = F.cross_entropy(outputs_bd, boundary_batch)
                        outputs_bd_soft = F.softmax(outputs_bd, dim=1)
                        outputs_bd = torch.argmax(outputs_bd_soft, dim=1, keepdim=True)
                        dice_losses = []
                        for i in range(num_classes):
                            target_i = (boundary_batch == i).float()  
                            dice_loss_i = dice_loss(outputs_bd_soft[:, i, :, :, :], target_i)
                            dice_losses.append(dice_loss_i)
                        loss_bd_dice = torch.mean(torch.stack(dice_losses))

                        loss = 0.65*(0.6*loss_seg_dice + 0.4*loss_seg) + 0.35*(0.6*loss_bd_dice+0.4*loss_bd )
                        dice_mean = ((1-loss_seg_dice.item())+(1-loss_bd_dice.item()))/2

                        print('---val for seg:', 1 - loss_seg_dice.item())
                        print('---val for bd:', 1 - loss_bd_dice.item())
                        print('---val for mean:', dice_mean)
                        test_loss = test_loss + loss
                        iter_test = iter_test + 1
                writer.add_scalar('loss_test/test_loss', test_loss/iter_test, iter_num)
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                logging.info('iteration %d : val for seg : %f' % (iter_num, 1-loss_seg_dice.item()))
                logging.info('iteration %d : val for mean : %f' % (iter_num, dice_mean))
                logging.info('iteration %d : val for bd : %f' % (iter_num, 1-loss_bd_dice.item()))
                net.train()
                del volume_batch, label_batch, bd_batch, loss_seg, outputs_soft, loss_seg_dice, loss_bd, outputs, outputs_bd, outputs_bd_soft, dice_mean    
        if iter_num > max_iterations:
            break
        
    save_mode_path = os.path.join(pth_path, 'iter_single_2cls'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()