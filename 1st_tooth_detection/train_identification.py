import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import pynvml
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


from networks.DAUNet_1st import daunet_tiny as DAUNet_1st

from utils.losses import dice_loss
from dataloaders.toothLoader import toothLoader, TverskyLoss, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, LabelCrop, DataScale

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='Main_Result_', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=150000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path

snapshot_path = "/home/data2/DA-CFBC/model/" + args.exp + "/Main_Result_"
log_path = "/home/data2/DA-CFBC/model/" + args.exp + "/" + 'log_Main_Result/'
pth_path = "/home/data2/DA-CFBC/model/" + args.exp + "/" + 'pth_Main_Result/'
copy_src_path = "/home/data2/DA-CFBC/tooth_detection/"

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

patch_size = (256, 256, 256)
num_classes = 3

reg_criterion = torch.nn.L1Loss()

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

    logging.basicConfig(filename=snapshot_path+"_Main_Result_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = DAUNet_1st()
    net = net.cuda()
    
    db_train = toothLoader(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                           #RandomCrop(patch_size),
                           #DataScale(),
                           ToTensor()
                       ]))
    db_test = toothLoader(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           #RandomCrop(patch_size),
                           #DataScale(),
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
    alpha = 0.5
    beta = 0.5
    tversky_loss = TverskyLoss(alpha, beta)
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            image_name, volume_batch, label_batch, edge_batch =sampled_batch['image_name'], sampled_batch['image'], sampled_batch['label'], sampled_batch['edge']
            offset_batch = sampled_batch['offset_cnt']
            file_name = os.path.basename(str(image_name[0]))
            unique_values = np.unique(label_batch)
           
            volume_batch, offset_batch, label_batch, edge_batch = volume_batch.cuda(), offset_batch.cuda(), label_batch.cuda(),edge_batch.cuda()
            outputs_seg,outputs_off= net(volume_batch)

            label_batch[label_batch > 0.5] = 1
            loss_seg = F.cross_entropy(outputs_seg, label_batch)
            outputs_soft = F.softmax(outputs_seg, dim=1)
            outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss_off = (reg_criterion(outputs_off[0, :, label_batch[0, :, :, :]==1], offset_batch[0, :, label_batch[0, :, :, :]==1])+
                        reg_criterion(outputs_off[1, :, label_batch[1, :, :, :]==1], offset_batch[1, :, label_batch[1, :, :, :]==1]))/2
            

            mse_loss = F.mse_loss(outputs_off[:, :, label_batch[0, :, :, :]==1], offset_batch[:, :, label_batch[0, :, :, :]==1], reduction='mean')
            smooth_l1_loss = F.smooth_l1_loss(outputs_off[:, :, label_batch[0, :, :, :]==1], offset_batch[:, :, label_batch[0, :, :, :]==1], reduction='mean', beta=1.0)

            print(file_name,'test for seg:', 1-loss_seg_dice.item())
            loss = 0.66*(0.8*loss_seg_dice + 0.2*loss_seg) + 0.34*loss_off  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss_seg_dice/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss_off/loss_off', loss_off, iter_num)
            writer.add_scalar('seg_dice/seg_dice', 1-(loss_seg_dice.item()), iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : test for seg : %f' % (iter_num, 1-loss_seg_dice.item()))
            logging.info('iteration %d : test for off : %f' % (iter_num, loss_off.item()))
            

            if iter_num % 1000 == 0:
                outputs_np = outputs[0, 0, :, :, :].cpu().detach().numpy().astype(np.float32)
                volume_batch_np = volume_batch[:, 0, :, :].cpu().detach().numpy().astype(np.float32)
                img = nib.Nifti1Image(volume_batch_np[0], affine=np.eye(4))  # 使用一个单位矩阵作为仿射矩阵
                pred = nib.Nifti1Image(outputs_np, affine=np.eye(4))
  
                save_path = "/home/data2/DA-CFBC/tooth_detection/visualize_cnt/"
                img_filename = f'{save_path}img_{iter_num}.nii.gz'
                pred_filename = f'{save_path}pred_{iter_num}.nii.gz'

                img.to_filename(img_filename)
                pred.to_filename(pred_filename)

            del volume_batch, offset_batch,  label_batch, loss_seg, outputs_soft, loss_seg_dice, loss_off
            ## change lr
            if iter_num % 4000 == 0 and iter_num < 8001:
                lr_ = base_lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            #if iter_num % 10000 == 0:
            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(pth_path, 'iter_cnt' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_cnt' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
            
            if iter_num % 600 == 0:
                net.eval()
                test_loss = 0
                iter_test = 0
                for i_batch, sampled_batch in enumerate(testloader):
                    volume_batch, offset_batch,  label_batch = sampled_batch['image'], sampled_batch['offset_cnt'], sampled_batch['label']
                    volume_batch, offset_batch,  label_batch = volume_batch.cuda(), offset_batch.cuda(),  label_batch.cuda()
                    with torch.no_grad():
                        outputs_seg, outputs_off = net(volume_batch)

                        label_batch[label_batch > 0.5] = 1
                        loss_seg = F.cross_entropy(outputs_seg, label_batch)
                        outputs_soft = F.softmax(outputs_seg, dim=1)
                        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
                        loss_off = reg_criterion(outputs_off[:, :, label_batch[0, :, :, :]==1], offset_batch[:, :, label_batch[0, :, :, :]==1])

                        loss = (loss_seg_dice + loss_seg) + loss_off 

                        print('---test for seg:', 1 - loss_seg_dice.item())
                        test_loss = test_loss + loss
                        iter_test = iter_test + 1
                writer.add_scalar('loss_test/test_loss', test_loss/iter_test, iter_num)
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                logging.info('iteration %d : val for seg : %f' % (iter_num, 1-loss_seg_dice.item()))
                logging.info('iteration %d : val for off : %f' % (iter_num, loss_off.item()))
                net.train()
                del volume_batch, offset_batch, label_batch, loss_seg, outputs_soft, loss_seg_dice, loss_off
            
                                
        if iter_num > max_iterations:
            break
        
    save_mode_path = os.path.join(pth_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()