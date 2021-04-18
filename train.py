'''
This script handling the training process.
'''
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import argparse

import torch.optim as optim
import torch.utils.data

import util.cameras as cameras

from util.utils import *
from util.evaluation import calcualte_mmd
from model import *

parser = argparse.ArgumentParser()


parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-test_batch_size',type=int,default=20)
parser.add_argument('-dropout', type=float, default=0.1)

parser.add_argument('-actions', type=str, default='all')
parser.add_argument('-n_class', type=int, default=10)
parser.add_argument('-cameras_path', type=str, default='data/h36m/cameras.h5')
parser.add_argument('-dataset', type=str, default='data/h36m/')
parser.add_argument('--n_joint', type=int, default=15,
                     help='num of joint for in one frame')
parser.add_argument('-seq_length', type=int, default=50)
parser.add_argument('-vec_length', type=int, default=30)
parser.add_argument('-tmp_dir', type=str, default='tmp/')
parser.add_argument('-video_dir', type=str, default='video/')
parser.add_argument('-lr', type=float, default=0.0002)
parser.add_argument('--cuda', type=int, default=1, help='set -1 when you use cpu')
parser.add_argument('--niter', type=int, default=100000,
                     help='set num of iterations, default: 100000')




opt = parser.parse_args()
device = torch.device('cuda' if opt.cuda else 'cpu')
opt.device = device
n_iter     = opt.niter
batch_size = opt.batch_size
cuda       = opt.cuda
opt.total_joint = opt.n_joint * opt.seq_length



''' calculate adjacency A matrix for the wwhole sequence '''
A_seq = torch.eye(opt.n_joint).repeat(opt.seq_length,opt.seq_length) #connections out-frame

# each character in Human3.6 has 15 joints
# 1 in m row and n column means m-th joint is connected with n-th joint
A = torch.tensor([ [1,1,0,1,0,0,1,0,0,0,0,0,0,0,0],
                   [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [1,0,0,1,1,0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                   [1,0,0,1,0,0,1,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,0,0,1,0,0],
                   [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,1,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0,0,0,1,1,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]])

A_i = torch.zeros((opt.total_joint,opt.total_joint)) #connections in-frame

for i in range(opt.seq_length):
    A_i[i * opt.n_joint:(i + 1) * opt.n_joint, i * opt.n_joint:(i + 1) * opt.n_joint] = A

A_seq = A_seq - torch.eye((opt.n_joint*opt.seq_length))

''' parameter '''
seed = 0
opt.dropout_rate = opt.dropout
opt.d_E = 30
opt.hidden_size = 100

k_frame = 20 #sample 20 frames for img discriminator
opt.n_head = 3  #for self attention
opt.seq_connect = 5 #connect past 5 frames
opt.A_seq = A_seq
opt.A_i = A_i
opt.d_E = 30



np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


criterion = nn.BCELoss()
criterion_c = nn.CrossEntropyLoss()

dis_i = Discriminator_I(k_frame=k_frame).to(device)
dis_v = Discriminator_V().to(device)
gen = Generator(opt = opt).to(device)

betas=(0.5, 0.999)
optim_Di  = optim.Adam(dis_i.parameters(), lr=opt.lr, betas=betas)
optim_Dv  = optim.Adam(dis_v.parameters(), lr=opt.lr, betas=betas)
optim_G   = optim.Adam(gen.parameters(), lr=opt.lr, betas=betas)

def h36m_data_loader(batch_size, dataset, sequence_length, actions,opt):
    counter = 0
    batch = []
    class_batch = []
    label = []
    while 1:
        for k in dataset.keys():

            if k[1] in actions:
                class_vec = [1.0 if k[1] == action_name else 0.0 for action_name in actions]
                video = dataset[k]
                start = random.randint(300, video.shape[0] - 1 - opt.seq_length * 3)
                clip = video[start:start + opt.seq_length * 3, :]
                idxs = np.arange(0, opt.seq_length * 3, 3)
                clip = clip[idxs]

                counter += 1
                batch.append(clip)
                index = class_vec.index(1.0)
                class_batch.append(index)
                label.append(class_vec)

                if counter >= batch_size:
                    batch = np.array(batch)
                    class_batch = np.array(class_batch)
                    label = np.array(label)
                    yield batch, class_batch,label
                    counter = 0
                    batch = []
                    class_batch = []
                    label = []

def pre_data():
    actions = data_utils.define_actions(opt.actions)
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(opt.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data_corrected(
        actions, opt.dataset, rcams)
    print("done reading and normalizing data.")
    tr_loader = h36m_data_loader(opt.batch_size, train_set_2d, opt.seq_length, actions, opt)
    te_loader = h36m_data_loader(opt.test_batch_size, train_set_2d, opt.seq_length, actions, opt)
    return tr_loader,te_loader,data_mean_2d, data_std_2d, dim_to_ignore_2d


def bp_v(inputs,label,label_num,y,retain=False):
    y_label = y * torch.ones(batch_size, 1).to(device).type(torch.float32)
    label_v = label.unsqueeze(1).repeat(1,opt.seq_length,1)
    outputs = dis_v(inputs,label_v)
    err = criterion(outputs, y_label)
#     err = criterion(outputs, y_label)+criterion_c(output2,label_num)
    err.backward(retain_graph=retain)
    return err.item()

def bp_i(inputs,label,y,retain=False):
    y_label = y * torch.ones(batch_size, 1).to(device).type(torch.float32)
    label_v = label.unsqueeze(1).repeat(1, k_frame+1, 1)
    outputs = dis_i(inputs, label_v)
    err = criterion(outputs,y_label)
    err.backward(retain_graph=retain)
    return err.item()



''' train models '''
tr_loader, te_loader,data_mean_2d, data_std_2d, dim_to_ignore_2d = pre_data()
h_avg,h_joint = 1.0,1.0
idx = 0
test_sum = 0
for epoch in range(1, n_iter + 1):
    gen.train()
    ''' prepare real images '''
    batch_images, batch_class, label = tr_loader.__next__()
    batch_images = torch.tensor(batch_images, device=device).type(torch.float32)
    batch_class = torch.tensor(batch_class,device=device).type(torch.long)
    label = torch.tensor(label,device=device).type(torch.float32)
    ''' prepare fake images '''
    fake_actions = gen(label.to(device))
    fake_imgs = fake_actions[:,np.random.randint(0,opt.seq_length),:].unsqueeze(1)
    for _ in range(k_frame):
        fake_img = fake_actions[:,np.random.randint(0,opt.seq_length),:].unsqueeze(1)
        fake_imgs = torch.cat((fake_imgs,fake_img),1)
    real_imgs = batch_images[:,np.random.randint(0,opt.seq_length),:].unsqueeze(1)
    for _ in range(k_frame):
        real_img = batch_images[:,np.random.randint(0,opt.seq_length),:].unsqueeze(1)
        real_imgs = torch.cat((real_imgs,real_img),1)

    ''' train discriminators '''
    dis_v.zero_grad()
    err_v_real = bp_v(batch_images,label.detach(),batch_class.detach(),0.9)
    err_v_fake = bp_v(fake_actions.detach(),label.detach(),batch_class.detach(),0)
    optim_Dv.step()
    err_Dv = err_v_fake + err_v_real


    dis_i.zero_grad()
    err_i_real = bp_i(real_imgs,label.detach(),0.9)
    err_i_fake = bp_i(fake_imgs.detach(),label,0)
    optim_Di.step()
    err_Di = err_i_real + err_i_fake
    ''' train generators '''

    gen.zero_grad()
    err_gv = bp_v(fake_actions,label.detach(),batch_class.detach(),0.9,True)
    err_gi = bp_i(fake_imgs,label.detach(),0.9)
    optim_G.step()

    if epoch % 200 == 0:
        # print(test_sum)
        test_sum = 0
        d_loss = err_v_real + err_v_fake + err_i_real + err_i_fake
        g_loss = err_gv + err_gi
        print('[%d/%d] d_loss:%.4f g_loss:%.4f' % (epoch, n_iter, d_loss, g_loss))

        gen.eval()
        fake_actions_batch = []
        real_actions_batch = []
        label_batch = []

        for i in range(50):
            batch_images_test, batch_class_test, label_test = te_loader.__next__()
            batch_images_test = torch.tensor(batch_images_test, device=device).type(torch.float32)
            batch_class = torch.tensor(batch_class_test, device=device).type(torch.float32)
            label_test = torch.tensor(label_test, device=device).type(torch.float32)
            ''' prepare fake images '''
            fake_actions = gen(label_test.to(device))
            fake_actions_batch.extend(fake_actions.detach().cpu().numpy())
            real_actions_batch.extend(batch_images_test.detach().cpu().numpy())
            label_batch.extend(label_test.detach().cpu().numpy())

        avg, joint = calcualte_mmd(np.asarray(fake_actions_batch), np.asarray(real_actions_batch),
                                   np.asarray(label_batch))
        if joint<h_joint:
            h_joint = joint
            idx = epoch
            ''' save checkpoints '''
            #torch.save(gen.state_dict(), './ckpt/%.4fgen.pt' % (joint))
            #torch.save(dis_i.state_dict(), './ckpt/%.4fdisi.pt' % (joint))
            #torch.save(dis_i.state_dict(), './ckpt/%.4fdisv.pt' % (joint))

            ''' generate video '''
            if epoch>300000:
                outcome = fake_actions.to(torch.device('cpu')).detach().numpy()
                for i in range(opt.batch_size):
                    class_num = int((i % 20) / 2)
                    video(np.squeeze(outcome[i]), opt.tmp_dir,
                          os.path.join(opt.video_dir, 'train_%02d_%04d' % (epoch, 0), 'class_%02d' % class_num),
                          '%04d' % (i // 10),
                          data_mean_2d, data_std_2d, dim_to_ignore_2d)


        print('current joint mmd: %.4f history mmd: %.4f (epoch %d)'%(joint,h_joint,idx))




















