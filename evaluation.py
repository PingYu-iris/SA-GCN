import numpy as np
from tqdm import tqdm
import torch


import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class MMD:  # class to compute MMD between two distributions

    def __init__(self, mode,use_torch):
        #  mode: 'avg' or 'joint', 'avg' computes the frame-average MMD, 'joint' regard a sequence as joint distribution
        self.mode = mode
        self.use_torch = use_torch

    def reset(self, new_mode):  # reset the status of MMD
        self.mode = new_mode

    def rkhs_mmd(self, samples_1, samples_2, bandwidth):  # two given sample groups, shape (N*dim)
        if self.use_torch:
            m, dim = samples_1.size()
            n, _ = samples_2.size()
        else:
            m, dim = np.shape(samples_1)
            n, _ = np.shape(samples_2)

        def rbf_kernel(z_1, z_2, bandwidth):
            if self.use_torch:
                z_1_expand = z_1.unsqueeze(1)
                dist_square = ((z_1_expand - z_2).pow(2.)).sum(-1)
                kernel_matrix = (-dist_square/bandwidth).exp()
            else:
                z_1_expand = np.expand_dims(z_1, axis=1)
                dist_square = np.sum((z_1_expand - z_2)**2, axis=-1)
                kernel_matrix = np.exp(-dist_square/bandwidth)
            return kernel_matrix
        kxx = rbf_kernel(samples_1, samples_1, bandwidth)
        kyy = rbf_kernel(samples_2, samples_2, bandwidth)
        kxy = rbf_kernel(samples_1, samples_2, bandwidth)
        hxy = kxx + kyy - 2*kxy

        if self.use_torch:
            mmd_ = ((hxy - hxy.diag().diag()).sum()/(m*(m-1))).pow(0.5)
            del kxx, kyy, kxy, hxy
            torch.cuda.empty_cache()
            return mmd_.item()
        else:
            mmd_ = np.sqrt(np.sum(hxy - np.diag(np.diag(hxy)))/(m*(m-1)))
            return mmd_

    def compute_sequence_mmd(self, sequence_1, sequence_2, bandwidth):  # compute the mmd between sequences, shape (N*len*dim)
        if self.use_torch:
            _, seq_len, dim = sequence_1.size()
        else:
            _, seq_len, dim = np.shape(sequence_1)
        result = 0.
        if self.mode == 'avg':
            for frames in range(seq_len):
                result += self.rkhs_mmd(sequence_1[:, frames, :], sequence_2[:, frames, :], bandwidth)/seq_len
        elif self.mode == 'joint':
            if self.use_torch:
                flat_seq_1 = sequence_1.view(-1, dim*seq_len)
                flat_seq_2 = sequence_2.view(-1, dim*seq_len)
            else:
                flat_seq_1 = np.reshape(sequence_1, (-1, dim*seq_len))
                flat_seq_2 = np.reshape(sequence_2, (-1, dim*seq_len))
            result = self.rkhs_mmd(flat_seq_1, flat_seq_2, bandwidth)
        else:
            raise Exception('undefined mode')
        return result


def calcualte_mmd(gen,real,label):
    use_torch = 0
    class_num = np.shape(label)[-1]
    gen_data_list = [[] for i in range(class_num)]
    real_data_list = [[] for j in range(class_num)]
    # mode = "avg"
    # compute_mmd = MMD(mode,use_torch)
    # # for i in tqdm(range(len(gen)), total=len(gen)):
    # for i in range(len(gen)):
    #     cl = np.argmax(label[i])
    #     if len(gen_data_list[cl]) < 2000:  # NOTE: joint mode can not afford to large matrix
    #         gen_data_list[cl].append(gen[i])
    #         real_data_list[cl].append(real[i])

    # result_list = []
    # for i in range(class_num):
    #     new_r = 0
    #     new_gen = np.asarray(gen_data_list[i])
    #     new_real = np.asarray(real_data_list[i])
    #     if use_torch:
    #         new_gen = torch.tensor(new_gen).cuda()
    #         new_real = torch.tensor(new_real).cuda()
    #     # for j in tqdm(range(-4, 10), total=14):
    #     for j in range(-4,10):
    #         new_new_r = compute_mmd.compute_sequence_mmd(new_gen, new_real, 10 ** j)
    #         if new_new_r > new_r:
    #             new_r = new_new_r
    #     # print(new_r)
    #     result_list.append(new_r)
    #     del new_real, new_gen
    #     torch.cuda.empty_cache()
    # avg = np.mean(result_list)

    avg = 0

    mode = "joint"
    compute_mmd = MMD(mode,use_torch)
    # for i in tqdm(range(len(gen)), total=len(gen)):
    for i in range(len(gen)):
        cl = np.argmax(label[i])
        if len(gen_data_list[cl]) < 2000:  # NOTE: joint mode can not afford to large matrix
            gen_data_list[cl].append(gen[i])
            real_data_list[cl].append(real[i])

    result_list = []
    for i in range(class_num):
        new_r = 0
        new_gen = np.asarray(gen_data_list[i])
        new_real = np.asarray(real_data_list[i])
        if use_torch:
            new_gen = torch.tensor(new_gen).cuda()
            new_real = torch.tensor(new_real).cuda()
        # for j in tqdm(range(-4, 10), total=14):
        for j in range(-4,10):
            new_new_r = compute_mmd.compute_sequence_mmd(new_gen, new_real, 10 ** j)
            if new_new_r > new_r:
                new_r = new_new_r
        # print(new_r)
        result_list.append(new_r)
        del new_real, new_gen
        torch.cuda.empty_cache()
    joint = np.mean(result_list)



    return avg,joint



# path = input('Input the path containing the result data: ')
# mode = input('Input the MMD computing mode: ')
# use_torch = int(input('Input use torch or not(1/0): '))
# mode = "joint"
# use_torch = 0
# data = np.load('actions.npz')
# gen = data['GEN'].astype(np.float32)
# real = data['REAL'].astype(np.float32)
# label = data['LABEL'].astype(np.float32)
# class_num = np.shape(label)[-1]
# gen_data_list = [[] for i in range(class_num)]
# real_data_list = [[] for j in range(class_num)]
# compute_mmd = MMD(mode)
# for i in tqdm(range(len(gen)), total=len(gen)):
#     cl = np.argmax(label[i])
#     if len(gen_data_list[cl]) < 2000:  # NOTE: joint mode can not afford to large matrix
#         gen_data_list[cl].append(gen[i])
#         real_data_list[cl].append(real[i])
#
# result_list = []
# for i in range(class_num):
#     new_r = 0
#     new_gen = np.asarray(gen_data_list[i])
#     new_real = np.asarray(real_data_list[i])
#     if use_torch:
#         new_gen = torch.tensor(new_gen).cuda()
#         new_real = torch.tensor(new_real).cuda()
#     print('Label class %i, generated data shape %s, real data shape %s, ' % (i, np.shape(new_gen), np.shape(new_real)))
#     for j in tqdm(range(-4, 10), total=14):
#         new_new_r = compute_mmd.compute_sequence_mmd(new_gen, new_real, 10**j)
#         if new_new_r > new_r:
#             new_r = new_new_r
#     print(new_r)
#     result_list.append(new_r)
#     del new_real, new_gen
#     torch.cuda.empty_cache()
#
# print('result list is ', result_list)
# print('final mmd is ', np.mean(result_list))
# # print(path, mode)
