import numpy as np
import matplotlib.pyplot as plt


class Record:
    def __init__(self, root):
        self.loss = ([], [])
        self.mse_loss = ([], [])
        self.kld_loss = ([], [])
        self.tfr = ([], [])
        self.kld_weight = ([], [])
        self.psnr = ([], [])
        self.cur_cnt = 0
        self.root = root

    def add(self, loss=None, mse_loss=None, kld_loss=None, tfr=None, kld_weight=None, psnr=None, step=1):
        self.cur_cnt += step
        if loss is not None:
            self.loss[0].append(self.cur_cnt)
            self.loss[1].append(loss)
        if mse_loss is not None:
            self.mse_loss[0].append(self.cur_cnt)
            self.mse_loss[1].append(mse_loss)
        if kld_loss is not None:
            self.kld_loss[0].append(self.cur_cnt)
            self.kld_loss[1].append(kld_loss)
        if tfr is not None:
            self.tfr[0].append(self.cur_cnt)
            self.tfr[1].append(tfr)
        if kld_weight is not None:
            self.kld_weight[0].append(self.cur_cnt)
            self.kld_weight[1].append(kld_weight)
        if psnr is not None:
            self.psnr[0].append(self.cur_cnt)
            self.psnr[1].append(psnr)

    def plot(self):
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(right=0.75)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(0, int(np.percentile(self.mse_loss[1] + self.kld_loss[1], 90)) + 1)
        loss, = ax1.plot(self.loss[0], self.loss[1], 'go', label='Loss', markersize=0.5)
        mse_loss, = ax1.plot(self.mse_loss[0], self.mse_loss[1], 'ro', label='Reconstruction Loss', markersize=0.5)
        kld_loss, = ax1.plot(self.kld_loss[0], self.kld_loss[1], 'bo',  label='Prior Loss', markersize=0.5)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Ratio/Weight')
        tfr, = ax2.plot(self.tfr[0], self.tfr[1], label='Teacher ratio' , linewidth=1, linestyle='dashed')
        kld_weight, = ax2.plot(self.kld_weight[0], self.kld_weight[1], label='KLD weight' , linewidth=1, linestyle='dashed')
        ax2.tick_params(axis='y')

        ax3 = ax1.twinx()
        ax3.spines.right.set_position(('axes', 1.2))
        ax3.set_ylabel('PSNR')
        psnr, = ax3.plot(self.psnr[0], self.psnr[1], label='PSNR' , linewidth=1)
        ax3.tick_params(axis='y')

        ax1.legend(loc='upper left')
        plt.legend(handles=[tfr, kld_weight, psnr], loc='upper right')
        plt.title('Training loss/ratio curve') 
        plt.savefig(f'{self.root}/iter_{self.cur_cnt:03}')
        plt.close("all")
        plt.close()
