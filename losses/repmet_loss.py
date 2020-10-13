import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functions import make_one_hot, euclidean_distance, cosine_distance


class RepmetLoss(nn.Module):

    def __init__(self, N, k, emb_size, alpha=1.0, sigma=0.5, dist='euc'):
        super(RepmetLoss, self).__init__()
        self.N = N
        self.k = k
        self.emb_size = emb_size
        self.alpha = alpha
        self.sigma = sigma
        self.dist = dist

        # TODO mod this from hardcoded with the device
#        self.reps = nn.Parameter(F.normalize(torch.randn(N*k, emb_size, dtype=torch.float).cuda()))
        self.reps = nn.Parameter(F.normalize(torch.randn(N*k, emb_size, dtype=torch.float)).cuda())
    def forward(self, input, target):
        """
        Equation (4) of repmet paper

        :param dists: n_samples x n_classes x n_k
        :param labels: n_samples
        :param alpha:
        :return:
        """

        input = input.cuda()
        target = target.cuda()
        # batch size
        self.n_samples = len(target)

        # self.reps.data = F.normalize(self.reps)

        # distances = euclidean_dist(input, F.normalize(self.reps))  # todo normalize the reps before dist? default no
        if self.dist == 'cos':
            distances = cosine_distance(input, self.reps)
        else:
            distances = euclidean_distance(input, self.reps)


        # make mask with ones where correct class, zeros otherwise
        mask, mask_inc_ = make_one_hot(target, n_classes=self.N)
        mask = mask.cuda()
        mask_inc_ = mask_inc_.cuda()
       	mask_cor = mask.transpose(0, 1).repeat(1, self.k).view(-1, self.n_samples).transpose(0, 1)
        #mask_inc = ~mask_cor
        mask_inc = mask_inc_.transpose(0,1).repeat(1, self.k).view(-1, self.n_samples).transpose(0,1)
        valmax, argmax = distances.max(-1)
        valmax, argmax = valmax.max(-1)
        valmax += 10

        cor = distances + mask_inc.float()*valmax
        inc = distances + mask_cor.float()*valmax
        min_cor, _ = cor.min(1)
        min_inc, _ = inc.min(1)

        # Eqn. 4 of repmet paper
        losses = F.relu(min_cor - min_inc + self.alpha)

        # mean the sample losses over the batch
        total_loss = torch.mean(losses)

        # Eqn. 1 of repmet paper
        probs = torch.exp(- (distances**2) / (2 * self.sigma ** 2))  # todo is the dist meant to be squared?

        # Eqn 2. of repmet paper
        hard_probs, _ = probs.view(-1, self.N, self.k).max(2)

        # Eqn 3. of repmet paper
        back_p = 1 - hard_probs.max(1)[0]  # todo useful for detection

        # classification (soft) version of eqn 2 (considers all the ks for a class) (eqn 5 of repmet paper)
        numerator = probs.view(-1, self.N, self.k).sum(2)
        denominator = numerator.sum(1).view(-1, 1)

        epsilon = 1e-8

        soft_probs = numerator / (denominator + epsilon) + epsilon

        _, pred = soft_probs.max(1)
        acc = pred.eq(target.squeeze().long()).float().mean()

        # calc a second acc.. just the smallest distances, equiv if k=1
        d = distances.view(-1, self.N, self.k)
        dm, _ = d.min(2)
        _, dma = dm.min(1)
        acc_b = dma.eq(target.squeeze().long()).float().mean()

        return total_loss, losses, pred, acc

    def get_reps(self):
        return self.reps.data.cpu().detach().numpy()

    def set_reps(self, reps, start=None, stop=None):
        if start is not None and stop is not None:
            self.reps.data[start:stop] = torch.Tensor(reps).cuda().float()
        else:
            # self.reps = nn.Parameter(torch.Tensor(reps, dtype=torch.float).cuda())
            self.reps.data = torch.Tensor(reps).cuda().float()

if __name__ == "__main__":
    print("Simple test of emb loss")
    repmet = RepmetLoss(N=3, k=2, emb_size=2)

    p = [[1, 0, 0],
         [1, 0, 0],
         [1, 0, 0],
         [.25, .5, .25],
         [.75, 0, .25]]
    l = [1, 0, 0, 1, 2]
#    d = [[[1, 0.00], [1, 0.00], [1, 0.00]], #C0K0, C0K1, C1K0 ... C2K1
#         [[0.001, 0.001], [1, 1], [1, 1]],
#         [[0.001, 0.001], [1, 1], [1, 1]],
#         [[0.001, 0.002], [1, 1], [1, 1]],
#         [[.6, 1], [.6, 1], [.5, 0.001]]]

    d = [[1, 0.0],
         [0.001, 0.001],
         [0.001, 1],
         [0.001, 0.002],
         [.6, 1]]

    d = torch.autograd.Variable(torch.Tensor(d), requires_grad=True)
    l = torch.autograd.Variable(torch.Tensor(l), requires_grad=True)
    total_loss, losses, pred, acc  = repmet(d, l)

    print("Total Loss: {}".format(total_loss))
    print("losses: {}".format(losses))
    print("pred: {}".format(pred))
    print("acc: {}".format(acc))
    print('done')
