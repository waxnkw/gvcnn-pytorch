import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable


class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn,
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)

                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward()
                self.optimizer.step()

                if (i + 1) % 30 == 0:
                    print('epoch %d, step [%d/%d]: train_loss %.3f; train_acc %.3f'
                          % (epoch + 1, i + 1, len(self.train_loader), loss, acc))
            i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch % 10)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(21)
        samples_class = np.zeros(21)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)

            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        val_class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        val_class_acc = [(i, val_class_acc[i]) for i in range(21)]
        print('val class acc. :', val_class_acc)
        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
