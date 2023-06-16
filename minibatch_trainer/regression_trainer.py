import torch
from minibatch_trainer.minibatch_trainer import Trainer, name_to_evaluators, names_to_criterions


class RegressionTrainer(Trainer):
    '''
    Trainer for multitask regression tasks for graph prediction
    '''

    def __init__(self, model, optimizer, data, train_loader, val_loader, test_loader, device, 
        epochs, log_steps, checkpoint_dir, degrees, degree_thres, task_idxes,
        criterion="regression", evaluator="mse", monitor="avg", decoupling=False, lr_scheduler=None, 
        mnt_mode = "min", eval_separate=False):
        super().__init__(model, optimizer, data, train_loader, test_loader, device, epochs, 
        log_steps, checkpoint_dir, degrees, degree_thres, criterion, evaluator, monitor, decoupling)
        self.task_idxes = torch.LongTensor(task_idxes).to(device)
        self.task_num = len(self.task_idxes)
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.eval_separate = eval_separate
        self.mnt_mode = mnt_mode

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0; steps = 0
        # The model is assumed to be GINE
        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            outputs = self.model(data)
            labels = data.y

            labels = labels[:, self.task_idxes]
            is_labeled = ~torch.isnan(labels)
            loss = names_to_criterions[self.criterion](outputs[:, self.task_idxes][is_labeled], labels[is_labeled])

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item(); steps += 1
        
        return total_loss / steps

    def train(self):
        best_val_error = test_error = 1e10 if self.mnt_mode == "min" else -1e10

        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch(epoch)
            log = self.test()
            train_error, valid_error, tmp_test_error = \
                log[f"train_{self.evaluator}"], log[f"valid_{self.evaluator}"], log[f"test_{self.evaluator}"]

            self.lr_scheduler.step(valid_error)

            monitor_metric = valid_error
            if (self.mnt_mode == "min" and monitor_metric < best_val_error) or \
                (self.mnt_mode == "max" and monitor_metric > best_val_error):
                best_val_error = monitor_metric
                test_error = tmp_test_error

                ''' Save checkpoint '''
                self.save_checkpoint()

            if epoch % self.log_steps == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train mae: {train_error:.4f},  '
                      f'Valid mae: {valid_error:.4f},  '
                      f'Test mae: {tmp_test_error:.4f}, ')
        return test_error

    def test(self):
        log = {}
        self.model.eval()

        train_error = torch.zeros(self.task_num)
        valid_error = torch.zeros(self.task_num)
        test_error = torch.zeros(self.task_num)
        
        with torch.no_grad():
            if self.eval_separate:
                y_true = []
                y_pred = []
                for data in self.val_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)  
                    labels = data.y
                    y_pred.append(outputs[:, self.task_idxes].cpu())
                    y_true.append(labels[:, self.task_idxes].cpu())
                y_true = torch.cat(y_true, dim = 0)
                y_pred = torch.cat(y_pred, dim = 0)

                for i, idx in enumerate(self.task_idxes):
                    is_labeled = ~torch.isnan(y_true[:, i])
                    if is_labeled.sum() == 0 or (y_true[is_labeled, i] == 1).sum() == 0 or (y_true[is_labeled, i] == 0).sum() == 0:
                        continue
                    errors = name_to_evaluators[self.evaluator](y_pred[is_labeled, i], y_true[is_labeled, i])
                    valid_error[i] = errors.cpu()

                y_true = []
                y_pred = []
                for data in self.test_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)  
                    labels = data.y
                    y_pred.append(outputs[:, self.task_idxes].cpu())
                    y_true.append(labels[:, self.task_idxes].cpu())
                y_true = torch.cat(y_true, dim = 0)
                y_pred = torch.cat(y_pred, dim = 0)

                for i, idx in enumerate(self.task_idxes):
                    is_labeled = ~torch.isnan(y_true[:, i])
                    if is_labeled.sum() == 0 or (y_true[is_labeled, i] == 1).sum() == 0 or (y_true[is_labeled, i] == 0).sum() == 0:
                        continue
                    errors = name_to_evaluators[self.evaluator](y_pred[is_labeled, i], y_true[is_labeled, i])
                    test_error[i] = errors.cpu()
            else:
                steps = 0
                for data in self.val_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)  
                    labels = data.y
                    errors = name_to_evaluators[self.evaluator](outputs[:, self.task_idxes], labels[:, self.task_idxes], reduction=None)
                    valid_error = valid_error + errors.sum(dim=0).cpu()
                    steps += data.y.shape[0]
                valid_error = valid_error/steps

                steps = 0
                for data in self.test_loader:
                    data = data.to(self.device)
                    outputs = self.model(data)  
                    labels = data.y
                    errors = name_to_evaluators[self.evaluator](outputs[:, self.task_idxes], labels[:, self.task_idxes], reduction=None)
                    test_error = test_error + errors.sum(dim=0).cpu()
                    steps += data.y.shape[0]
                test_error = test_error/steps

        log = {
                f"train_{self.evaluator}": train_error.mean().item(), 
                f"valid_{self.evaluator}": valid_error.mean().item(), 
                f"test_{self.evaluator}": test_error.mean().item(), 
            }
        for i, task_idx in enumerate(self.task_idxes):
            task_log = {
                f"task_{task_idx}_train_{self.evaluator}": train_error[i].item(), 
                f"task_{task_idx}_valid_{self.evaluator}": valid_error[i].item(), 
                f"task_{task_idx}_test_{self.evaluator}": test_error[i].item(), 
            }
            log.update(**task_log)

        return log