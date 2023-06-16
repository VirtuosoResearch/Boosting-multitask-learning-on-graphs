import torch
from minibatch_trainer.minibatch_trainer import Trainer, name_to_evaluators, names_to_criterions

class MultitaskTrainer(Trainer):

    def __init__(self, model, optimizer, data, train_loader, test_loader, device, 
        epochs, log_steps, checkpoint_dir, degrees, degree_thres, task_idxes,
        criterion="multiclass", evaluator="accuracy", monitor="avg", decoupling=False):
        super().__init__(model, optimizer, data, train_loader, test_loader, device, epochs, log_steps, checkpoint_dir, degrees, degree_thres, criterion, evaluator, monitor, decoupling)
        self.task_idxes = task_idxes
        self.task_num = len(self.task_idxes)

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0; steps = 0
        if self.decoupling:
            # For decoupling trainer, the train loader only loads propagated features and labels
            for batch in self.train_loader:
                xs, y, train_mask = batch
                xs, y, train_mask = xs.to(self.device), y.to(self.device), train_mask.to(self.device)
                xs = [x for x in torch.split(xs, self.input_dim, -1)]
                self.optimizer.zero_grad()
                
                outputs = self.model(xs, return_softmax=self.criterion=="multiclass")
                labels = y.squeeze(1) if self.criterion == "multiclass"  else y

                if len(train_mask.shape) == 2:
                    loss = 0; sample_count = 0
                    for task_idx in range(train_mask.shape[1]):
                        task_train_mask = train_mask[:, task_idx]
                        loss += names_to_criterions[self.criterion](outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx])*task_train_mask.sum()
                        sample_count += task_train_mask.sum()
                    loss = loss/sample_count
                else:
                    loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1
        else:
            for data in self.train_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                if hasattr(data, "adj_t") and data.adj_t is not None:
                    outputs = self.model(data.x, edge_index = data.adj_t, return_softmax=self.criterion=="multiclass")
                else:
                    outputs = self.model(data.x, edge_index = data.edge_index, return_softmax=self.criterion=="multiclass")
                labels = data.y.squeeze(1) if self.criterion == "multiclass"  else data.y
                train_mask = data.train_mask

                if len(train_mask.shape) == 2:
                    loss = 0; sample_count = 0
                    for task_idx in range(train_mask.shape[1]):
                        task_train_mask = train_mask[:, task_idx]
                        loss += names_to_criterions[self.criterion](outputs[task_train_mask][:, task_idx], labels[task_train_mask][:, task_idx])*task_train_mask.sum()
                        sample_count += task_train_mask.sum()
                    loss = loss/sample_count
                else:
                    loss = names_to_criterions[self.criterion](outputs[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); steps += 1
        
        return total_loss / steps

    def train(self):
        best_val_acc = test_acc = test_longtail_acc = 0

        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch(epoch)
            if len(self.data.train_mask.shape) == 2:
                log = self.test_in_task_mask()
            else:
                log = self.test()
            train_acc, valid_acc, tmp_test_acc, train_loss, valid_loss, test_loss, valid_longtail_acc, tmp_test_longtail_acc = \
                log[f"train_{self.evaluator}"], log[f"valid_{self.evaluator}"], log[f"test_{self.evaluator}"], \
                log["train_loss"], log["valid_loss"], log["test_loss"], log[f"valid_longtail_{self.evaluator}"], log[f"test_longtail_{self.evaluator}"]

            monitor_metric = valid_acc if self.monitor == 'avg' else valid_longtail_acc
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                test_acc = tmp_test_acc
                test_longtail_acc = tmp_test_longtail_acc

                ''' Save checkpoint '''
                self.save_checkpoint()

            if epoch % self.log_steps == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, train loss: {train_loss:.4f}, '
                      f'Valid: {100 * valid_acc:.2f}%, valid loss: {valid_loss:.4f}, '
                      f'Test: {100 * tmp_test_acc:.2f}%, test_loss: {test_loss:.4f}, ' 
                      f'Valid longtail: {100*valid_longtail_acc:.2f}% '
                      f'Test longtail: {100*tmp_test_longtail_acc:.2f}%')
        return test_acc, test_longtail_acc
    
    
    def split_predictions_by_masks(self, y_pred, y_true, mask):
        assert len(mask.shape) == 2 and (mask.shape[1] == y_pred.shape[1])
        indices = list(mask.sum(dim=1).numpy())
        y_pred_split = y_pred.T[mask.T]
        y_true_split = y_true.T[mask.T]
        y_pred_split = torch.split(y_pred_split, indices)
        y_true_split = torch.split(y_true_split, indices)
        return y_pred_split, y_true_split
        

    def test_in_task_mask(self):
        log = {}
        self.model.eval()
        
        data = self.data.to("cpu")
        if self.decoupling:
            out = self.model.inference(data.xs, device = self.device, return_softmax = self.criterion=="multiclass")
        else:
            out = self.model.inference(data.x, test_loader = self.test_loader, device = self.device, return_softmax = self.criterion=="multiclass")
        
        if self.criterion == "multiclass": 
            y_pred = out.argmax(dim=-1, keepdim=True)
            y_true = data.y.squeeze(1)
        elif self.criterion == "multilabel":
            y_pred = out if self.evaluator == "roc_auc" else (out > 0).type(torch.float)
            y_true = data.y
        
        task_num = data.train_mask.shape[1]

        train_accs = torch.zeros(task_num); valid_accs = torch.zeros(task_num); test_accs = torch.zeros(task_num)
        train_precisions = torch.zeros(task_num); valid_precisions = torch.zeros(task_num); test_precisions = torch.zeros(task_num)
        train_losses = torch.zeros(task_num); valid_losses = torch.zeros(task_num); test_losses = torch.zeros(task_num)
        for task_idx in range(task_num):
            train_accs[task_idx] = name_to_evaluators[self.evaluator](
                target = y_true[data.train_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.train_mask[:, task_idx]][:, task_idx]
            ).cpu()
            valid_accs[task_idx] = name_to_evaluators[self.evaluator](
                target = y_true[data.val_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.val_mask[:, task_idx]][:, task_idx]
            ).cpu()
            test_accs[task_idx] = name_to_evaluators[self.evaluator](
                target = y_true[data.test_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.test_mask[:, task_idx]][:, task_idx]
            ).cpu()

            # also evaluate precisions
            train_precisions[task_idx] = name_to_evaluators["precision"](
                target = y_true[data.train_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.train_mask[:, task_idx]][:, task_idx]
            ).cpu()
            valid_precisions[task_idx] = name_to_evaluators["precision"](
                target = y_true[data.val_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.val_mask[:, task_idx]][:, task_idx]
            ).cpu()
            test_precisions[task_idx] = name_to_evaluators["precision"](
                target = y_true[data.test_mask[:, task_idx]][:, task_idx],
                preds = y_pred[data.test_mask[:, task_idx]][:, task_idx]
            ).cpu()
            
            train_losses[task_idx] = names_to_criterions[self.criterion](out[data.train_mask[:, task_idx]][:, task_idx], y_true[data.train_mask[:, task_idx]][:, task_idx]).cpu()
            valid_losses[task_idx] = names_to_criterions[self.criterion](out[data.val_mask[:, task_idx]][:, task_idx], y_true[data.val_mask[:, task_idx]][:, task_idx]).cpu()
            test_losses[task_idx]  = names_to_criterions[self.criterion](out[data.test_mask[:, task_idx]][:, task_idx], y_true[data.test_mask[:, task_idx]][:, task_idx]).cpu()
        # print(test_accs)
        log = {
                "train_loss": train_losses.mean().item(), 
                "valid_loss": valid_losses.mean().item(), 
                "test_loss": test_losses.mean().item(), 
                f"train_{self.evaluator}": train_accs.mean().item(), 
                f"valid_{self.evaluator}": valid_accs.mean().item(), 
                f"test_{self.evaluator}": test_accs.mean().item(), 
                f"valid_longtail_{self.evaluator}": 0, 
                f"test_longtail_{self.evaluator}": 0,
                f"train_precision": train_precisions.mean().item(), 
                f"valid_precision": valid_precisions.mean().item(), 
                f"test_precision":   test_precisions.mean().item(), 
            }
        for i, task_idx in enumerate(self.task_idxes):
            task_log = {
                f"task_{task_idx}_train_{self.evaluator}": train_accs[i].item(), 
                f"task_{task_idx}_valid_{self.evaluator}": valid_accs[i].item(), 
                f"task_{task_idx}_test_{self.evaluator}": test_accs[i].item(), 
                f"task_{task_idx}_valid_longtail_{self.evaluator}": 0, 
                f"task_{task_idx}_test_longtail_{self.evaluator}": 0,
                f"task_{task_idx}_train_precision": train_precisions[i].item(), 
                f"task_{task_idx}_valid_precision": valid_precisions[i].item(), 
                f"task_{task_idx}_test_precision":   test_precisions[i].item(), 
            }
            log.update(**task_log)

        return log

    def test(self):
        log = {}
        self.model.eval()
        
        data = self.data.to("cpu")
        if self.decoupling:
            out = self.model.inference(data.xs, device = self.device, return_softmax = self.criterion=="multiclass")
        else:
            out = self.model.inference(data.x, test_loader = self.test_loader, device = self.device, return_softmax = self.criterion=="multiclass")
        
        if self.criterion == "multiclass": 
            y_pred = out.argmax(dim=-1, keepdim=True)
            y_true = data.y.squeeze(1)
        elif self.criterion == "multilabel":
            y_pred = out if self.evaluator == "roc_auc" else (out > 0).type(torch.float)
            y_true = data.y
        
        train_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.train_mask],
            preds = y_pred[data.train_mask],
            reduction = None
        ).cpu()
        valid_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.val_mask],
            preds = y_pred[data.val_mask],
            reduction = None
        ).cpu()
        test_acc = name_to_evaluators[self.evaluator](
            target = y_true[data.test_mask],
            preds = y_pred[data.test_mask],
            reduction = None
        ).cpu()
        
        # also evaluate precisions
        train_precisions = name_to_evaluators["precision"](
            target = y_true[data.train_mask],
            preds = y_pred[data.train_mask],
            reduction = None
        ).cpu()
        valid_precisions = name_to_evaluators["precision"](
            target = y_true[data.val_mask],
            preds = y_pred[data.val_mask],
            reduction = None
        ).cpu()
        test_precisions = name_to_evaluators["precision"](
            target = y_true[data.test_mask],
            preds = y_pred[data.test_mask],
            reduction = None
        ).cpu()

        train_loss = names_to_criterions[self.criterion](out[data.train_mask], y_true[data.train_mask]).cpu().item()
        valid_loss = names_to_criterions[self.criterion](out[data.val_mask], y_true[data.val_mask]).cpu().item()
        test_loss  = names_to_criterions[self.criterion](out[data.test_mask], y_true[data.test_mask]).cpu().item()

        ''' Evaluate longtail performance'''
        # # valid_degrees = self.degrees[data.val_mask]
        # longtail_valid_mask = torch.logical_and(data.val_mask, self.degrees<=self.degree_thres)
        # valid_longtail_acc = name_to_evaluators[self.evaluator](
        #     target = y_true[longtail_valid_mask],
        #     preds = y_pred[longtail_valid_mask],
        #     reduction = None
        # ).cpu()

        # # test_degrees = self.degrees[data.test_mask]
        # longtail_test_mask = torch.logical_and(data.test_mask, self.degrees<=self.degree_thres)
        # test_longtail_acc = name_to_evaluators[self.evaluator](
        #     target = y_true[longtail_test_mask],
        #     preds = y_pred[longtail_test_mask],
        #     reduction = None
        # ).cpu()

        log = {
                "train_loss": train_loss, 
                "valid_loss": valid_loss, 
                "test_loss": test_loss, 
                f"train_{self.evaluator}": train_acc.mean().item(), 
                f"valid_{self.evaluator}": valid_acc.mean().item(), 
                f"test_{self.evaluator}": test_acc.mean().item(), 
                f"valid_longtail_{self.evaluator}": 0, # valid_longtail_acc.mean().item(), 
                f"test_longtail_{self.evaluator}": 0, # test_longtail_acc.mean().item()
                f"train_precision": train_precisions.mean().item(), 
                f"valid_precision": valid_precisions.mean().item(), 
                f"test_precision":   test_precisions.mean().item(), 
            }
        for i, task_idx in enumerate(self.task_idxes):
            task_log = {
                f"task_{task_idx}_train_{self.evaluator}": train_acc[i].item(), 
                f"task_{task_idx}_valid_{self.evaluator}": valid_acc[i].item(), 
                f"task_{task_idx}_test_{self.evaluator}": test_acc[i].item(), 
                f"task_{task_idx}_valid_longtail_{self.evaluator}": 0, # valid_longtail_acc[i].item(), 
                f"task_{task_idx}_test_longtail_{self.evaluator}": 0, # test_longtail_acc[i].item()
                f"task_{task_idx}_train_precision": train_precisions[i].item(), 
                f"task_{task_idx}_valid_precision": valid_precisions[i].item(), 
                f"task_{task_idx}_test_precision":   test_precisions[i].item(), 
            }
            log.update(**task_log)

        return log