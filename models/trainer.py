import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.cuda.amp as amp
import time
from statistics import mean
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data_loader import get_train_validation_loader, get_test_loader
from model import SiameseNetwork
from utils import AverageMeter

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.best_valid_acc = 0
        self.counter = 0

    def train(self):
        results = {
            'process_times': [],
            'cpu_times': [],
            'highest_accuracies': [],
            'average_accuracies': [],
            'training_times': [],
            'losses': [],
            'precisions': [],
            'recalls': []
        }

        train_loader, valid_loader = get_train_validation_loader(
            self.config.data_dir,
            self.config.batch_size,
            self.config.num_workers,
            self.config.pin_memory,
        )

        model = SiameseNetwork()
        if self.config.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=self.config.lr, weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-5)

        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        if self.config.use_gpu:
            model.cuda()

        if self.config.resume:
            start_epoch, best_epoch, best_valid_acc, model_state, optim_state = self.load_checkpoint(best=False)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
        else:
            best_epoch = 0
            start_epoch = 0
            best_valid_acc = 0

        writer = SummaryWriter(os.path.join(self.config.logs_dir, 'logs'), filename_suffix=str(self.config.num_model))
        (img1, img2), labels = next(iter(valid_loader))
        img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
        writer.add_graph(model, (img1, img2))

        num_train = len(train_loader)
        num_valid = len(valid_loader)
        print(f"[*] Train on {len(train_loader.dataset)} samples, validate on {len(valid_loader.dataset)} samples")

        start_process_time = time.time()
        main_pbar = tqdm(range(start_epoch, self.config.epochs), initial=start_epoch, position=0, total=self.config.epochs, desc="Epochs")
        for epoch in main_pbar:
            train_losses = AverageMeter()
            valid_losses = AverageMeter()

            model.train()
            scaler = amp.GradScaler()
            start_training_time = time.time()
            train_pbar = tqdm(enumerate(train_loader), total=num_train, desc="Train", position=1, leave=False)
            for i, ((img1, img2), labels) in train_pbar:
                if self.config.use_gpu:
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                with amp.autocast():
                    outputs = model(img1, img2)
                    outputs = outputs.squeeze()  # Ensure outputs are squeezed to correct dimension
                    labels = labels.view(-1)  # Ensure labels are reshaped to match outputs
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_losses.update(loss.item(), img1.shape[0])
                writer.add_scalar("Loss/Train", train_losses.val, epoch * len(train_loader) + i)
                train_pbar.set_postfix_str(f"loss: {train_losses.val:.3f}")

            end_training_time = time.time()
            training_time = end_training_time - start_training_time

            model.eval()
            correct_sum = 0
            all_labels = []
            all_preds = []
            valid_pbar = tqdm(enumerate(valid_loader), total=num_valid, desc="Valid", position=1, leave=False)
            with torch.no_grad():
                for i, ((img1, img2), labels) in valid_pbar:
                    if self.config.use_gpu:
                        img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                    outputs = model(img1, img2)
                    outputs = outputs.squeeze()  # Ensure outputs are squeezed to correct dimension
                    labels = labels.view(-1)  # Ensure labels are reshaped to match outputs
                    loss = criterion(outputs, labels)

                    preds = torch.round(torch.sigmoid(outputs))
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    correct_sum += torch.sum(preds == labels).item()

                    valid_losses.update(loss.item(), img1.shape[0])
                    valid_acc = correct_sum / len(valid_loader.dataset)
                    writer.add_scalar("Loss/Valid", valid_losses.val, epoch * len(valid_loader) + i)
                    valid_pbar.set_postfix_str(f"accuracy: {valid_acc:.3f}")

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            writer.add_scalar("Acc/Valid", accuracy, epoch)
            writer.add_scalar("Precision/Valid", precision, epoch)
            writer.add_scalar("Recall/Valid", recall, epoch)

            if accuracy > best_valid_acc:
                is_best = True
                best_valid_acc = accuracy
                best_epoch = epoch
            else:
                is_best = False

            if is_best or epoch % 5 == 0 or epoch == self.config.epochs - 1:
                self.save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                        'best_valid_acc': best_valid_acc,
                        'best_epoch': best_epoch,
                    }, is_best
                )

            scheduler.step(valid_acc)
            print(f"Epoch {epoch}: Train Loss: {train_losses.avg:.4f}, Valid Loss: {valid_losses.avg:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            main_pbar.set_postfix_str(f"best acc: {best_valid_acc:.3f} best epoch: {best_epoch}")
            tqdm.write(f"[{epoch}] train loss: {train_losses.avg:.3f} - valid loss: {valid_losses.avg:.3f} - valid acc: {accuracy:.3f} {'[BEST]' if is_best else ''}")

        end_process_time = time.time()
        process_time = end_process_time - start_process_time

        results['process_times'].append(process_time)
        results['cpu_times'].append(end_process_time - start_process_time)
        results['highest_accuracies'].append(best_valid_acc)
        results['average_accuracies'].append(valid_acc)
        results['training_times'].append(training_time)
        results['losses'].append(valid_losses.avg)
        results['precisions'].append(precision)
        results['recalls'].append(recall)

        writer.close()

        # Display results
        print("\n\nSummary of Results:")
        print(f"Average Process Time: {mean(results['process_times']):.2f} seconds")
        print(f"Average CPU Time: {mean(results['cpu_times']):.2f} seconds")
        print(f"Highest Accuracy: {max(results['highest_accuracies']):.4f}")
        print(f"Average Accuracy: {mean(results['average_accuracies']):.4f}")
        print(f"Average Training Time: {mean(results['training_times']):.2f} seconds")
        print(f"Average Loss: {mean(results['losses']):.4f}")
        print(f"Average Precision: {mean(results['precisions']):.4f}")
        print(f"Average Recall: {mean(results['recalls']):.4f}")

    def test(self):
        model = SiameseNetwork()
        _, _, _, model_state, _ = self.load_checkpoint(best=True)
        model.load_state_dict(model_state)
        if self.config.use_gpu:
            model.cuda()
        test_loader = get_test_loader(
            self.config.data_dir,
            self.config.batch_size, 
            self.config.num_workers,
            self.config.pin_memory,
        )

        model.to(self.device)  # Ensure model is moved to the correct device

        correct_sum = 0
        total_samples = 0
        print(f"[*] Test on {len(test_loader.dataset)} samples.")

        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test")
        with torch.no_grad():
            for i, ((img1, img2), labels) in pbar:
                if self.config.use_gpu:
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                outputs = model(img1, img2)
                outputs = outputs.squeeze()  # Ensure outputs are squeezed to correct dimension
                labels = labels.view(-1)  # Ensure labels are reshaped to match outputs

                preds = torch.round(torch.sigmoid(outputs))
                correct_predictions = (preds == labels).sum().item()
                correct_sum += correct_predictions
                total_samples += labels.size(0)

                pbar.set_postfix_str(f"accuracy: {correct_sum / total_samples:.4f}")

            test_acc = (100. * correct_sum) / total_samples
            print(f"Test Acc: {correct_sum}/{total_samples} ({test_acc:.2f}%)")



    def save_checkpoint(self, state, is_best):
        model_dir = os.path.join(self.config.logs_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        filename = 'best_model.pt' if is_best else f'model_ckpt_{state["epoch"]}.pt'
        model_path = os.path.join(model_dir, filename)
        torch.save(state, model_path)

    def load_checkpoint(self, best):
        print(f"[*] Loading model Num.{self.config.num_model}...", end="")

        model_dir = os.path.join(self.config.logs_dir, 'models')
        if best:
            model_path = os.path.join(model_dir, 'best_model.pt')
        else:
            model_path = sorted(glob(os.path.join(model_dir, 'model_ckpt_*.pt')), key=os.path.getmtime)[-1]

        ckpt = torch.load(model_path)

        if best:
            print(f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']} with best valid acc of {ckpt['best_valid_acc']:.3f}")
        else:
            print(f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']}")

        return ckpt['epoch'], ckpt['best_epoch'], ckpt['best_valid_acc'], ckpt['model_state'], ckpt['optim_state']

