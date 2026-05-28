import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import json
from tqdm import tqdm
import math

from Dataload import load_data
from main_model_expression_tsp import setup_model


def setup_cuda():
    """设置CUDA设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        return device
    else:
        return torch.device('cpu')


def create_result_dir(base_dir='train_result_1302_1'):
    """创建训练结果目录结构"""
    os.makedirs(base_dir, exist_ok=True)
    for subdir in ['models', 'losses', 'outputs', 'reports']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    return base_dir


def get_model():
    """创建Transformer模型"""
    model = setup_model()
    return model.to(device=setup_cuda())


class RMSELoss(nn.Module):
    """RMSE损失函数 - 稳定版本"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = torch.mean((input - target) ** 2)
        return torch.sqrt(loss + self.eps)


class FastConvergenceScheduler:
    """快速收敛学习率调度器"""

    def __init__(self, optimizer, total_epochs=100, **kwargs):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.epoch_count = 0

        self.initial_lr = optimizer.param_groups[0]['lr']
        self.min_lr = kwargs.get('min_lr', 1e-6)

        self.warmup_epochs = kwargs.get('warmup_epochs', 10)
        self.stage1_epochs = int(total_epochs * 0.3)
        self.stage2_epochs = int(total_epochs * 0.5)
        self.stage3_epochs = total_epochs - self.stage1_epochs - self.stage2_epochs

        self.loss_improvement_threshold = kwargs.get('improvement_threshold', 0.01)
        self.last_loss = float('inf')

    def step(self, current_loss=None):
        self.epoch_count += 1

        if self.epoch_count <= self.warmup_epochs:
            new_lr = self._warmup_schedule()
        elif self.epoch_count <= self.stage1_epochs + self.warmup_epochs:
            new_lr = self._stage1_schedule()
        elif self.epoch_count <= self.stage1_epochs + self.stage2_epochs + self.warmup_epochs:
            new_lr = self._stage2_schedule()
        else:
            new_lr = self._stage3_schedule()

        if current_loss is not None:
            new_lr = self._adaptive_adjustment(new_lr, current_loss)

        self._set_learning_rate(new_lr)
        self.last_loss = current_loss if current_loss is not None else self.last_loss

    def _warmup_schedule(self):
        progress = self.epoch_count / self.warmup_epochs
        return self.initial_lr * 0.1 + (self.initial_lr - self.initial_lr * 0.1) * progress

    def _stage1_schedule(self):
        epoch_in_stage = self.epoch_count - self.warmup_epochs
        progress = epoch_in_stage / self.stage1_epochs
        return self.initial_lr * (1 - 0.2 * progress)

    def _stage2_schedule(self):
        epoch_in_stage = self.epoch_count - self.warmup_epochs - self.stage1_epochs
        progress = epoch_in_stage / self.stage2_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr * 0.8 - self.min_lr) * cosine_decay

    def _stage3_schedule(self):
        epoch_in_stage = self.epoch_count - self.warmup_epochs - self.stage1_epochs - self.stage2_epochs
        progress = epoch_in_stage / self.stage3_epochs
        return max(self.min_lr, self.initial_lr * 0.1 * (0.5 ** (progress * 3)))

    def _adaptive_adjustment(self, proposed_lr, current_loss):
        if self.last_loss == float('inf'):
            return proposed_lr

        loss_improvement = (self.last_loss - current_loss) / self.last_loss

        if loss_improvement > self.loss_improvement_threshold:
            return min(proposed_lr * 1.1, self.initial_lr)
        elif loss_improvement < -0.05:
            return max(proposed_lr * 0.7, self.min_lr)
        elif loss_improvement < 0:
            return max(proposed_lr * 0.9, self.min_lr)

        return proposed_lr

    def _set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


def get_optimizer_and_scheduler(model, epochs, learning_rate=0.001):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = FastConvergenceScheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=min(10, epochs // 10),
        min_lr=learning_rate * 0.001,
        improvement_threshold=0.02
    )

    return optimizer, scheduler


def save_model_and_training_results(model, loss_history, test_outputs, learning_rates,
                                    result_dir='train_result_1302_1', epoch=None, is_best=False):
    """
    保存模型和训练结果 - 增强版本
    """
    model_info = {
        'model_type': model.__class__.__name__,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epoch': epoch,
        'is_best': is_best
    }

    # 注意：这里是你原本写死的配置（如需一致请改成从 setup_model 里读）
    model_config = {
        'input_dim': 14,
        'seq_len': 30,
        'hidden_dim': 64,
        'mlp_hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2
    }

    if is_best:
        best_checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'model_info': model_info,
            'epoch': epoch,
            'best_test_loss': loss_history.get('best_test_loss', None),
            'best_test_rmse': loss_history.get('best_test_rmse', None),
            'learning_rate': learning_rates[-1] if learning_rates else 0.001,
            'train_loss': loss_history.get('train_total', [])[-1] if loss_history.get('train_total') else None,
            'train_rmse': loss_history.get('train_rmse', [])[-1] if loss_history.get('train_rmse') else None,
            'loss_history': {k: v[-100:] for k, v in loss_history.items() if isinstance(v, list)}
        }

        best_model_path = os.path.join(result_dir, 'models', 'best_model_complete.pth')
        torch.save(best_checkpoint, best_model_path)

        weights_only_path = os.path.join(result_dir, 'models', 'best_model_weights.pth')
        torch.save(model.state_dict(), weights_only_path)

        config_info = model_config.copy()
        config_info.update({
            'best_epoch': epoch,
            'best_rmse': loss_history.get('best_test_rmse', None),
            'best_loss': loss_history.get('best_test_loss', None),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'total_params': model_info['total_params'],
                'trainable_params': model_info['trainable_params']
            }
        })

        config_path = os.path.join(result_dir, 'models', 'best_model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=4, ensure_ascii=False)
        print(f"✓ 最佳模型配置已保存: {config_path}")
        print(f"{'=' * 60}")

    else:
        if epoch is not None and epoch % 10 == 0:
            full_checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'model_info': model_info,
                'epoch': epoch,
                'loss_history': loss_history,
                'learning_rates': learning_rates
            }

            epoch_model_path = os.path.join(result_dir, 'models', f'model_epoch_{epoch:03d}.pth')
            torch.save(full_checkpoint, epoch_model_path)
            print(f"✓ 第{epoch}个epoch模型已保存: {epoch_model_path}")

    loss_history_path = os.path.join(result_dir, 'losses', 'loss_history.json')
    with open(loss_history_path, 'w', encoding='utf-8') as f:
        json.dump(loss_history, f, indent=4, ensure_ascii=False, default=str)

    lr_history_path = os.path.join(result_dir, 'losses', 'learning_rates.json')
    with open(lr_history_path, 'w', encoding='utf-8') as f:
        json.dump(learning_rates, f, indent=4, ensure_ascii=False)

    if test_outputs is not None:
        output_path = os.path.join(result_dir, 'outputs', 'test_outputs.npz')
        np.savez_compressed(output_path, **test_outputs)

    summary = {
        'model_info': model_info,
        'model_config': model_config,
        'training_stats': {
            'total_epochs': len(learning_rates),
            'final_learning_rate': learning_rates[-1] if learning_rates else 0,
            'train_loss': loss_history.get('train_total', [])[-1] if loss_history.get('train_total') else None,
            'test_loss': loss_history.get('test_total', [])[-1] if loss_history.get('test_total') else None,
            'best_test_rmse': loss_history.get('best_test_rmse', None),
            'best_test_loss': loss_history.get('best_test_loss', None),
            'best_epoch': loss_history.get('best_epoch', None)
        },
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'saved_files': {
            'best_model_complete.pth': '包含模型结构和权重的完整检查点',
            'best_model_weights.pth': '仅权重文件，加载速度快',
            'best_model_config.json': '模型配置信息',
            'loss_history.json': '训练损失历史',
            'learning_rates.json': '学习率历史'
        }
    }

    summary_path = os.path.join(result_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


def compute_losses(output, target, model, l2_lambda=0.001):
    mae_loss = nn.L1Loss()(output, target)
    mse_loss = nn.MSELoss()(output, target)
    rmse_loss = torch.sqrt(mse_loss + 1e-8)

    reconstruction_loss = mse_loss + mae_loss

    l2_reg = torch.tensor(0.).to(output.device)
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.norm(param, p=2)

    total_loss = reconstruction_loss + l2_lambda * l2_reg
    return total_loss, mse_loss, rmse_loss, reconstruction_loss


def normalize_data(train_X, train_Y, test_X=None, test_Y=None):
    if isinstance(train_X, torch.Tensor):
        train_X_np = train_X.numpy()
        train_Y_np = train_Y.numpy()
    else:
        train_X_np = train_X
        train_Y_np = train_Y

    x_mean = np.mean(train_X_np, axis=(0, 1), keepdims=True)
    x_std = np.std(train_X_np, axis=(0, 1), keepdims=True) + 1e-8

    y_mean = np.mean(train_Y_np, axis=0, keepdims=True)
    y_std = np.std(train_Y_np, axis=0, keepdims=True) + 1e-8

    train_X_norm = (train_X_np - x_mean) / x_std
    train_Y_norm = train_Y_np

    print(f"\n数据标准化统计:")
    print(f"  X - 均值范围: [{x_mean.min():.4f}, {x_mean.max():.4f}]")
    print(f"  X - 标准差范围: [{x_std.min():.4f}, {x_std.max():.4f}]")
    print(f"  Y - 均值: {y_mean[0][0]:.4f}, 标准差: {y_std[0][0]:.4f}")

    test_X_norm = None
    test_Y_norm = None
    if test_X is not None and test_Y is not None:
        if isinstance(test_X, torch.Tensor):
            test_X_np = test_X.numpy()
            test_Y_np = test_Y.numpy()
        else:
            test_X_np = test_X
            test_Y_np = test_Y

        test_X_norm = (test_X_np - x_mean) / x_std
        test_Y_norm = test_Y_np
        print(f"  测试数据标准化完成")

    train_X_tensor = torch.FloatTensor(train_X_norm)
    train_Y_tensor = torch.FloatTensor(train_Y_norm)

    if test_X_norm is not None:
        test_X_tensor = torch.FloatTensor(test_X_norm)
        test_Y_tensor = torch.FloatTensor(test_Y_norm)
        return train_X_tensor, train_Y_tensor, test_X_tensor, test_Y_tensor, y_mean, y_std



def create_data_loaders(train_X, train_Y, test_X=None, test_Y=None, batch_size=32):
    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2
    )

    test_loader = None
    if test_X is not None and test_Y is not None:
        test_dataset = TensorDataset(test_X, test_Y)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=2
        )

    return train_loader, test_loader


def train_model(model, train_X, train_Y, test_X=None, test_Y=None, epochs=100,
                batch_size=32, learning_rate=0.001, result_dir='train_result_1302_1'):
    """训练模型 - 从第5个epoch开始保存最佳模型（按Val RMSE）"""
    device = setup_cuda()
    model = model.to(device)

    train_loader, test_loader = create_data_loaders(train_X, train_Y, test_X, test_Y, batch_size)
    optimizer, scheduler = get_optimizer_and_scheduler(model, epochs, learning_rate)

    history = {
        'train': {'mse': [], 'rmse': [], 'scale': [], 'total': []},
        'test': {'mse': [], 'rmse': [], 'scale': [], 'total': []},
        'learning_rates': []
    }

    best_test_loss = float('inf')
    best_test_rmse = float('inf')
    best_epoch = 0
    best_model_state = None

    nan_epochs = 0
    max_nan_epochs = 3

    min_epoch_to_save = 5  # 只保留这个门槛

    print(f"\n开始训练:")
    print(f"  训练集: {len(train_X)} 样本")
    print(f"  验证集: {len(test_X) if test_X is not None else 0} 样本")
    print(f"  Batch size: {batch_size}")
    print(f"  总轮数: {epochs}")
    print(f"  学习率: {learning_rate:.6f}")
    print(f"  保存策略: 从第{min_epoch_to_save}个epoch开始，Val RMSE 变好就保存最佳模型")
    print(f"  结果保存到: {result_dir}")
    print("-" * 80)

    start_time = time.time()

    for epoch in range(epochs):
        current_lr = scheduler.get_current_lr()
        history['learning_rates'].append(current_lr)

        # Train
        model.train()
        train_metrics = {'mse': 0.0, 'rmse': 0.0, 'scale': 0.0, 'total': 0.0}
        train_samples = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
        for batch_X, batch_Y in train_pbar:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            if torch.isnan(batch_X).any() or torch.isnan(batch_Y).any():
                print("警告：训练数据中包含NaN，跳过该批次")
                continue

            pred = model(batch_X)

            if torch.isnan(pred).any():
                print("警告：模型输出包含NaN，跳过该批次")
                continue

            total_loss, mse, rmse, recon_loss = compute_losses(pred, batch_Y, model)

            if torch.isnan(total_loss):
                print("警告：损失值为NaN，跳过该批次")
                continue

            optimizer.zero_grad()
            total_loss.backward()

            total_grad_norm = 0.0
            for _, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = math.sqrt(total_grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            mse_val = mse.item()
            rmse_val = rmse.item()
            recon_val = recon_loss.item()
            total_val = total_loss.item()

            bs = batch_X.size(0)
            train_metrics['mse'] += mse_val * bs
            train_metrics['rmse'] += rmse_val * bs
            train_metrics['scale'] += recon_val * bs
            train_metrics['total'] += total_val * bs
            train_samples += bs

            train_pbar.set_postfix({
                'Loss': f'{total_val:.4f}',
                'RMSE': f'{rmse_val:.4f}',
                'Grad': f'{total_grad_norm:.2f}'
            })

        if train_samples > 0:
            for key in train_metrics:
                train_metrics[key] /= train_samples
                history['train'][key].append(train_metrics[key])
        else:
            print("警告：没有有效的训练样本")
            for key in train_metrics:
                history['train'][key].append(float('nan'))

        # Val
        test_metrics = {'mse': 0.0, 'rmse': 0.0, 'scale': 0.0, 'total': 0.0}
        test_samples = 0
        all_outputs, all_targets = [], []

        is_best = False

        if test_loader:
            model.eval()
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]', leave=False)
                for test_batch_X, test_batch_Y in test_pbar:
                    test_batch_X, test_batch_Y = test_batch_X.to(device), test_batch_Y.to(device)

                    if torch.isnan(test_batch_X).any() or torch.isnan(test_batch_Y).any():
                        print("警告：测试数据中包含NaN，跳过该批次")
                        continue

                    test_pred = model(test_batch_X)

                    if torch.isnan(test_pred).any():
                        print("警告：测试输出包含NaN，跳过该批次")
                        continue

                    total_loss, mse, rmse, recon_loss = compute_losses(test_pred, test_batch_Y, model)

                    if torch.isnan(total_loss):
                        print("警告：测试损失值为NaN，跳过该批次")
                        continue

                    mse_val = mse.item()
                    rmse_val = rmse.item()
                    recon_val = recon_loss.item()
                    total_val = total_loss.item()

                    bs = test_batch_X.size(0)
                    test_metrics['mse'] += mse_val * bs
                    test_metrics['rmse'] += rmse_val * bs
                    test_metrics['scale'] += recon_val * bs
                    test_metrics['total'] += total_val * bs
                    test_samples += bs

                    all_outputs.append(test_pred.cpu().numpy())
                    all_targets.append(test_batch_Y.cpu().numpy())

                    test_pbar.set_postfix({
                        'Loss': f'{total_val:.4f}',
                        'RMSE': f'{rmse_val:.4f}'
                    })

            if test_samples > 0:
                for key in test_metrics:
                    test_metrics[key] /= test_samples
                    history['test'][key].append(test_metrics[key])
            else:
                print("警告：没有有效的测试样本")
                for key in test_metrics:
                    history['test'][key].append(float('nan'))

            # 只要从第5个epoch开始，Val RMSE 变好就保存
            if (epoch + 1) >= min_epoch_to_save and test_samples > 0:
                is_rmse_valid = not (torch.isnan(torch.tensor(test_metrics['rmse'])) or
                                     torch.isinf(torch.tensor(test_metrics['rmse'])))
                is_loss_valid = not (torch.isnan(torch.tensor(test_metrics['total'])) or
                                     torch.isinf(torch.tensor(test_metrics['total'])))

                if is_rmse_valid and is_loss_valid:
                    if test_metrics['rmse'] < best_test_rmse:
                        best_test_loss = test_metrics['total']
                        best_test_rmse = test_metrics['rmse']
                        best_epoch = epoch + 1
                        best_model_state = model.state_dict().copy()
                        is_best = True

        # Scheduler step
        current_loss_for_scheduler = test_metrics['total'] if test_loader and test_samples > 0 else train_metrics['total']
        if not torch.isnan(torch.tensor(current_loss_for_scheduler)):
            scheduler.step(current_loss_for_scheduler)

        # NaN early stop
        epoch_has_nan = any(torch.isnan(torch.tensor(val)).item() for val in train_metrics.values())
        if epoch_has_nan:
            nan_epochs += 1
            print(f"警告：第 {epoch + 1} 个epoch出现NaN，计数: {nan_epochs}")
            if nan_epochs >= max_nan_epochs:
                print(f"连续 {nan_epochs} 个epoch出现NaN，停止训练")
                break
        else:
            nan_epochs = 0

        # Print status + save best immediately
        train_loss_str = f'{train_metrics["total"]:.6f}' if not torch.isnan(torch.tensor(train_metrics["total"])) else 'NaN'
        train_rmse_str = f'{train_metrics["rmse"]:.6f}' if not torch.isnan(torch.tensor(train_metrics["rmse"])) else 'NaN'

        status = f'Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss_str} | Train RMSE: {train_rmse_str}'

        if test_loader and test_samples > 0:
            val_loss_str = f'{test_metrics["total"]:.6f}' if not torch.isnan(torch.tensor(test_metrics["total"])) else 'NaN'
            val_rmse_str = f'{test_metrics["rmse"]:.6f}' if not torch.isnan(torch.tensor(test_metrics["rmse"])) else 'NaN'
            status += f' | Val Loss: {val_loss_str} | Val RMSE: {val_rmse_str}'

            if is_best:
                status += f' | Best RMSE: {best_test_rmse:.6f} (Epoch {best_epoch}) ✓'

                full_loss_history = {
                    'best_test_rmse': best_test_rmse,
                    'best_test_loss': best_test_loss,
                    'best_epoch': best_epoch,
                    'train_total': history['train']['total'],
                    'train_rmse': history['train']['rmse'],
                    'test_total': history['test']['total'],
                    'test_rmse': history['test']['rmse']
                }

                test_outputs_for_save = None
                if all_outputs:
                    test_outputs_for_save = {
                        'predictions': np.concatenate(all_outputs, axis=0),
                        'targets': np.concatenate(all_targets, axis=0)
                    }

                save_model_and_training_results(
                    model, full_loss_history, test_outputs_for_save,
                    history['learning_rates'], result_dir, epoch + 1, is_best=True
                )
            else:
                status += f' | Best RMSE: {best_test_rmse:.6f} (Epoch {best_epoch})'
        else:
            status += f' | Best Train Loss: {best_test_loss:.6f}'

        status += f' | LR: {current_lr:.2e}'
        print(status)

    print("=" * 80)
    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time / 60:.2f} 分钟")

    if test_loader:
        if best_epoch > 0:
            print(f"最佳验证RMSE: {best_test_rmse:.6f} (在第 {best_epoch} 个epoch保存)")
            print(f"对应的验证损失: {best_test_loss:.6f}")
        else:
            print("警告：从第5个epoch开始未出现更优RMSE，未保存最佳模型")
    else:
        print(f"最佳训练RMSE: {best_test_rmse:.6f}")

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已恢复最佳模型状态")
    else:
        print("警告：没有找到最佳模型状态，使用最终模型")
        best_model_path = os.path.join(result_dir, 'models', 'best_model_complete.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_test_rmse = checkpoint.get('best_test_rmse', best_test_rmse)
            best_test_loss = checkpoint.get('best_test_loss', best_test_loss)
            best_epoch = checkpoint.get('epoch', best_epoch)
            print(f"从文件加载最佳模型 (Epoch {best_epoch}, RMSE: {best_test_rmse:.6f})")

    # Final save (training summary etc.)
    full_loss_history = {
        'train_total': history['train']['total'],
        'train_mse': history['train']['mse'],
        'train_rmse': history['train']['rmse'],
        'train_scale': history['train']['scale'],
        'test_total': history['test']['total'] if test_loader else [],
        'test_mse': history['test']['mse'] if test_loader else [],
        'test_rmse': history['test']['rmse'] if test_loader else [],
        'test_scale': history['test']['scale'] if test_loader else [],
        'best_test_rmse': best_test_rmse,
        'best_test_loss': best_test_loss,
        'best_epoch': best_epoch,
        'training_params': {
            'epochs_trained': epoch + 1,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'min_epoch_to_save_best': min_epoch_to_save
        }
    }

    test_outputs_for_final = None
    if all_outputs:
        test_outputs_for_final = {
            'predictions': np.concatenate(all_outputs, axis=0),
            'targets': np.concatenate(all_targets, axis=0)
        }

    save_model_and_training_results(
        model, full_loss_history, test_outputs_for_final,
        history['learning_rates'], result_dir, epochs, is_best=False
    )

    return model, full_loss_history


if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 模型训练 - 从第5个epoch开始保存最佳模型（按Val RMSE）")
    print("=" * 60)

    print("加载数据...")
    path = './train_data/FD004_train_data_a_0.1_seq_30_thresh_130.pt'

    tensor_data_dict = load_data(path, device='cpu')
    train_X = tensor_data_dict['x_train']
    train_Y = tensor_data_dict['y_train']
    test_X = tensor_data_dict['x_val']
    test_Y = tensor_data_dict['y_val']

    print(f"\n原始数据形状:")
    print(f"  训练集 X: {train_X.shape}, Y: {train_Y.shape}")
    print(f"  测试集 X: {test_X.shape}, Y: {test_Y.shape}")

    print("\n数据标准化处理...")
    train_X_norm, train_Y_norm, test_X_norm, test_Y_norm, y_mean, y_std = normalize_data(
        train_X, train_Y, test_X, test_Y
    )

    print(f"\n标准化后数据形状:")
    print(f"  训练集 X: {train_X_norm.shape}, Y: {train_Y_norm.shape}")
    print(f"  验证集 X: {test_X_norm.shape}, Y: {test_Y_norm.shape}")

    result_dir = create_result_dir('train_result')

    print(f"\n创建模型...")
    model = get_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print(f"\n开始训练...")
    print("=" * 60)

    trained_model, loss_history = train_model(
        model=model,
        train_X=train_X_norm,
        train_Y=train_Y_norm,
        test_X=test_X_norm,
        test_Y=test_Y_norm,
        epochs=100,
        batch_size=32,
        learning_rate=0.005,
        result_dir=result_dir
    )

    # 在最后给出“保存最佳模型的RMSE”
    best_rmse = loss_history.get('best_test_rmse', None)
    best_ep = loss_history.get('best_epoch', None)
    if best_rmse is not None and best_ep is not None and best_ep > 0:
        print(f"\n保存的最佳模型 RMSE: {best_rmse:.6f}（Epoch {best_ep}）")
    else:
        print("\n保存的最佳模型 RMSE: N/A（可能从第5个epoch开始未出现更优RMSE，或无验证集）")

    print(f"结果保存到: {result_dir}")

    norm_params = {
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist()
    }
    norm_path = os.path.join(result_dir, 'models', 'normalization_params.json')
    with open(norm_path, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=4, ensure_ascii=False)
    print(f"标准化参数已保存到: {norm_path}")

    print("=" * 60)