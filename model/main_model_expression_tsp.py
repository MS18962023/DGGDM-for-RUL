import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


# ==============================================================================
# 0. DCTN Components (Time Series Decomposition)
# ==============================================================================

class DepthwiseConvBlock(nn.Module):
    """Depthwise convolution block with dropout"""
    def __init__(self, in_channels, num_layers, output_channels=None, kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.output_channels = output_channels if output_channels is not None else in_channels

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels * num_layers, kernel_size=(kernel_size, 1),
            stride=1, padding=(kernel_size // 2, 0), groups=in_channels, bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels * num_layers, self.output_channels, kernel_size=1,
            stride=1, padding=0, bias=True
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, activator: bool) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if activator:
            x = self.activation(x)
            x = self.dropout(x)
        return x


class MultiChannelMultiLayerCausalCNN(nn.Module):
    """Multi-channel multi-layer causal CNN for trend-season decomposition"""
    def __init__(self, in_channels, num_layers, kernel_size=3, padding_mode='replicate',
                 depthwise_kernel_size=3, output_channels=None, use_depthwise_conv=False, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.output_channels = output_channels if output_channels is not None else in_channels
        self.use_depthwise_conv = use_depthwise_conv

        self.conv_layers = nn.ModuleList()
        self.paddings = []
        self._build_network()

        if self.use_depthwise_conv:
            self.trend_depthwise = DepthwiseConvBlock(
                in_channels, num_layers, self.output_channels, depthwise_kernel_size, dropout
            )
            self.season_depthwise = DepthwiseConvBlock(
                in_channels, num_layers, self.output_channels, depthwise_kernel_size, dropout
            )

    def _create_fixed_kernel(self):
        return torch.ones(1, 1, self.kernel_size) / self.kernel_size

    def _build_network(self):
        for i in range(self.num_layers):
            dilation = 2 ** i
            padding = (self.kernel_size - 1) * dilation
            conv = nn.Conv1d(self.in_channels, self.in_channels, self.kernel_size,
                             stride=1, padding=0, dilation=dilation,
                             groups=self.in_channels, bias=False)
            kernel = self._create_fixed_kernel().repeat(self.in_channels, 1, 1)
            conv.weight = nn.Parameter(kernel, requires_grad=False)
            self.conv_layers.append(conv)
            self.paddings.append(padding)

    def _causal_pad(self, x, padding):
        return F.pad(x, (padding, 0), mode=self.padding_mode) if padding > 0 else x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trends, seasons = [], []
        current = x

        for conv, padding in zip(self.conv_layers, self.paddings):
            current = self._causal_pad(current, padding)
            trend = conv(current)[:, :, :current.shape[-1]]
            season = current - trend
            trends.append(trend)
            seasons.append(season)
            current = trend

        trends_tensor = torch.stack(trends, dim=2)
        seasons_tensor = torch.stack(seasons, dim=2)

        if self.use_depthwise_conv:
            trends_tensor = self.trend_depthwise(trends_tensor, activator=True)
            seasons_tensor = self.season_depthwise(seasons_tensor, activator=True)

        return trends_tensor, seasons_tensor


def initialize_dctn_model(num_channels, num_layers, kernel_size=3, padding_mode='replicate',
                          depthwise_kernel_size=3, output_channels=None, use_depthwise_conv=False, dropout=0.1):
    return MultiChannelMultiLayerCausalCNN(
        in_channels=num_channels, num_layers=num_layers, kernel_size=kernel_size,
        padding_mode=padding_mode, depthwise_kernel_size=depthwise_kernel_size,
        output_channels=output_channels, use_depthwise_conv=use_depthwise_conv, dropout=dropout
    )


# ==============================================================================
# 1. Encoder/DDPM Components
# ==============================================================================

class SeasonalStrengthNet(nn.Module):
    """Seasonal strength estimation network"""
    def __init__(self, num_channels, kernel_size=7, dropout=0.1):
        super().__init__()
        dilation = 2
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=padding, dilation=dilation),
            nn.Softplus()
        )

    def forward(self, seasonal: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(seasonal), min=1e-6, max=10.0)


class SimilarityGate(nn.Module):
    """Similarity gating mechanism"""
    def __init__(self, num_channels):
        super().__init__()
        self.upper_bound = nn.Parameter(torch.ones(num_channels) * 0.8)
        self.slope = nn.Parameter(torch.ones(num_channels) * 10.0)
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        slope = torch.clamp(self.slope, min=0.1, max=100.0)
        gate = torch.sigmoid(slope * (similarity - self.upper_bound) + self.bias)
        return torch.clamp(similarity * (1 - gate) + self.upper_bound * gate, 0, 1)


class DDPM(nn.Module):
    """Diffusion model with guidance"""
    def __init__(self, scale_idx, num_channels=3, diffusion_steps=50):
        super().__init__()
        self.num_channels = num_channels
        self.diffusion_steps = diffusion_steps

        self.register_buffer('betas', torch.linspace(0.0001, 0.02, diffusion_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        self.beta_adjustments = nn.Parameter(torch.ones(diffusion_steps) * 0.1)
        self.up_similarity = nn.Parameter(torch.ones(num_channels) * 1)

        self.similarity_gate = SimilarityGate(num_channels)
        self.max_value = 100.0
        self.noise_scale = 0.5

    def _build_guide_net(self, current, target, scale_gamma, up_similarity):
        distance = torch.sqrt(torch.mean((current - target) ** 2, dim=-1) + 1e-8)
        scale_gamma = torch.clamp(scale_gamma, min=0.01, max=10.0)
        similarity = self.similarity_gate(torch.exp(-scale_gamma * distance))
        exp_arg = torch.clamp(100.0 * (similarity - up_similarity), min=-50, max=50)
        transition = 1.0 / (1.0 + torch.exp(exp_arg))
        return (1.0 - similarity) * transition

    def _get_adaptive_noise_schedule(self, k):
        adj = torch.clamp(self.beta_adjustments[k - 1], min=-0.5, max=2.0)
        beta = torch.clamp(self.betas[k - 1] * (1.0 + adj), min=1e-6, max=0.1)
        return 1.0 - beta, beta

    def forward_diffusion_process(self, x_start, target, scale_gamma, guidance_intensity_map, steps=50):
        x = x_start.clone()

        for k in range(1, steps + 1):
            x = torch.nan_to_num(x, nan=0.0)
            guide_weight = self._build_guide_net(x, target, scale_gamma, self.up_similarity).unsqueeze(-1)
            guide_weight = torch.clamp(guide_weight, min=0.0, max=1.0)
            direction = torch.where(x > target, -1.0, 1.0)
            guidance = torch.clamp(guidance_intensity_map, min=0.0, max=5.0)
            epsilon = guide_weight * direction * guidance

            alpha, beta = self._get_adaptive_noise_schedule(k)
            noise = torch.randn_like(x) * self.noise_scale
            x = torch.clamp(torch.sqrt(alpha) * x + epsilon + torch.sqrt(beta) * noise,
                            min=-self.max_value, max=self.max_value)

        loss = torch.clamp(torch.mean((x - target) ** 2, dim=(1, 2)), min=0.0, max=1e6)
        return x, loss

class MultiScaleNoiser(nn.Module):
    """Multi-scale noise injection module"""
    def __init__(self, num_scales, num_channels=3, diffusion_steps=100, strength_net_kernel_size=7, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.num_channels = num_channels

        self.scale_gammas = nn.Parameter(torch.ones(num_scales - 1, num_channels))
        self.strength_net = SeasonalStrengthNet(num_channels, strength_net_kernel_size, dropout)
        self.diffusion_models = nn.ModuleList([
            DDPM(i, num_channels, diffusion_steps) for i in range(num_scales - 1)
        ])

    def forward(self, trend_components, seasonal_components):
        batch_size, _, _, timesteps = trend_components.shape
        device = trend_components.device

        x_state = trend_components[:, :, -1]
        scale_diff_sequences = torch.zeros(batch_size, self.num_scales - 1, self.num_channels, timesteps, device=device)
        losses = []

        for idx in range(self.num_scales - 1):
            target_idx = self.num_scales - 2 - idx
            gamma = torch.clamp(self.scale_gammas[idx], min=0.01, max=10.0)
            guidance = self.strength_net(seasonal_components[:, :, target_idx])

            x_state, loss = self.diffusion_models[idx].forward_diffusion_process(
                x_state, trend_components[:, :, target_idx], gamma, guidance, self.diffusion_models[idx].diffusion_steps
            )
            scale_diff_sequences[:, idx, :, :] = x_state - trend_components[:, :, target_idx]
            losses.append(loss)

        return x_state, scale_diff_sequences, torch.stack(losses, dim=1).mean(dim=1)


class MinimalDenoiser(nn.Module):
    """Minimal denoiser with scale aggregation"""
    def __init__(self, num_channels, num_scales, timesteps, diffusion_steps, hidden_channels=4, dropout=0.1):
        super().__init__()
        betas = torch.linspace(0.0001, 0.02, diffusion_steps)
        alpha_bars = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('sqrt_alpha_bar_T', torch.sqrt(alpha_bars[-1]))
        self.register_buffer('sqrt_one_minus_alpha_bar_T', torch.sqrt(1. - alpha_bars[-1]))

        agg_in = (num_scales - 1) * num_channels
        self.scale_aggregator = nn.Sequential(
            nn.Conv1d(agg_in, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.noise_predictor = nn.Sequential(
            nn.Conv1d(num_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_state, scale_diff_sequences):
        batch, _, _, t = scale_diff_sequences.shape
        flat = scale_diff_sequences.view(batch, -1, t)
        agg = self.scale_aggregator(flat)
        merged = torch.cat([x_state, agg], dim=1)
        noise = self.noise_predictor(merged).repeat(1, x_state.shape[1], 1)

        sqrt_alpha = torch.clamp(self.sqrt_alpha_bar_T, min=1e-6)
        x_denoised = (x_state - self.sqrt_one_minus_alpha_bar_T * noise) / sqrt_alpha
        return torch.clamp(x_denoised, min=-100.0, max=100.0)


class Encoder(nn.Module):
    """Main encoder combining noiser and denoiser"""
    def __init__(self, num_scales, num_channels=42, diffusion_steps=40, timesteps=196,
                 strength_net_kernel_size=7, denoiser_hidden_channels=4, dropout=0.1):
        super().__init__()
        self.noiser = MultiScaleNoiser(num_scales, num_channels, diffusion_steps, strength_net_kernel_size, dropout)
        self.denoiser = MinimalDenoiser(num_channels, num_scales, timesteps, diffusion_steps,
                                        denoiser_hidden_channels, dropout)
        self.num_scales = num_scales

    def forward(self, trend_components, seasonal_components):
        x_state, diff, loss = self.noiser(trend_components, seasonal_components)
        return self.denoiser(x_state, diff), loss


# ==============================================================================
# 2. Base Predictor (Simplified)
# ==============================================================================

class BasePredictor(nn.Module):
    """Simplified base predictor with dropout"""
    def __init__(self, input_channels: int, mlp_hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels, 1)
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.gap(feature).squeeze(-1))


# ==============================================================================
# 3. Final Integrated Model
# ==============================================================================

class DCTNEncoderModel(nn.Module):
    """End-to-end time series forecasting model with DCTN + Encoder + Base Predictor"""
    def __init__(self, num_channels: int, num_layers: int, timesteps: int, kernel_size: int = 3,
                 diffusion_steps: int = 40, use_depthwise_conv: bool = True, strength_net_kernel_size: int = 7,
                 denoiser_hidden_channels: int = 4, mlp_hidden_channels: int = 64,
                 predictor_dropout: float = 0.3, global_dropout: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_layers
        self.timesteps = timesteps

        self.dctn = initialize_dctn_model(
            num_channels, num_layers, kernel_size, output_channels=num_channels,
            use_depthwise_conv=use_depthwise_conv, dropout=global_dropout
        )

        self.encoder = Encoder(
            num_scales=num_layers, num_channels=num_channels, timesteps=timesteps,
            diffusion_steps=diffusion_steps, strength_net_kernel_size=strength_net_kernel_size,
            denoiser_hidden_channels=denoiser_hidden_channels, dropout=global_dropout
        )

        self.predictor = BasePredictor(
            input_channels=num_channels, mlp_hidden_channels=mlp_hidden_channels, dropout=predictor_dropout
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        signals = torch.nan_to_num(signals, nan=0.0)
        trends, seasons = self.dctn(signals)
        feature, _ = self.encoder(trends, seasons)
        feature = torch.nan_to_num(feature, nan=0.0)
        return torch.nan_to_num(self.predictor(feature), nan=0.0)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'DCTNEncoderModel (Base Predictor)',
            'num_channels': self.num_channels,
            'num_layers': self.num_scales,
            'timesteps': self.timesteps,
            'use_depthwise_conv': self.dctn.use_depthwise_conv,
        }

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"Model: {self.get_model_info()['model_type']}")
        print(f"Parameters: {total:,} total, {trainable:,} trainable")
        print(f"Size: {total * 4 / (1024 ** 2):.2f} MB")
        print("=" * 60)


def setup_model(num_channels: int = 14, num_layers: int = 4, timesteps: int = 30,
                kernel_size: int = 3, diffusion_steps: int = 20, use_depthwise_conv: bool = True,
                strength_net_kernel_size: int = 7, denoiser_hidden_channels: int = 32,
                mlp_hidden_channels: int = 64, predictor_dropout: float = 0.3,
                global_dropout: float = 0.15) -> DCTNEncoderModel:
    return DCTNEncoderModel(
        num_channels=num_channels, num_layers=num_layers, timesteps=timesteps,
        kernel_size=kernel_size, diffusion_steps=diffusion_steps, use_depthwise_conv=use_depthwise_conv,
        strength_net_kernel_size=strength_net_kernel_size, denoiser_hidden_channels=denoiser_hidden_channels,
        mlp_hidden_channels=mlp_hidden_channels, predictor_dropout=predictor_dropout, global_dropout=global_dropout
    )


# ==============================================================================
# 4. Main Execution
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing DCTNEncoderModel on {device}")

    model = setup_model().to(device)
    model.print_model_summary()

    signals = torch.randn(2, 14, 30).to(device)

    model.train()
    with torch.no_grad():
        pred_train = model(signals)
    print(f"Train mode prediction: {pred_train.shape}, NaN: {torch.isnan(pred_train).any()}")

    model.eval()
    with torch.no_grad():
        pred_eval = model(signals)
    print(f"Eval mode prediction: {pred_eval.shape}, NaN: {torch.isnan(pred_eval).any()}")

    extreme = torch.randn(2, 14, 30).to(device) * 100
    with torch.no_grad():
        pred_extreme = model(extreme)
    print(f"Extreme input test passed, NaN: {torch.isnan(pred_extreme).any()}")