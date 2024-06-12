import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin
import pdb


class HEURI_DDIMScheduler(DDIMScheduler, SchedulerMixin, ConfigMixin):

    def set_timesteps(self, num_inference_steps: int, t_start: int, device: Union[str, torch.device] = None):
            """
            Sets the discrete timesteps used for the diffusion chain (to be run before inference).

            Args:
                num_inference_steps (`int`):
                    The number of diffusion steps used when generating samples with a pre-trained model.
            """

            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
                )

            timesteps = torch.from_numpy(timesteps).to(device)


            naive_sampling_step = num_inference_steps //2

            # TODO for debug
            # naive_sampling_step = 0

            self.naive_sampling_step = naive_sampling_step

            timesteps[:naive_sampling_step] = timesteps[naive_sampling_step] # refine on step 5 for 5 steps, then backward from step 6

            timesteps = [timestep + 1 for timestep in timesteps]

            self.timesteps = timesteps
            self.gap = self.config.num_train_timesteps // self.num_inference_steps
            self.prev_timesteps = [timestep for timestep in self.timesteps[1:]]
            self.prev_timesteps.append(torch.zeros_like(self.prev_timesteps[-1]))

    def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            prev_timestep: int,
            sample: torch.Tensor,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator=None,
            cur_step=None,
            variance_noise: Optional[torch.Tensor] = None,
            gaus_noise: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[DDIMSchedulerOutput, Tuple]:
            """
            Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
            process from the learned model outputs (most often the predicted noise).

            Args:
                model_output (`torch.Tensor`):
                    The direct output from learned diffusion model.
                timestep (`float`):
                    The current discrete timestep in the diffusion chain.
                pre_timestep (`float`):
                    next_timestep
                sample (`torch.Tensor`):
                    A current instance of a sample created by the diffusion process.
                eta (`float`):
                    The weight of noise for added noise in diffusion step.
                use_clipped_model_output (`bool`, defaults to `False`):
                    If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                    because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                    clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                    `use_clipped_model_output` has no effect.
                generator (`torch.Generator`, *optional*):
                    A random number generator.
                variance_noise (`torch.Tensor`):
                    Alternative to generating noise with `generator` by directly providing the noise for the variance
                    itself. Useful for methods such as [`CycleDiffusion`].
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

            Returns:
                [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
                    If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                    tuple is returned where the first element is the sample tensor.

            """
            if self.num_inference_steps is None:
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )

            # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
            # Ideally, read DDIM paper in-detail understanding

            # Notation (<variable name> -> <name in paper>
            # - pred_noise_t -> e_theta(x_t, t)
            # - pred_original_sample -> f_theta(x_t, t) or x_0
            # - std_dev_t -> sigma_t
            # - eta -> η
            # - pred_sample_direction -> "direction pointing to x_t"
            # - pred_prev_sample -> "x_t-1"

            # 1. get previous step value (=t-1)

            # trick from heuri_sampling
            if cur_step == self.naive_sampling_step  and timestep == prev_timestep:
                timestep += self.gap


            prev_timestep = prev_timestep  # NOTE naive sampling

            # 2. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

            beta_prod_t = 1 - alpha_prod_t

            # 3. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                pred_epsilon = model_output
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
                pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction`"
                )

            # 4. Clip or threshold "predicted x_0"
            if self.config.thresholding:
                pred_original_sample = self._threshold_sample(pred_original_sample)

            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)


            if use_clipped_model_output:
                # the pred_epsilon is always re-derived from the clipped x_0 in Glide
                pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            if eta > 0:
                if variance_noise is not None and generator is not None:
                    raise ValueError(
                        "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                        " `variance_noise` stays `None`."
                    )

                if variance_noise is None:
                    variance_noise = randn_tensor(
                        model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                    )
                variance = std_dev_t * variance_noise

                prev_sample = prev_sample + variance

            if cur_step < self.naive_sampling_step:
                prev_sample = self.add_noise(pred_original_sample, torch.randn_like(pred_original_sample), timestep)

            if not return_dict:
                return (prev_sample,)


            return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)



    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples