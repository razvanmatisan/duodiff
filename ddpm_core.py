from collections import defaultdict

import torch
from tqdm import tqdm

from benchmark import get_gflops


def early_exit(lambda_threshold, model_output, earliest_exit_index, verbose=False):
    predicted_noise, u_i, g_i = model_output
    lambda_threshold = torch.tensor(lambda_threshold, device=u_i.device)
    bs, L, C, H, W = g_i.shape  # batch size, number of layers, channels, height, width

    # Compute the average of u_i over the spatial dimensions (C, H, W)
    average_u_i = u_i.mean(dim=(-1, -2, -3))  # Resulting shape (bs, L)

    # Calculate 1-average_u_i and compare with lambda
    condition = (1 - average_u_i) > lambda_threshold

    # Find the first layer where the condition is True for each element in the batch
    max_value = L  # Use L as a sentinel value indicating no valid layer was found
    masked_condition = torch.where(
        condition, torch.arange(L).unsqueeze(0).to(u_i.device), max_value
    )
    first_true_index = masked_condition.min(dim=1)[0]  # Resulting shape (bs,)

    # Apply the earliest_exit_index constraint
    adjusted_first_true_index = torch.clamp(first_true_index, min=earliest_exit_index)

    # Check if condition was never true (if index is max_value)
    never_met = first_true_index >= L

    # If condition is met, gather the corresponding g_i for each first occurrence, otherwise use predicted_noise
    final_selected_noise = torch.zeros_like(predicted_noise)
    for idx in range(bs):
        if never_met[idx]:
            final_selected_noise[idx] = predicted_noise[idx]
        else:
            final_selected_noise[idx] = g_i[idx, adjusted_first_true_index[idx]]

    # Prepare exit_layer_indices with the actual exit layers; if never met, assign L
    exit_layers = torch.where(
        never_met,
        torch.tensor(L, device=first_true_index.device),
        adjusted_first_true_index,
    )
    if verbose:
        print("Condition is:", condition)
        print("Average u_i is:", average_u_i)
        print("First true index is:", first_true_index)
        print("Adjusted first true index is:", adjusted_first_true_index)
        print("Exit layers are:", exit_layers)

    return final_selected_noise, exit_layers


class NoiseScheduler:
    def __init__(
        self, beta_init=1e-4, beta_final=0.02, beta_steps=1000, variance_mode="beta"
    ):
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.beta_steps = beta_steps
        self.variance_mode = variance_mode  # can be 'beta' or 'beta_tilde'

        self.betas = torch.linspace(beta_init, beta_final, beta_steps)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alphas_bar[:-1]])
        self.betas_tilde = (
            (1 - self.alpha_bar_prev) / (1 - self.alphas_bar) * self.betas
        )

    def sigma_squared(self):
        # Calculate sigma squared based on the chosen variance mode
        if self.variance_mode == "beta":
            return self.betas
        elif self.variance_mode == "beta_tilde":
            return self.betas_tilde
        else:
            raise ValueError("Invalid variance mode. Choose 'beta' or 'beta_tilde'.")

    def add_noise(self, x0, timesteps):
        """
        Add noise to the clean data x0 for given timesteps.

        Parameters:
        x0 (Tensor): The clean data batch.
        timesteps (Tensor): A batch of timesteps, one for each data point in x0.

        Returns:
        Tensor: The noisy data batch.
        """
        # Ensure timesteps are within valid range
        device = x0.device
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.beta_steps):
            raise ValueError("Timesteps must be within the range of 0 and beta_steps-1")
        # Get the corresponding alpha_bar values for the given timesteps
        alpha_bar_t = self.alphas_bar[timesteps]
        # Reshape alpha_bar_t to have the same number of dimensions as x0
        alpha_bar_t = self._match_shape(alpha_bar_t, x0.shape).to(device)
        # Sample noise ε from N(0, I) for each data point in the batch
        noise = torch.randn_like(x0).to(x0.device)
        # Calculate the noisy version of x0 for each data point
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noise, noisy_x

    def sample(
        self,
        model,
        num_steps,
        data_shape,
        num_samples,
        seed,
        model_type,
        keep_initial_timesteps,
        benchmarking=False,
        train_mode=False,
        time_frequency=None,
        space_frequency=None,
        coordinates=None,
        field_noise_dict=None,
    ):
        """
        Generate samples using the reverse diffusion process.

        Parameters:
        model (nn.Module): The diffusion model.
        num_steps (int): The number of steps to iterate over.
        data_shape (tuple): The shape of the data.
        num_samples (int): The number of samples to generate.
        model_type (str): The type of model used. Can be 'perceiver' or 'huggingface'.
        time_frequency (int): The number of frequencies to use for encoding time.
        space_frequency (int): The number of frequencies to use for encoding space.

        Returns:
        Tensor: Batch of generated samples.
        """
        logging_dict = defaultdict(list)
        # Get the device from the model
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        model.eval()

        with torch.no_grad():
            # Step 1: Generate the initial batch of samples
            x_t = torch.randn(
                (num_samples, *data_shape), generator=generator, device=device
            )
            # Step 2: Iterate from num_steps down to 1
            for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling Progress"):
                # Step 2.5: Calculate the noise
                if model_type == "uvit":
                    if keep_initial_timesteps:
                        time_tensor = torch.tensor([t], device=device).repeat(
                            num_samples
                        )
                    else:
                        time_tensor = torch.tensor(
                            [t / num_steps], device=device
                        ).repeat(num_samples)

                    eps = model(x_t, time_tensor)
                elif model_type == "deediff_uvit":
                    if train_mode:
                        model.train()
                    if keep_initial_timesteps:
                        time_tensor = torch.tensor([t], device=device).repeat(
                            num_samples
                        )
                    else:
                        time_tensor = torch.tensor(
                            [t / num_steps], device=device
                        ).repeat(num_samples)

                    model_output = model(x_t, time_tensor)
                    eps = model_output[0]
                    logging_dict["classifier_outputs"].append(model_output[1])

                    if not model.training:
                        logging_dict["early_exit_layers"].append((t, model_output[2]))
                    else:
                        logging_dict["outputs"].append(model_output[2])

                # [Optional] Step 2.6: Benchmarking
                if benchmarking:
                    gflops = get_gflops(model, x_t, time_tensor)
                    logging_dict["benchmarking"].append((t, gflops))

                # Step 3: Sample z from N(0, I) if t > 1, else z = 0
                z = (
                    torch.randn(x_t.size(), generator=generator, device=device)
                    if t > 1
                    else torch.zeros_like(x_t)
                )

                # Get the corresponding values for alpha_t and alpha_bar_t
                alpha_t = self._match_shape(self.alphas[t : t + 1], x_t.shape).to(
                    device
                )
                alpha_t = alpha_t.repeat(num_samples, *((1,) * len(data_shape)))

                alpha_bar_t = self._match_shape(
                    self.alphas_bar[t : t + 1], x_t.shape
                ).to(device)
                alpha_bar_t = alpha_bar_t.repeat(num_samples, *((1,) * len(data_shape)))

                sigma_squared_t = self._match_shape(
                    self.sigma_squared()[t : t + 1], x_t.shape
                ).to(device)
                sigma_t = torch.sqrt(sigma_squared_t)

                lmbd = 0 if t == 0 else 1
                x_t_minus_1 = (
                    torch.sqrt(1 / alpha_t)
                    * (x_t - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * eps)
                ) + lmbd * sigma_t * z

                if model.training:
                    denoised_images = []
                    for noise in logging_dict["outputs"][-1]:
                        denoised_image = (
                            torch.sqrt(1 / alpha_t)
                            * (
                                x_t
                                - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * noise
                            )
                        ) + lmbd * sigma_t * z

                        denoised_images.append(denoised_image)
                    logging_dict["denoised_images"].append(denoised_images)
                # Update x_t for the next iteration
                x_t = x_t_minus_1
                logging_dict["samples_over_time"].append(x_t)
            # Step 6: Return the batch of generated samples

            model.train()
            return x_t, logging_dict

    def early_exit_sample(
        self,
        model,
        num_steps,
        data_shape,
        num_samples,
        model_type="perceiver",
        lambda_threshold=0.5,
    ):
        """
        Generate samples using the reverse diffusion process.

        Parameters:
        model (nn.Module): The diffusion model.
        num_steps (int): The number of steps to iterate over.
        data_shape (tuple): The shape of the data.
        num_samples (int): The number of samples to generate.
        model_type (str): The type of model used. Can be 'perceiver' or 'huggingface'.
        time_frequency (int): The number of frequencies to use for encoding time.
        space_frequency (int): The number of frequencies to use for encoding space.

        Returns:
        Tensor: Batch of generated samples.
        """
        # Get the device from the model
        exit_layer_list = []
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            # Step 1: Generate the initial batch of samples
            x_t = torch.randn((num_samples, *data_shape)).to(device)
            # Step 2: Iterate from num_steps down to 1
            for t in tqdm(range(num_steps - 1, -1, -1), desc="Sampling Progress"):
                # Step 2.5: Calculate the noise
                if model_type == "huggingface":
                    eps = model(x_t, t, return_dict=False)[0]
                elif model_type == "UViT":
                    t_normalized = t / num_steps
                    time_tensor = (
                        torch.tensor([t_normalized]).repeat(num_samples).to(device)
                    )
                    eps = model(x_t, time_tensor)
                elif model_type == "DeeDiff_UViT":
                    t_normalized = t / num_steps
                    time_tensor = (
                        torch.tensor([t_normalized]).repeat(num_samples).to(device)
                    )
                    model_output = model(x_t, time_tensor)
                    eps, exit_layers = early_exit(
                        lambda_threshold, model_output, earliest_exit_index=1
                    )
                    exit_layer_list.append(exit_layers)
                # Step 3: Sample z from N(0, I) if t > 1, else z = 0
                z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

                # Get the corresponding values for alpha_t and alpha_bar_t
                alpha_t = self._match_shape(self.alphas[t : t + 1], x_t.shape).to(
                    device
                )
                alpha_t = alpha_t.repeat(num_samples, *((1,) * len(data_shape)))

                alpha_bar_t = self._match_shape(
                    self.alphas_bar[t : t + 1], x_t.shape
                ).to(device)
                alpha_bar_t = alpha_bar_t.repeat(num_samples, *((1,) * len(data_shape)))

                sigma_squared_t = self._match_shape(
                    self.sigma_squared()[t : t + 1], x_t.shape
                ).to(device)
                sigma_t = torch.sqrt(sigma_squared_t)
                x_t_minus_1 = (
                    torch.sqrt(1 / alpha_t)
                    * (x_t - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * eps)
                ) + sigma_t * z
                # Update x_t for the next iteration
                x_t = x_t_minus_1
            # Step 6: Return the batch of generated samples
            return x_t, exit_layer_list

    def _match_shape(self, tensor, target_shape):
        """
        Reshape the given tensor to have the same number of dimensions as the target shape,
        with a size of 1 for all new dimensions.

        Parameters:
        tensor (Tensor): The tensor to reshape.
        target_shape (tuple): The shape to match.

        Returns:
        Tensor: The reshaped tensor.
        """
        return tensor.view(tensor.shape[0], *((1,) * (len(target_shape) - 1)))

    def set_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.betas_tilde = self.betas_tilde.to(device)


def sample_images(model, n_samples, height, width, num_steps, seed):
    device = model.device
    generator = torch.Generator(device="cpu").manual_seed(seed)

    betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_previous = torch.cat(
        [torch.tensor([1.0], device=device), alphas_bar[:-1]]
    )
    betas_tilde = betas * (1 - alphas_bar_previous) / (1 - alphas_bar)

    x = torch.randn(n_samples, 3, height, width, generator=generator).to(device)
    for t in range(num_steps, 0, -1):
        with torch.inference_mode():
            time_tensor = t * torch.ones(n_samples, device=device)
            epsilon = model(x, time_tensor)

        alpha_t = alphas[t - 1]
        alpha_bar_t = alphas_bar[t - 1]
        sigma_t = torch.sqrt(betas_tilde[t - 1])

        z = torch.rand(x.size(), generator=generator).to(device) if t > 1 else 0
        x = (
            1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / (1 - alpha_bar_t) * epsilon)
            + sigma_t * z
        )

    return x
