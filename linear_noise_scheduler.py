import torch

class LinearNoiseScheduler:
    def __init__(self, num_time_steps, beta_start, beta_end, device = "cuda") -> None:
        self.num_time_steps = num_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # making beta evenly spaced from start to end
        # for beta_start = 0.0001 
        # betas = [0.0001, 0.0002, 0.0003, 0.0004 .....] over number of time steps till beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps)
        
        # alpha = 1 - beta
        # alphas = [0.9999, 0.9998, 0.9997, 0.9996 .....] over number of time steps till beta_start
        self.alphas = 1 - self.betas

        # alpha_tilda = multiplication of all alphas i.e. cumulative product at i
        self.alpha_tilda = torch.cumprod(self.alphas, dim=0)

        # now we want square root of all alpha_tilda and square root of 1 - alpha_tilda
        self.sqrt_alpha_tilda = torch.sqrt(self.alpha_tilda).to(device)
        self.sqrt_1_minus_alpha_tilda = torch.sqrt(1. - self.sqrt_alpha_tilda).to(device)


    def add_noise(self, x0, epsilon_noise, t):
        """
        x at time step t after adding noise
        xt = square_root_alpha_tilda * x0 + square_root_1_minus_alpha_tilda * epsilon_noise

        return: image at time step t with added noise 
        """
        
        x0_shape = x0.shape                         # B x C x H x W
        batch_size = x0_shape[0]                    # batch size

        sqrt_alpha_tilda = self.sqrt_alpha_tilda[t].reshape(batch_size) 
        sqrt_1_minus_alpha_tilda = self.sqrt_1_minus_alpha_tilda[t].reshape(batch_size)


        for _ in range(len(x0_shape) - 1):
            sqrt_alpha_tilda = sqrt_alpha_tilda.unsqueeze(-1)
            sqrt_1_minus_alpha_tilda = sqrt_1_minus_alpha_tilda.unsqueeze(-1)

        return sqrt_alpha_tilda * x0 + sqrt_1_minus_alpha_tilda * epsilon_noise
    

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        sampling at time t with the predicted epsilon_noise ~ q( xt-1 | xt, x0 )

        return mean and x at time t
        """
        # at t=0
        x0_tilda =  ( xt - self.sqrt_1_minus_alpha_tilda[t] * noise_pred ) / self.sqrt_alpha_tilda[t]
        x0_tilda = torch.clamp(x0_tilda, min=-1, max=1)             # why clamped ?

        # mu_tilda
        mean_tilda = ( xt - ( ( self.betas[t] * noise_pred ) / self.sqrt_1_minus_alpha_tilda[t] ) ) / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean_tilda, x0_tilda
        else:
            variance = ( (1. - self.alpha_tilda[t-1]) / (1. - self.alpha_tilda[t]) ) * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean_tilda + sigma * z, x0_tilda









