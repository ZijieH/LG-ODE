from lib.base_models import VAE_Baseline
import lib.utils as utils
import torch

class LatentGraphODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
				 z0_prior, device, obsrv_std=None):

		super(LatentGraphODE, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.latent_dim =latent_dim



	def get_reconstruction(self, batch_en,batch_de, batch_g,n_traj_samples=1,run_backwards=True):

        #Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_attr,
														  batch_en.edge_index, batch_en.pos, batch_en.edge_same,
														  batch_en.batch, batch_en.y)  # [num_ball,10]

		means_z0 = first_point_mu.repeat(n_traj_samples,1,1) #[3,num_ball,10]
		sigmas_z0 = first_point_std.repeat(n_traj_samples,1,1) #[3,num_ball,10]
		first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0) #[3,num_ball,10]


		first_point_std = first_point_std.abs()

		time_steps_to_predict = batch_de["time_steps"]



		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())
		assert (not torch.isnan(first_point_enc).any())



		# ODE:Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)


        # Decoder:
		pred_x = self.decoder(sol_y)


		all_extra_info = {
			"first_point": (torch.unsqueeze(first_point_mu,0), torch.unsqueeze(first_point_std,0), first_point_enc),
			"latent_traj": sol_y.detach()
		}

		return pred_x, all_extra_info, None









