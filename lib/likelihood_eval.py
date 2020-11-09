from lib.likelihood_eval import *
import torch





def gaussian_log_likelihood(mu, data, obsrv_std):
	log_p = ((mu - data) ** 2) / (2 * obsrv_std * obsrv_std)
	neg_log_p = -1*log_p
	return neg_log_p

def generate_time_weight(n_timepoints,n_dims):
	value_min = 1
	value_max = 2
	interval = (value_max - value_min)/(n_timepoints-1)

	value_list = [value_min + i*interval for i in range(n_timepoints)]
	value_list= torch.FloatTensor(value_list).view(-1,1)

	value_matrix= torch.cat([value_list for _ in range(n_dims)],dim = 1)

	return value_matrix


def compute_masked_likelihood(mu, data, mask, likelihood_func,temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	if temporal_weights!= None:
		weight_for_times = torch.cat([temporal_weights for _ in range(n_dims)],dim = 1)
		weight_for_times = weight_for_times.to(mu.device)
		weight_for_times = weight_for_times.repeat(n_traj_samples, n_traj, 1, 1)
		log_prob_masked = torch.sum(log_prob * mask * weight_for_times, dim=2)  # [n_traj, n_traj_samples, n_dims]
	else:
		log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]


	timelength_per_nodes = torch.sum(mask.permute(0,1,3,2),dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	log_prob_masked_normalized = torch.div(log_prob_masked , timelength_per_nodes) #【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res = torch.mean(log_prob_masked_normalized, -1) # 【n_traj_sample, n_traj], average among features.
	res = res.transpose(0,1)
	return res


def compute_masked_likelihood_old(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()


	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]

	timelength_per_nodes = torch.sum(mask.permute(0, 1, 3, 2), dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	log_prob_masked_normalized = torch.div(log_prob_masked,
										   timelength_per_nodes)  # 【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res = torch.mean(log_prob_masked_normalized, -1)  # 【n_traj_sample, n_traj], average among features.
	res = res.transpose(0, 1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask,temporal_weights=None):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
	res = compute_masked_likelihood(mu, data,mask, func,temporal_weights)
	return res


def mse(mu,data):
	return  (mu - data) ** 2


def compute_mse(mu, data, mask):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	res = compute_masked_likelihood(mu, data, mask, mse)
	return res

	

