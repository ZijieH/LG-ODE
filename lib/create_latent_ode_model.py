from lib.gnn_models import GNN
from lib.latent_ode import LatentGraphODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver,GraphODEFunc



def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):


	# dim related
	latent_dim = args.latents # ode output dimension
	rec_dim = args.rec_dims
	input_dim = input_dim
	ode_dim = args.ode_dims #ode gcn dimension

	#encoder related

	encoder_z0 = GNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
						 aggregate=args.rec_attention)  # [b,n_ball,e]


	#ODE related
	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim
	else:
		ode_input_dim = latent_dim


	ode_func_net = GNN(in_dim = ode_input_dim,n_hid =ode_dim,out_dim = ode_input_dim,n_heads=args.n_heads,n_layers=args.gen_layers,dropout=args.dropout,conv_name = args.odenet,aggregate="add")

	gen_ode_func = GraphODEFunc(
		ode_func_net=ode_func_net,
		device=device).to(device)

	diffeq_solver = DiffeqSolver(gen_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    #Decoder related
	decoder = Decoder(latent_dim, input_dim).to(device)


	model = LatentGraphODE(
		input_dim = input_dim,
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		).to(device)

	return model
