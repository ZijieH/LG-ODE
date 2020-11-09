'''
 every node have different observations
        train observation length [ob_min, ob_max]
'''

from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=20000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-test', type=int, default=5000,
                    help='Number of test simulations to generate.')
parser.add_argument('--ode', type=int, default=6000,
                    help='Length of trajectory.')
parser.add_argument('--num-test-box', type=int, default=1,
                    help='Length of test set trajectory.')
parser.add_argument('--num-test-extra', type=int, default=40,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_max', type=int, default=52,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_min', type=int, default=40,
                    help='Length of test set trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = "_springs"
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(args,num_sims,isTrain = True):
    loc_all = list()
    vel_all = list()
    edges = list()
    timestamps = list()


    for i in range(num_sims):
        t = time.time()
        #graph generation
        static_graph = sim.generate_static_graph()
        edges.append(static_graph)  # [5,5]



        loc, vel, T_samples = sim.sample_trajectory_static_graph_irregular_difflength_each(args, edges=static_graph,
                                                                                               isTrain=isTrain)
        print(123)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [49,2,5]
        vel_all.append(vel)  # [49,2,5]
        timestamps.append(T_samples)  # [99]


    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)

    return loc_all, vel_all, edges, timestamps


def generate_dataset_charged(args,num_sims,isTrain = True):
    loc_all = list()
    vel_all = list()
    edges = list()
    timestamps = list()


    for i in range(num_sims):
        t = time.time()
        #graph generation
        static_graph,diag_mask = sim.generate_static_graph()
        edges.append(static_graph)  # [5,5]



        loc, vel, T_samples = sim.sample_trajectory_static_graph_irregular_difflength_each(args, edges=static_graph,diag_mask = diag_mask,
                                                                                               isTrain=isTrain)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [49,2,5]
        vel_all.append(vel)  # [49,2,5]
        timestamps.append(T_samples)  # [99]


    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)

    return loc_all, vel_all, edges, timestamps


if args.simulation =="springs":
    print("Generating {} test simulations".format(args.num_test))

    loc_test, vel_test, edges_test, timestamps_test = generate_dataset(args, args.num_test, isTrain=False)
    np.save('loc_test' + suffix + '.npy', loc_test)
    np.save('vel_test' + suffix + '.npy', vel_test)
    np.save('edges_test' + suffix + '.npy', edges_test)
    np.save('times_test' + suffix + '.npy', timestamps_test)

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, timestamps_train = generate_dataset(args, args.num_train, isTrain=True)

    np.save('loc_train' + suffix + '.npy', loc_train)
    np.save('vel_train' + suffix + '.npy', vel_train)
    np.save('edges_train' + suffix + '.npy', edges_train)
    np.save('times_train' + suffix + '.npy', timestamps_train)



elif args.simulation == "charged":

    print("Generating {} test simulations".format(args.num_test))

    loc_test, vel_test, edges_test, timestamps_test = generate_dataset_charged(args, args.num_test, isTrain=False)
    np.save('loc_test' + suffix + '.npy', loc_test)
    np.save('vel_test' + suffix + '.npy', vel_test)
    np.save('edges_test' + suffix + '.npy', edges_test)
    np.save('times_test' + suffix + '.npy', timestamps_test)

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, timestamps_train = generate_dataset_charged(args, args.num_train, isTrain=True)

    np.save('loc_train' + suffix + '.npy', loc_train)
    np.save('vel_train' + suffix + '.npy', vel_train)
    np.save('edges_train' + suffix + '.npy', edges_train)
    np.save('times_train' + suffix + '.npy', timestamps_train)




