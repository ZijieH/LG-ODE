import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence

class ParseData(object):

    def __init__(self, dataset_path,args,suffix='_springs5',mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.cutting_edge = args.cutting_edge
        self.num_pre = args.extrap_num

        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)



    def load_data(self,sample_percent,batch_size,data_type="train"):
        self.batch_size = batch_size
        self.sample_percent = sample_percent
        if data_type == "train":
            cut_num = 20000
        else:
            cut_num = 5000


        # Loading Data
        loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy')[:cut_num]
        vel = np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy')[:cut_num]
        edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy')[:cut_num]  # [500,5,5]
        times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy')[:cut_num]  # 【500，5]

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
        print("number graph in   "+data_type+"   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)

        if self.suffix == "_springs5" or self.suffix == "_charged5":
            # Normalize features to [-1, 1], across test and train dataset

            if self.max_loc == None:
                loc, max_loc, min_loc = self.normalize_features(loc,
                                                                self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
                self.max_loc = max_loc
                self.min_loc = min_loc
                self.max_vel = max_vel
                self.min_vel = min_vel
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        else:
            self.timelength = 49



        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="interp":
            loc_en,vel_en,times_en = self.interp_extrap(loc,vel,times,self.mode,data_type)
            loc_de = loc_en
            vel_de = vel_en
            times_de = times_en
        elif self.mode == "extrap":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = self.interp_extrap(loc,vel,times,self.mode,data_type)

        #Encoder dataloader
        series_list_observed, loc_observed, vel_observed, times_observed = self.split_data(loc_en, vel_en, times_en)
        if self.mode == "interp":
            time_begin = 0
        else:
            time_begin = 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin)


        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)


        # Decoder Dataloader
        if self.mode=="interp":
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            series_list_de = self.decoder_data(loc_de,vel_de,times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]


        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch



    def interp_extrap(self,loc,vel,times,mode,data_type):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)
        if mode =="interp":
            if data_type== "test":
                # get ride of the extra nodes in testing data.
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed,vel_observed,times_observed/self.total_step
            else:
                return loc,vel,times/self.total_step


        elif mode == "extrap":# split into 2 parts and normalize t seperately
            loc_observed = np.ones_like(loc)
            vel_observed = np.ones_like(vel)
            times_observed = np.ones_like(times)

            loc_extrap = np.ones_like(loc)
            vel_extrap = np.ones_like(vel)
            times_extrap = np.ones_like(times)

            if data_type == "test":
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                        loc_extrap[i][j] = loc[i][j][-self.num_pre:]
                        vel_extrap[i][j] = vel[i][j][-self.num_pre:]
                        times_extrap[i][j] = times[i][j][-self.num_pre:]
                times_observed = times_observed/self.total_step
                times_extrap = (times_extrap - self.total_step)/self.total_step
            else:
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        times_current_mask = np.where(times_current<self.total_step//2,times_current,0)
                        num_observe_current = np.argmax(times_current_mask)+1

                        loc_observed[i][j] = loc[i][j][:num_observe_current]
                        vel_observed[i][j] = vel[i][j][:num_observe_current]
                        times_observed[i][j] = times[i][j][:num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        vel_extrap[i][j] = vel[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:]

                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step//2) / self.total_step

            return loc_observed,vel_observed,times_observed,loc_extrap,vel_extrap,times_extrap


    def split_data(self,loc,vel,times):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)

        # split encoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])




        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]

            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1)
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))

            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))


        return series_list, loc_observed, vel_observed, times_observed

    def decoder_data(self, loc, vel, times):

        # split decoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list



    def transfer_data(self,loc, vel, edges, times, time_begin=0):
        data_list = []
        graph_list = []
        edge_size_list = []

        for i in tqdm(range(self.num_graph)):
            data_per_graph, edge_data, edge_size = self.transfer_one_graph(loc[i], vel[i], edges[i], times[i],
                                                                           time_begin=time_begin)
            data_list.append(data_per_graph)
            graph_list.append(edge_data)
            edge_size_list.append(edge_size)

        print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=self.batch_size)
        graph_loader = DataLoader(graph_list, batch_size=self.batch_size)

        return data_loader, graph_loader

    def transfer_one_graph(self,loc, vel, edge, time, time_begin=0, mask=True, forward=False):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creeating pos 【N】
        # forward: t0=0;  otherwise: t0=tN/2



        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix == "_springs5" or self.suffix == "_charged5":
                max_gap = (self.total_step - 40*self.sample_percent) /self.total_step
            else:
                max_gap = (self.total_step - 30 * self.sample_percent) / self.total_step
        else:
            max_gap = 100


        if self.mode=="interp":
            forward= False
        else:
            forward=True


        y = np.zeros(self.num_atoms)
        x = list()
        x_pos = list()
        node_number = 0
        node_time = dict()
        ball_nodes = dict()

        # Creating x, y, x_pos
        for i, ball in enumerate(loc):
            loc_ball = ball
            vel_ball = vel[i]
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball)

            # Creating x and x_pos, by tranverse each ball's sequence
            for j in range(loc_ball.shape[0]):
                xj_feature = np.concatenate((loc_ball[j], vel_ball[j]))
                x.append(xj_feature)

                x_pos.append(time_ball[j] - time_begin)
                node_time[node_number] = time_ball[j]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1

        '''
         matrix computing
         '''
        # Adding self-loop
        edge_with_self_loop = edge + np.eye(self.num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0)
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos)))

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if edge_with_self_loop[i][j] == 1:
                    sender_index_start = int(np.sum(y[:i]))
                    sender_index_end = int(sender_index_start + y[i])
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        if mask == None:
            edge_time_matrix = np.where(abs(edge_time_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are thosewhose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are thosewhose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        edge_index, edge_attr = self.convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge)



        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x = torch.ones(self.num_atoms),edge_index = graph_index_original)


        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=x_pos, edge_same=edge_is_same)
        edge_size = edge_index.shape[1]

        return graph_data,edge_data,edge_size

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True) #【including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])


        for b, ( tt, vals, mask) in enumerate(batch):

            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()


        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
            }
        return data_dict

    def normalize_features(self,inputs, num_balls):
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr





