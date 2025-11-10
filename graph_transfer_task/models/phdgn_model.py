import torch
from models.gnn_model import BasicModel
from models.phdgn_utils import PortHamiltonianConv


class PHDGN_Model(BasicModel):
    def init_conv(self, in_channels: int, out_channels: int, activation:str, *args, **kwargs):
        self.double_dim = kwargs['double_dim']
        self.final_state = kwargs['pq'] # p, q, pq
        
        if self.double_dim:
            self.inp = 2 * self.inp
        if self.final_state != 'pq':
            self.inp = self.inp // 2

        conv = PortHamiltonianConv(
            in_channels = in_channels*2 if self.double_dim else in_channels,
            num_iters = kwargs['num_iters'],
            epsilon=kwargs['epsilon'],
            activ_fun=activation,
            p_conv_mode=kwargs['p_conv_mode'], # conv_names = ['naive', 'gcn']
            q_conv_mode=kwargs['q_conv_mode'], 
            alpha = kwargs.get('alpha', 0.),
            beta = kwargs.get('beta', 0.),
            dampening_mode = kwargs.get('dampening_mode', None),
            external_mode = kwargs.get('external_mode', None))
        return conv

    def forward(self, x, edge_index):
        h = self.emb(x) if self.emb else x
        
        if self.double_dim:
            h = torch.cat([h,h],dim=1)

        for conv in self.conv:
            h = conv(h, edge_index)

        if self.final_state == 'p': #taking p
            h = h[:,:self.inp]
        elif self.final_state == 'q': # taking q
             h = h[:,self.inp:]
        else: # self.final_state == 'pq'
            pass # x contains both p and q already

        h = self.readout(h)

        return h