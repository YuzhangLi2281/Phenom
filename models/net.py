import torch
from torch_geometric.nn import global_add_pool
from torch.nn import Embedding, Linear
import torch.nn.functional as F
from torch_geometric.utils import scatter
from models.layers import (
    MLP,
    Drug_PNAConv,
    Protein_PNAConv,
    PosLinear,
    GCNCluster,
    dropout_edge,
    dropout_node,
    LayerNorm,
)
from models.pooling import MotifPool, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, softmax
from models.enzyme_prediction import EnzymePredictionHead
from models.interaction_layer import SuperNode2TokenConv
from models.interaction_layer import PhenomInteraction
from models.interaction_layer import CatalyticInteraction
from torch_geometric.nn.norm import GraphNorm






class Phenom(torch.nn.Module):
    """Sequence-based graph network with an interpretable interaction fingerprint
    and native Gaussian prediction heads.

    The six-output checkpoint layout is retained for tensor compatibility;
    production inference exposes only the trained kcat and Km dimensions.
    """
    def __init__(self, mol_deg, prot_deg,
                 # MOLECULE
                 mol_in_channels=43,  mol_edge_channels=30,
                 clique_pe_walk_length=20,
                 prot_in_channels=33, prot_evo_channels=1024,
                 hidden_channels=200,
                 pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,                
                 K = [5,10,20],
                 t = 1,
                 # training
                 heads=5, 
                 dropout=0,
                 dropout_attn_score=0.2,
                 drop_residue=0,
                 dropout_cluster_edge=0,
                 gaussian_noise=0,
                 protein_token_dropout=0,
                 ligand_fingerprint_dropout=0,
                 prediction_head_layout='task_independent',
                 
                 device='cuda:0'):
        super().__init__()
        self.total_layer = total_layer
        self.hidden_channels = hidden_channels
        self.heads = heads

        if isinstance(K, int):
            K = [K]*total_layer 

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 

        self.clique_encoder = Embedding(4, hidden_channels)

        # PROTEIN IN FEAT
        # Keep this historical state-dict key: it now receives ESMC-600M
        # residue features (1152 channels), not EnzGFM embeddings.
        self.enzgfm_adapter = Linear(prot_evo_channels, 1280, bias=False)
        self.prot_evo = MLP([1280, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 
    
        ### MOLECULE and PROTEIN
        self.mol_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()

        self.mol_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()

        self.interaction = PhenomInteraction(
            hidden_channels, heads, total_layer, dropout_attn_score
        )
        self.enzyme_aware_interaction = True
        self.catalytic_interaction = CatalyticInteraction(
            hidden_channels, heads, max_residues=64, dropout=0.1
        )
        self.big2residue = SuperNode2TokenConv()

        self.num_cluster = K
        self.t = t
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()

        self.atom_lins = torch.nn.ModuleList() 
        self.residue_lins = torch.nn.ModuleList() 
        
        self.c2a_mlps = torch.nn.ModuleList()
        self.c2r_mlps = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels,edge_channels=mol_edge_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.prot_convs.append(Protein_PNAConv(
                prot_deg, hidden_channels, edge_channels=hidden_channels, # None,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.cluster.append(GCNCluster([hidden_channels, hidden_channels*2, self.num_cluster[idx]], in_norm=True))
            self.mol_pools.append(MotifPool(hidden_channels, heads, clique_pe_walk_length,dropout_attn_score))
            self.prot_norms.append(LayerNorm(hidden_channels))
            
            self.atom_lins.append( Linear(hidden_channels, hidden_channels, bias=False) )
            self.residue_lins.append( Linear(hidden_channels, hidden_channels, bias=False) )

            self.c2a_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))

            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))

        self.atom_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value= 1 / heads) #(heads * total_layer))
        self.residue_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value= 1 / heads) #(heads * total_layer))
        
        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        self.protein_global_score = Linear(hidden_channels, 1)
        self.protein_global = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels * 2),
            torch.nn.SiLU(),
            Linear(hidden_channels * 2, hidden_channels),
        )
        self.ligand_global = torch.nn.Sequential(
            Linear(2215, hidden_channels * 2),
            torch.nn.SiLU(),
            Linear(hidden_channels * 2, hidden_channels),
        )
        self.ligand_language_global = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Linear(768, hidden_channels),
            torch.nn.SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        torch.nn.init.zeros_(self.ligand_language_global[-1].weight)
        torch.nn.init.zeros_(self.ligand_language_global[-1].bias)
        self.fusion = torch.nn.Sequential(
            Linear(hidden_channels * 6, hidden_channels * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_channels * 4, hidden_channels * 2),
        )
        self.fusion_norm = torch.nn.LayerNorm(hidden_channels * 2)
        self.catalytic_fusion = torch.nn.Sequential(
            Linear(hidden_channels * 6, hidden_channels * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_channels * 4, hidden_channels * 2),
        )
        torch.nn.init.zeros_(self.catalytic_fusion[-1].weight)
        torch.nn.init.zeros_(self.catalytic_fusion[-1].bias)

        self.prediction_head = EnzymePredictionHead(
            hidden_channels * 2, hidden_channels, layout=prediction_head_layout
        )

        self.dropout = dropout
        self.drop_residue = drop_residue
        self.gaussian_noise = gaussian_noise
        self.dropout_cluster_edge = dropout_cluster_edge
        self.protein_token_dropout = protein_token_dropout
        self.ligand_fingerprint_dropout = ligand_fingerprint_dropout
        self.device = device


    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()
        self.clique_encoder.reset_parameters()
        self.enzgfm_adapter.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()
        
        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()

            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()

            self.cluster[idx].reset_parameters()
            self.mol_pools[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()

            self.atom_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()

            self.c2a_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()
    
        self.atom_attn_lin.reset_parameters()
        self.residue_attn_lin.reset_parameters()
        self.interaction.reset_parameters()
        for module in self.catalytic_interaction.modules():
            if module is not self.catalytic_interaction and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()

        self.protein_global_score.reset_parameters()
        for module in (*self.protein_global, *self.ligand_global, *self.fusion):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.ligand_language_global:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        torch.nn.init.zeros_(self.ligand_language_global[-1].weight)
        torch.nn.init.zeros_(self.ligand_language_global[-1].bias)
        self.fusion_norm.reset_parameters()
        for module in self.catalytic_fusion:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        torch.nn.init.zeros_(self.catalytic_fusion[-1].weight)
        torch.nn.init.zeros_(self.catalytic_fusion[-1].bias)

        self.prediction_head.reset_parameters()

    def forward(self, data, save_cluster=False, protein_only=False):
        """Return all head outputs, auxiliary graph losses and attention traces."""
        # Protein
        residue_x=data.prot_node_aa
        residue_evo_x=self.enzgfm_adapter(data.prot_node_evo)
        residue_edge_index=data.prot_edge_index
        residue_edge_weight=data.prot_edge_weight
        # Mol-Protein Interaction batch
        prot_batch=data.prot_node_aa_batch

        if not protein_only:
            mol_x=data.mol_x
            mol_x_feat=data.mol_x_feat
            bond_x=data.mol_edge_attr
            atom_edge_index=data.mol_edge_index
            clique_x=data.clique_x
            clique_x_pe=data.clique_x_pe
            atom2clique_index=data.atom2clique_index
            mol_batch=data.mol_x_batch
            clique_batch=data.clique_x_batch

        # Init variables        
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)
        # residue_edge_attr = None
        # PROTEIN Featurize
        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)
        if self.training and self.protein_token_dropout:
            keep = torch.rand(
                residue_x.size(0), 1, device=residue_x.device
            ).ge(self.protein_token_dropout)
            residue_x = residue_x * keep / (1.0 - self.protein_token_dropout)
        initial_residue_x = residue_x
        protein_global_attention = softmax(self.protein_global_score(initial_residue_x), prot_batch)
        protein_global = self.protein_global(torch.cat([
            scatter(initial_residue_x, prot_batch, dim=0, reduce="mean"),
            global_add_pool(initial_residue_x * protein_global_attention, prot_batch),
        ], dim=-1))
        if not protein_only:
            atom_x = self.atom_type_encoder(mol_x.view(-1)) + self.atom_feat_encoder(mol_x_feat)
            clique_x = self.clique_encoder(clique_x.view(-1))

        # clique_id = clique_x.squeeze()
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)

        clique_scores = []
        residue_scores = []
        interaction_edges = []
        interaction_scores = []
        layer_s = {}
        # MOLECULE-PROTEIN Layers
        for idx in range(self.total_layer):
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)

            if not protein_only:
                atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
                drug_x, clique_x, clique_score = self.mol_pools[idx](
                    atom_x, clique_x, clique_x_pe, atom2clique_index, clique_batch
                )
                clique_scores.append(clique_score)

                atom_s = self.cluster[idx](atom_x, atom_edge_index)
                atom_hx, atom_mask = to_dense_batch(atom_x, mol_batch)
                atom_s, _ = to_dense_batch(atom_s, mol_batch)
                atom_adj = to_dense_adj(atom_edge_index, mol_batch)
                _, _, _, mol_cl_loss, mol_o_loss = dense_mincut_pool(
                    atom_hx, atom_adj, atom_s, atom_mask
                )

            ## Cluster protein residues
            dropped_residue_edge_index, _ = dropout_edge(residue_edge_index, p=self.dropout_cluster_edge, 
                                                         force_undirected=True,training=self.training)
            s = self.cluster[idx](residue_x, dropped_residue_edge_index)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)

            if save_cluster:
                layer_s[idx] = s 

            # cluster features
            s, _ = to_dense_batch(s, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            cluster_mask = residue_mask 

            cluster_drop_mask = None
            if self.drop_residue != 0 and self.training:
                _, _, residue_drop_mask = dropout_node(residue_edge_index, self.drop_residue, residue_x.size(0), prot_batch,
                                                       self.training)  # drop residue for regularization
                residue_drop_mask, _ = to_dense_batch(residue_drop_mask.reshape(-1,1), prot_batch) # drop residue for regularization
                residue_drop_mask = residue_drop_mask.squeeze()
                cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()

            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, s, cluster_mask, cluster_drop_mask)

            # spectral_loss += sp_loss
            if not protein_only:
                o_loss = 0.5 * (o_loss + mol_o_loss)
                cl_loss = 0.5 * (cl_loss + mol_cl_loss)
            ortho_loss += o_loss / self.total_layer
            cluster_loss += cl_loss / self.total_layer
            
            cluster_x = self.prot_norms[idx](cluster_x)
            
            batch_size = s.size(0)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)
            cluster_x = cluster_x.reshape(batch_size*self.num_cluster[idx], -1)

            if not protein_only:
                cluster_x, clique_x, inter_edge_index, inter_attn = self.interaction(
                    idx, cluster_x, cluster_residue_batch, clique_x, clique_batch
                )
                interaction_edges.append(inter_edge_index)
                interaction_scores.append(inter_attn)
                row, col = atom2clique_index
                atom_x = atom_x + F.relu(
                    self.atom_lins[idx](scatter(clique_x[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean'))
                )
                atom_x = atom_x + self.c2a_mlps[idx](atom_x)

            residue_cluster_weight = s[residue_mask]
            cluster_index = (
                prot_batch[:, None] * self.num_cluster[idx]
                + torch.arange(self.num_cluster[idx], device=residue_x.device)[None, :]
            )
            residue_index = torch.arange(residue_x.size(0), device=residue_x.device)
            big2residue_index = torch.stack([
                cluster_index.reshape(-1),
                residue_index[:, None].expand_as(cluster_index).reshape(-1),
            ])
            cluster_feedback = self.big2residue(
                (cluster_x, residue_x), big2residue_index,
                residue_cluster_weight.reshape(-1),
                size=(cluster_x.size(0), residue_x.size(0)),
            )
            residue_x = residue_x + F.relu(self.residue_lins[idx](cluster_feedback))
            residue_x = residue_x + self.c2r_mlps[idx](residue_x)
            
            ## Graph Normalization
            if not protein_only:
                atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)

            if protein_only:
                residue_scores.append(residue_x.new_zeros((residue_x.size(0), self.heads)))
            else:
                pocket_score = scatter(
                    inter_attn, inter_edge_index[0], dim=0,
                    dim_size=cluster_x.size(0), reduce='mean'
                )
                pocket_score, _ = to_dense_batch(pocket_score, cluster_residue_batch)
                residue_scores.append((s @ pocket_score)[residue_mask])


        # Pool based on attn scores
        if protein_only:
            clique_scores = None
            atom_score = None
            mol_batch = None
            clique_batch = None
        else:
            row, col = atom2clique_index
            clique_scores = torch.cat(clique_scores, dim=-1)
            atom_scores = scatter(clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean')
            atom_score = softmax(self.atom_attn_lin(atom_scores), mol_batch)
            mol_pool_feat = self.mol_out(global_add_pool(atom_x * atom_score, mol_batch))

        residue_scores = torch.cat(residue_scores, dim=-1)
        residue_score = softmax(self.residue_attn_lin(residue_scores),prot_batch)
        prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)
        
        prot_pool_feat = self.prot_out(prot_pool_feat)
        if protein_only:
            mol_pool_feat = prot_pool_feat.new_zeros(prot_pool_feat.shape)

        if protein_only:
            ligand_global = protein_global.new_zeros(protein_global.shape)
            ligand_language_feature = ligand_global
        else:
            ligand_fingerprint = F.dropout(
                data.mol_global_fingerprint,
                p=self.ligand_fingerprint_dropout,
                training=self.training,
            )
            ligand_global = self.ligand_global(ligand_fingerprint)
            ligand_language_feature = self.ligand_language_global(
                data.mol_language_embedding
            )
            ligand_global = ligand_global + ligand_language_feature
        interaction_fingerprint = torch.cat([mol_pool_feat, prot_pool_feat], dim=-1)
        global_fingerprint = torch.cat([
            protein_global,
            ligand_global,
            protein_global * ligand_global,
            torch.abs(protein_global - ligand_global),
        ], dim=-1)
        mol_prot_feat = self.fusion_norm(
            self.fusion(torch.cat([interaction_fingerprint, global_fingerprint], dim=-1))
            + interaction_fingerprint
        )
        # Add the enzyme-aware residual after the pocket–motif fingerprint fusion. The
        # zero-initialized final layer preserves the pretrained base at start.
        catalytic_attention = {}
        if not protein_only:
            catalytic_fingerprint, _, _, catalytic_attention = self.catalytic_interaction(
                residue_x, prot_batch, atom_x, mol_batch, ligand_global, data.task_index
            )
            mol_prot_feat = mol_prot_feat + self.catalytic_fusion(torch.cat([
                catalytic_fingerprint, global_fingerprint
            ], dim=-1))

        mu, sigma = self.prediction_head(mol_prot_feat)

        attention_dict = {
            'residue_final_score':residue_score,
            'atom_final_score': atom_score,
            'clique_layer_scores':clique_scores,
            'residue_layer_scores':residue_scores,
            'drug_atom_index':mol_batch,
            'drug_clique_index':clique_batch,
            'protein_residue_index':prot_batch, 
            'mol_feature': mol_pool_feat,
            'prot_feature':prot_pool_feat,
            'protein_global_feature': protein_global,
            'ligand_global_feature': ligand_global,
            'ligand_language_feature': ligand_language_feature,
            'cluster_s': layer_s,
            'pocket_motif_edge_index': interaction_edges,
            'pocket_motif_attention': interaction_scores,
            'interaction_fingerprint': mol_prot_feat,
            **catalytic_attention,
        }
        return mu, sigma, ortho_loss, cluster_loss, attention_dict
def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
