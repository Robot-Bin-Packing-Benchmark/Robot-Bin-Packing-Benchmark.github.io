import torch.nn as nn
from packer.pct_model.tools import init
from numpy import sqrt
from packer.pct_model.attention_model import AttentionModel

class DRL_GAT(nn.Module):
    def __init__(self, embedding_size=64, hidden_size=128, gat_layer_num=1, internal_node_holder=80, internal_node_length=6, leaf_node_holder=50):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(embedding_size,
                                    hidden_size,
                                    n_encode_layers = gat_layer_num,
                                    n_heads = 1,
                                    internal_node_holder = internal_node_holder,
                                    internal_node_length = internal_node_length,
                                    leaf_node_holder = leaf_node_holder,
                                    )
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), sqrt(2))
        self.critic = init_(nn.Linear(embedding_size, 1))

    def forward(self, items, deterministic = False, normFactor = 1, evaluate = False):
        o, p, dist_entropy, hidden, _= self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate)
        values = self.critic(hidden)
        return o, p, dist_entropy,values

    def evaluate_actions(self, items, actions, normFactor = 1):
        _, p, dist_entropy, hidden, dist = self.actor(items, evaluate_action = True, normFactor = normFactor)
        action_log_probs = dist.log_probs(actions)
        values =  self.critic(hidden)
        return values, action_log_probs, dist_entropy.mean()
