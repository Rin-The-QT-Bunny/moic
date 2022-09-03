import torch
import torch.nn as nn
import torch.nn.functional as F
from aluneth.rinlearn.cv.utils import *

from typing import Tuple
from utils import Tensor
from utils import assert_shape
from utils import build_grid
from utils import conv_transpose_out_shape

class SlotAttention(nn.Module):
    def __init__(self,in_features,num_iters,num_slots,slot_size,mlp_size,epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iters
        self.nu_slots = num_slots
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        
        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_mem_slots = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slots? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slots_size,self.slot_size,bias = False)
        self.project_k1 = nn.Linear(self.slots_size,self.slot_size,bias = False)
        self.project_v1 = nn.Linear(self.slot_size,self.slot_size,bias = False)
        # Linear maps for the memory attention module.
        self.project_k2 = nn.Linear(self.slot_size,self.slot_size,bias = False)
        self.project_v2 = nn.Linear(self.slot_size,self.slot_size,bias = False)

        self.mlp_i = nn.Sequential(
            nn.Linear(self.slot_size*2,self.mlp_size),
            nn.ReLU(),
            nn.Linear(self.mlp_size,self.slot_size),
        )

        self.gru = nn.GRUCell(self.slot_size,self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size,self.mlp_size),
            nn.ReLU(),
            nn.Linear(self.mlp_size,self.slot_size),
        )

        self.mlp_m = nn.Sequential(
            nn.Linear(self.slot_size,self.mlp_size),
            nn.ReLU(),
            nn.Linear(self.mlp_size,self.slot_size),
        )

        self.register_buffer(
            "slot_mu",
            nn.init.xavier_uniform_(torch.zeros([1,1,self.slot_size]),
            gain=nn.init.calculate_gain("linear"))
        )
        self.register_buffer(
            "slot_log_sigma",
            nn.init.xavier_uniform_(torch.zeros([1,1,self.slot_size]),
            gain = nn.init.calculate_gain("linear"))
        )
        self.register_buffer(
            "mem_slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "mem_slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.project_qm = nn.Linear(self.slot_size,self.slot_size,bias=False)
        self.project_km = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_vm = nn.Linear(self.slot_size, self.slot_size, bias=False)

    def forward(self,inputs:Tensor,memory_slots,instant_slots):
        #`inputs` has shape [batch_size,num_inputs,input_size]
        batch_size,num_inputs,inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs) # apply layer norm to the inputs slots

        batch_size,num_mem_slots,mem_slot_size = memory_slots.shape
        memory_slots = self.norm_mem_slots(memory_slots) # apply layer norm to the memory slots

        k1 = self.project_k1(inputs) # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k1.size(), (batch_size, num_inputs, self.slot_size))
        v1 = self.project_v1(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v1.size(), (batch_size, num_inputs, self.slot_size))

        k2 = self.project_k2(memory_slots)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k2.size(), (batch_size, num_mem_slots, self.slot_size))
        v2 = self.project_v2(memory_slots)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v2.size(), (batch_size, num_mem_slots, self.slot_size))

        # Multiple rounds of attention
        for _ in range(self.num_iterations):
            slots_prev = instant_slots
            instant_slots = self.norm_slots(instant_slots)

            # Attention
            q = self.project_q(instant_slots)
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slots**-0.5

            ##### Attention on Input features #####
            attn_logits1 = attn_norm_factor * torch.matmul(k1, q.transpose(2, 1))
            attn1 = F.softmax(attn_logits1, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn1.size(), (batch_size, num_inputs, self.num_slots))
             # Weighted mean.
            attn1 = attn1 + self.epsilon
            attn1 = attn1 / torch.sum(attn1, dim=1, keepdim=True)
            updates1 = torch.matmul(attn1.transpose(1, 2), v1)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates1.size(), (batch_size, self.num_slots, self.slot_size))
            #################################################################################

            ############### Attention on Memory Slots ########################################
            attn_logits2 = attn_norm_factor * torch.matmul(k2, q.transpose(2, 1))
            attn2 = F.softmax(attn_logits2, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn1.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn2 = attn2 + self.epsilon
            attn2 = attn2 / torch.sum(attn2, dim=1, keepdim=True)
            updates2 = torch.matmul(attn2.transpose(1, 2), v2)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates2.size(), (batch_size, self.num_slots, self.slot_size))
            #################################################################################

            updates = self.mlp_i(torch.cat((updates1, updates2), dim=-1))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            instant_slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            instant_slots = instant_slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))
            instant_slots = instant_slots + self.mlp(self.norm_mlp(instant_slots))
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))

        ############### Updating on Memory Slots with Attention ##########################
        km = self.project_km(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(km.size(), (batch_size, self.num_slots, self.slot_size))
        vm = self.project_vm(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(vm.size(), (batch_size, self.num_slots, self.slot_size))
        qm = self.project_qm(memory_slots)  # Shape: [batch_size, num_mem_slots, slot_size].
        assert_shape(qm.size(), (batch_size, num_mem_slots, self.slot_size))

        attn_logitsm = attn_norm_factor * torch.matmul(qm, km.transpose(2, 1))
        attnm = F.softmax(attn_logitsm, dim=-1)
        # `attn` has shape: [batch_size, num_inputs, num_slots].
        assert_shape(attnm.size(), (batch_size, num_mem_slots, self.num_slots))

        # Weighted mean.
        attnm = attnm + self.epsilon
        attnm = attnm / torch.sum(attnm, dim=1, keepdim=True)
        updatesm = torch.matmul(attnm, vm)
        # `updates` has shape: [batch_size, num_slots, slot_size].
        assert_shape(updatesm.size(), (batch_size, num_mem_slots, self.slot_size))

        memory_slots = memory_slots + self.mlp_m(updatesm)
        #################################################################################

        return memory_slots, instant_slots