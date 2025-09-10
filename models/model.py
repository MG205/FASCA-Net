import torch
from torch import nn

from models.TA_FARModuls import FGFA_TA
from models.TV_FARModuls import FGFA_TV
from models.cross_attn_encoder import  TriModalFusion
from models.last_model import FCRegressor
from models.text_model import TextSentimentModel
from models.audio_model import AudioSentimentModel
from models.vision_model import VisionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Fine-grained Alignment via Self- and Cross-Attention Network


class FASCA_Net(nn.Module):
    def __init__(self, config):        
        super().__init__()

        self.T_output_layers = TextSentimentModel(input_dim=768)
        self.A_output_layers = AudioSentimentModel(input_dim=5, hidden_dim=64, n_heads=8)

        self.V_output_layers = VisionModel(input_dim=20)
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=5)
        self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=20)
        

        # CME layers
        self.CME_layers = nn.ModuleList(
            [TriModalFusion() for _ in range(config.num_hidden_layers)]
        )

        self.MultiModalAligner_TV = FGFA_TV(dim=51).to(device)
        self.MultiModalFAR_TA = FGFA_TA(dim1=51, dim2=51)
        self.fused_output_layers = FCRegressor()

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask, vision_input, vision_mask):

        # text output layer
        T_output = self.T_output_layers(text_inputs)
        
        # audio output layer
        A_output = self.A_output_layers(audio_inputs)

        # vision output layer
        V_output = self.V_output_layers(vision_input)
        
        # CME layers
        text_inputs, text_attn_mask = self.prepend_cls(text_inputs, text_mask, 'text')     # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(audio_inputs, audio_mask, 'audio')  # add cls token
        vision_input, vision_attn_mask = self.prepend_cls(vision_input, vision_mask, 'vision')  # add cls token

        # pass through CME layers
        for layer_module in self.CME_layers:
            text_inputs, audio_inputs,  vision_input = layer_module(text_inputs, text_attn_mask, audio_inputs, audio_attn_mask, vision_input, vision_attn_mask)
        text_inputs = text_inputs.unsqueeze(2)
        vision_input = vision_input.unsqueeze(2)
        audio_inputs = audio_inputs.unsqueeze(2)
        output_TV = self.MultiModalAligner_TV(text_inputs, vision_input).squeeze(2)
        output_TA = self.MultiModalFAR_TA(text_inputs, audio_inputs).squeeze(2)

        fused_hidden_states = torch.cat((output_TV[:, 0, :], output_TA[:, 0, :]), dim=1)
        fused_output = self.fused_output_layers(fused_hidden_states)
        
        return {
                'T': T_output, 
                'A': A_output,
                'V': V_output,
                'M': fused_output
        }

