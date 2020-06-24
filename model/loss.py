import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_mask_from_lengths
import editdistance as ed
import numpy as np
import pdb

# i  = 0

def LetterErrorRate(pred_y,true_y):

    ed_accumalate = []
    for p,t in zip(pred_y,true_y):
        compressed_t = []
        for t_w in t:
            if t_w == 41:
                break
            compressed_t.append(t_w)
        
        compressed_p = []
        for p_w in p:
            if p_w == 41:
                break
            compressed_p.append(p_w)

        ed_accumalate.append(ed.eval(compressed_p,compressed_t)/len(compressed_t))
        acc = np.array(ed_accumalate).mean()
    return acc

class ParrotLoss(nn.Module):
    def __init__(self, hparams):
        super(ParrotLoss, self).__init__()
        self.hidden_dim = hparams.encoder_embedding_dim
        self.ce_loss = hparams.ce_loss

        self.L1Loss = nn.L1Loss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.n_frames_per_step = hparams.n_frames_per_step_decoder
        self.eos = hparams.n_symbols
        self.predict_spectrogram = hparams.predict_spectrogram

        self.contr_w = hparams.contrastive_loss_w
        self.consi_w = hparams.consistent_loss_w
        self.spenc_w = hparams.speaker_encoder_loss_w
        self.texcl_w = hparams.text_classifier_loss_w
        self.spadv_w = hparams.speaker_adversial_loss_w
        self.spcla_w = hparams.speaker_classifier_loss_w

    def parse_targets(self, targets, text_lengths):
        '''
        text_target [batch_size, text_len]
        mel_target [batch_size, mel_bins, T]
        spc_target [batch_size, spc_bins, T]
        speaker_target [batch_size]
        stop_target [batch_size, T]
        '''
        text_target, mel_target, speaker_target, stop_target = targets

        B = stop_target.size(0)
        stop_target = stop_target.reshape(B, -1, self.n_frames_per_step)
        stop_target = stop_target[:, :, 0]

        padded = torch.tensor(text_target.data.new(B,1).zero_())
        text_target = torch.cat((text_target, padded), dim=-1)

        return text_target, mel_target, speaker_target, stop_target

    def forward(self, model_outputs, targets, input_text, eps=1e-5):

        '''
        predicted_mel [batch_size, mel_bins, T]
        predicted_stop [batch_size, T/r]
        alignment 
            when input_text==True [batch_size, T/r, max_text_len] 
            when input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, max_text_len, hidden_dim]
        text_logit_from_mel_hidden [B, max_text_len+1, n_symbols+1]
        speaker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, max_text_len, n_speakers]
        text_lengths [B,]
        mel_lengths [B,]
        '''

        mel_hidden, text_logit_from_mel_hidden, audio_seq2seq_alignments, text_lengths = model_outputs

        text_target, mel_target, speaker_target, stop_target  = self.parse_targets(targets, text_lengths)

        text_mask = get_mask_from_lengths(text_lengths).float()
        text_mask_plus_one = get_mask_from_lengths(text_lengths + 1).float()


        n_symbols_plus_one = text_logit_from_mel_hidden.size(2)


        # text classification loss #
        text_logit_flatten = text_logit_from_mel_hidden.reshape(-1, n_symbols_plus_one)
        text_target_flatten = text_target.reshape(-1)
        _, predicted_text =  torch.max(text_logit_flatten, dim=1)
        _, predicted_text_no_reshape = torch.max(text_logit_from_mel_hidden, dim =2)
        text_classification_acc = LetterErrorRate(predicted_text_no_reshape.cpu().numpy(), text_target.cpu().data.numpy())
        loss = self.CrossEntropyLoss(text_logit_flatten, text_target_flatten)
        text_classification_loss = torch.sum(loss * text_mask_plus_one.reshape(-1)) / torch.sum(text_mask_plus_one)
        loss_list = [text_classification_loss]


        acc_list = [text_classification_acc]
        
        combined_loss1 = text_classification_loss

        return loss_list, acc_list, combined_loss1
