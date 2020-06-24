import torch
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder
from .layers import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet


class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        #print hparams
        # plus <sos> 
        self.embedding = nn.Embedding(
            hparams.n_symbols + 1, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std

        self.sos = hparams.n_symbols
        self.audio_seq2seq = AudioSeq2seq(hparams)

    def grouped_parameters(self,):

        params_group1 = [p for p in self.audio_seq2seq.parameters()]
        return params_group1

    def parse_batch(self, batch):
        text_input_padded, mel_padded, speaker_id, \
                    text_lengths, mel_lengths, stop_token_padded = batch
        
        text_input_padded = to_gpu(text_input_padded).long()
        mel_padded = to_gpu(mel_padded).float()
        speaker_id = to_gpu(speaker_id).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_lengths = to_gpu(mel_lengths).long()
        stop_token_padded = to_gpu(stop_token_padded).float()

        return ((text_input_padded, mel_padded, text_lengths, mel_lengths),
                (text_input_padded, mel_padded,  speaker_id, stop_token_padded))


    def forward(self, inputs, input_text):
        '''
        text_input_padded [batch_size, max_text_len]
        mel_padded [batch_size, mel_bins, max_mel_len]
        text_lengths [batch_size]
        mel_lengths [batch_size]

        #
        predicted_mel [batch_size, mel_bins, T]
        predicted_stop [batch_size, T/r]
        alignment input_text==True [batch_size, T/r, max_text_len] or input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, T/r, hidden_dim]
        spearker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, T/r, n_speakers]
        text_logit_from_mel_hidden [B, T/r, n_symbols]

        # '''
        # import pdb
        # pdb.set_trace()
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments = self.audio_seq2seq(
                audio_input, mel_lengths, text_input_embedded, start_embedding) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        outputs = [audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments,text_lengths ]

        return outputs

    
    def inference(self, inputs, input_text, mel_reference, beam_width):
        '''
        decode the audio sequence from input
        inputs x
        input_text True or False
        mel_reference [1, mel_bins, T]
        '''
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs

        B = text_input_padded.size(0) # B should be 1
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding) # [1, embedding_dim]

        audio_input = mel_padded
        
        logits = self.audio_seq2seq.inference_beam(
                audio_input, start_embedding, self.embedding, beam_width=beam_width) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        # -> [B, n_speakers], [B, speaker_embedding_dim] 

        if input_text:
            hidden = self.merge_net.inference(text_hidden)
        else:
            hidden = self.merge_net.inference(audio_seq2seq_hidden)

        L = hidden.size(1)
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)
          
        predicted_mel, predicted_stop, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_stop, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            speaker_id)


