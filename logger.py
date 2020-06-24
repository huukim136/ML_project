import os
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_alignment
from plotting_utils import plot_gate_outputs_to_numpy


class ParrotLogger(SummaryWriter):
    def __init__(self, logdir, ali_path='ali'):
        super(ParrotLogger, self).__init__(logdir)
        ali_path = os.path.join(logdir, ali_path)
        if not os.path.exists(ali_path):
            os.makedirs(ali_path)
        self.ali_path = ali_path

    def log_training(self, reduced_loss, reduced_losses, reduced_acces, grad_norm, learning_rate, duration,
                     iteration):
        
        self.add_scalar("training.loss", reduced_loss, iteration)

        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)  
        self.add_scalar('training.acc.texcl', reduced_acces[0], iteration)
  
    def log_validation(self, reduced_loss, reduced_losses, reduced_acces, model, y, y_pred, iteration, task):

        self.add_scalar('validation.loss.%s'%task, reduced_loss, iteration)
        self.add_scalar('validation.acc.%s.texcl'%task, reduced_acces[0], iteration)
        mel_hidden, text_logit_from_mel_hidden, audio_seq2seq_alignments, text_lengths = y_pred
        # plot alignment, mel target and predicted, stop target and predicted
        idx = random.randint(0, audio_seq2seq_alignments.size(0) - 1)
        audio_seq2seq_alignments = audio_seq2seq_alignments.data.cpu().numpy()

        self.add_image(
            "%s.audio_seq2seq_alignment"%task,
            plot_alignment_to_numpy(audio_seq2seq_alignments[idx].T),
            iteration, dataformats='HWC')
