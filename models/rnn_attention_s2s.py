from models.seq2seq import Seq2Seq
from modules import EncoderRNN, AttnDecoderRNN

class RnnAttentionS2S(Seq2Seq):

    def __init__(self, input_lang, output_lang, max_length, hidden_size=256, device='cpu'):
        encoder = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length, dropout_p=0.1, device=device).to(device)
        super().__init__(encoder, decoder, input_lang, output_lang, device)

