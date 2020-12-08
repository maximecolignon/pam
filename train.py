import click
import torch
from models.rnn_attention_s2s import RnnAttentionS2S
from utils import prepareData

@click.command(help="train env_name exp_dir data_path")
@click.option("-d","--data-path", default="data", type=str)
@click.option("-a", "--architecture", default="rnn_attention_s2s", type=str)
@click.option("-nit", "--n-iters", default=10000, type=int)
@click.option("-ml", "--max-length", default=10, type=int)
@click.option("-hs", "--hidden-size", default=256, type=int)
@click.option("-lr", "--learning_rate", default = 0.01)
@click.option("-tfr", "--teacher-forcing-ratio", default=0.5)


def main(
        data_path,
        architecture,
        n_iters,
        max_length,
        hidden_size,
        learning_rate,
        teacher_forcing_ratio,
         ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', path=data_path, max_length= max_length, prefixes=None)

    if architecture=="rnn_attention_s2s":
        model = RnnAttentionS2S(input_lang, output_lang, max_length=max_length, hidden_size=hidden_size, device=device)
    else:
        raise Exception('Unknown architecture')

    model.train(pairs, n_iters = n_iters, learning_rate=learning_rate, max_length=max_length, teacher_forcing_ratio=teacher_forcing_ratio)



if __name__ == '__main__':
    main()


