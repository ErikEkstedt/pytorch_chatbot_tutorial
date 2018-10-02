import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size,
                 output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class EncoderDecoder(nn.Module):

    def __init__(self,
                 model_name = 'chatbot_model'
                 vocab_size=24305,
                 word_vector_dim=128,
                 hidden_size=512,
                 encoder_n_layers=2,
                 decoder_n_layers=2,
                 attn_model='dot',
                 dropout=0.1):
        super(EncoderDecoder, self).__init__()
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.word_vector_dim = word_vector_dim
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.attn_model = attn_model
        self.dropout = dropout


        self.embedding = nn.Embedding(vocab_size, word_vector_dim)
        self.encoder = EncoderRNN(hidden_size, self.embedding, encoder_n_layers, dropout)
        self.decoder = LuongAttnDecoderRNN(attn_model, self.embedding,
                                           hidden_size, voc.num_words,
                                           decoder_n_layers, dropout)

    def load_pretrained_embeddings(self, encoder_sd, decoder_sd)
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)

    def forward(self, inputs, lengths, targets, mask,
                teacher_forcing_ratio, batch_size, max_target_len):
        loss = 0
        print_losses = []
        n_totals = 0
        # TODO
        # batch_size = inputs.shape[0]  # If batch first
        # max_target_len = 

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Loop over tokens
        for t in range(max_target_len):
            # Forward batch of sequences through decoder one time step at a time
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden,
                                                     encoder_outputs)
            # Teacher forcing works best for each token according to
            # Scheduled Sampling for Sequence Prediction 
            # with Recurrent Neural Networks, (S. Bengio et al 2015)
            if random.random() < teacher_forcing_ratio:
                decoder_input = target_variable[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        return loss, sum(print_losses) / n_totals


def train_step(inputs, lengths, targets, mask, max_target_len,
               model, model_optimizer, batch_size, clip, max_length):

    # Zero gradients
    model_optimizer.zero_grad()

    # Set device options
    inputs = inputs.to(device)
    lengths = lengths.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    # Forward pass
    loss, print_losses = self(inputs, lengths, targets, mask,
                              teacher_forcing_ratio, batch_size, max_target_len)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    model_optimizer.step()

    return print_losses


if __name__ == "__main__":
    model_name = 'encoder_decoder_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 512
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    vocab_size = 24305
    word_vector_dim = 128

    # TODO
    # This has not been run yet

    print('Building encoder and decoder ...')
    model = EncoderDecoder(model_name,
                           vocab_size,
                           word_vector_dim,
                           hidden_size,
                           encoder_n_layers,
                           decoder_n_layers,
                           attn_model,
                           dropout)
    model.to(device)

    train_step(inputs, lengths, targets, mask, max_target_len,
               model, model_optimizer, batch_size, clip, max_length)
