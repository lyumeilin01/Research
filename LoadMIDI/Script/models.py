import torch
import torch.nn as nn

class MidiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(MidiLSTMModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_TOKEN_ID)

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, batch_first=True)

        self.output_layer = nn.Linear(in_features=lstm_units, out_features=vocab_size)

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape (batch_size, sequence_length)
        hidden: initial hidden state for the lstm, shape (1, batch_size, lstm_units)
        """
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        lstm_output, hidden = self.lstm1(embedded, hidden)  # lstm_output: (batch_size, sequence_length, lstm_units)
        lstm_output, hidden = self.lstm2(lstm_output, hidden)  # lstm_output: (batch_size, sequence_length, lstm_units)

        logits = self.output_layer(lstm_output)  # (batch_size, sequence_length, vocab_size)

        return logits, hidden


class MidiGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_units, tokenizer, device):
        super(MidiGRUModel, self).__init__()
        
        self.tokenizer = tokenizer
        self.PAD_TOKEN_ID = tokenizer.vocab["PAD_None"]
        self.BAR_ID = tokenizer.vocab["Bar_None"]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=self.PAD_TOKEN_ID)

        self.gru1 = nn.GRU(input_size=embedding_dim, hidden_size=gru_units, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_units, hidden_size=gru_units, batch_first=True)

        self.output_layer = nn.Linear(in_features=gru_units, out_features=vocab_size)
        self.cur_sample = None
        self.cur_logits = None
        self.cur_hidden_state = None
        self.device = device
        self.generated_tokens = None

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape (batch_size, sequence_length)
        hidden: initial hidden state for the GRU, shape (1, batch_size, gru_units)
        """
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        gru_output, hidden = self.gru1(embedded, hidden)  # gru_output: (batch_size, sequence_length, gru_units)
        gru_output, hidden = self.gru2(gru_output, hidden)  # gru_output: (batch_size, sequence_length, gru_units)

        logits = self.output_layer(gru_output)  # (batch_size, sequence_length, vocab_size)

        return logits, hidden
    # def sample_token(self, token_sequence, hidden_state=None):

    #     self.eval()
    #     with torch.no_grad():
    #         # forward pass through model
    #         #logits, updated_hidden_state = model(token_sequence, hidden_state)
    #         logits, hidden_state = self(token_sequence, hidden_state)

    #         # logits shape: (batch_size, seq_len, vocab_size)
    #         final_step_logits = logits[:, -1, :]  # last timestep logits

    #         # sample from logits (multinomial sampling)
    #         probs = torch.nn.functional.softmax(final_step_logits, dim=-1)
    #         samples = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
    #         self.cur_sample =samples.detach()
    #         self.cur_logits = final_step_logits
    #         self.cur_hidden_state = hidden_state

    #     return samples, hidden_state, final_step_logits
    def sample_token(self, token_sequence, hidden_state=None, temperature = 1.0):

        self.eval()
        with torch.no_grad():
 
            logits, hidden_state = self(token_sequence, hidden_state)

            final_step_logits = logits[:, -1, :]

            scaled_logits = final_step_logits / temperature

            # 4) Softmax → probabilities → multinomial sampling
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            samples = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)

            # 5) Cache for external inspection:
            self.cur_sample = samples.detach()
            self.cur_logits = scaled_logits
            self.cur_hidden_state = hidden_state
        #can return temperature-scaled or original logits
        return samples, hidden_state, scaled_logits

    #need to specify device 
    def generate_midi(self, start_sequence, max_bars, temp):
        self.eval()
        generated_sequence = list(start_sequence)
        initial_state = None
        num_bars = generated_sequence.count(self.BAR_ID)

        while num_bars<max_bars+1:
            
            token_tensor = torch.LongTensor(generated_sequence).unsqueeze(0).to(self.device)
            sample, initial_state, _ = self.sample_token(token_tensor, initial_state, temp)
            #print(sample)
            if sample[0][0].item()==self.BAR_ID:
                num_bars+=1
                print(f"generating bar {num_bars-1}/{max_bars}")
            # Move sample to CPU before converting to numpy
            generated_sequence.append(sample.cpu().numpy()[0][0])
        self.generated_tokens = generated_sequence
        return generated_sequence
    def get_next_token(self, generated, temp = 1.0):
        #print(generated)
        torch_token = torch.LongTensor(generated).unsqueeze(0).to(self.device)

        token, hidden_state, logits = self.sample_token(torch_token.to(self.device), self.cur_hidden_state, temp)
        generated.append(token.squeeze().item())
        self.generated_tokens = generated
        return token, generated
    
    def reset_memory(self):
        self.cur_hidden_state = None
        self.cur_logits = None
        self.cur_sample = None
