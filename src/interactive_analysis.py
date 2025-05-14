# %%
import pretty_midi
import os
import seaborn as sns
import matplotlib.pyplot as plt # For plotting
import numpy as np
import pickle
from IPython.display import Audio


# %%
directory = "../data/melody"
midi_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mid', '.midi'))]

# %%
from dataclasses import dataclass

@dataclass
class MidiMessage:
    event: bool  # True for note_on, False for note_off
    note: int
    velocity: int
    delta_time: float  # Relative time from previous message

    @staticmethod
    def from_pretty_midi(note):
        # Create note_on message with delta time from previous message
        on_msg = MidiMessage(
            event=True,
            note=note.pitch,
            velocity=note.velocity,
            delta_time=note.start  # This will be adjusted in the track parsing
        )
        # Create note_off message with delta time from note_on
        off_msg = MidiMessage(
            event=False,
            note=note.pitch,
            velocity=note.velocity,
            delta_time=note.end - note.start  # Duration of the note
        )
        return [on_msg, off_msg]

    def to_pretty_midi_note(self):
        if self.event:  # note_on
            return pretty_midi.Note(
                velocity=self.velocity,
                pitch=self.note,
                start=self.delta_time,  # This will be adjusted in MidiTrack
                end=self.delta_time + 0.1  # Default duration, will be adjusted by note_off
            )
        return None

# %%
from typing import List

class MidiTrack:
    def __init__(self, messages: List[MidiMessage] = None):
        self.messages = messages if messages is not None else []

    def add_message(self, message: MidiMessage):
        self.messages.append(message)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]

    def __iter__(self):
        return iter(self.messages)

    def to_file(self, filename):
        """Serialize the MidiTrack to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.messages, f)

    @classmethod
    def from_file(cls, filename):
        """Deserialize a MidiTrack from a file using pickle."""
        with open(filename, 'rb') as f:
            messages = pickle.load(f)
        return cls(messages)

    def to_midi_file(self, filename):
        """Convert MidiTrack to a MIDI file using pretty_midi."""
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Track absolute time
        current_time = 0.0
        
        # Group messages by note to handle note_on/note_off pairs
        note_dict = {}
        for msg in self.messages:
            current_time += msg.delta_time
            
            if msg.event:  # note_on
                note_dict[msg.note] = (current_time, msg.velocity)
            else:  # note_off
                if msg.note in note_dict:
                    start_time, velocity = note_dict[msg.note]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=msg.note,
                        start=start_time,
                        end=current_time
                    )
                    instrument.notes.append(note)
                    del note_dict[msg.note]
        
        pm.instruments.append(instrument)
        pm.write(filename)

# %%
# Use the first MIDI file from midi_files
midi_path = os.path.join(directory, midi_files[0])

# Load MIDI file with pretty_midi
midi_data = pretty_midi.PrettyMIDI(midi_path)

# Synthesize to audio using pretty_midi's built-in synth
audio = midi_data.synthesize(fs=22050)

# Play audio inline
Audio(audio, rate=22050)

# %%
# Parse the first MIDI file into MidiTrack format
midi_path = os.path.join(directory, midi_files[0])
midi_data = pretty_midi.PrettyMIDI(midi_path)
messages = []

# Sort all notes by start time to ensure proper ordering
all_notes = []
for instrument in midi_data.instruments:
    all_notes.extend(instrument.notes)
all_notes.sort(key=lambda x: x.start)

# Convert pretty_midi notes to MidiMessages with proper timing
current_time = 0.0
for note in all_notes:
    # Calculate delta time for note_on
    on_delta = note.start - current_time
    current_time = note.start
    
    # Create note_on message
    messages.append(MidiMessage(
        event=True,
        note=note.pitch,
        velocity=note.velocity,
        delta_time=on_delta
    ))
    
    # Create note_off message
    messages.append(MidiMessage(
        event=False,
        note=note.pitch,
        velocity=note.velocity,
        delta_time=note.end - note.start
    ))
    current_time = note.end

# Create MidiTrack and save to MIDI file
track_obj = MidiTrack(messages)
track_obj.to_midi_file('test_miditrack.mid')

# Play the saved MIDI file
midi_data = pretty_midi.PrettyMIDI('test_miditrack.mid')
audio = midi_data.synthesize(fs=22050)
Audio(audio, rate=22050)

# %%
my_midi_data = [
    MidiMessage(event=True, note=60, velocity=100, delta_time=0),     # C4 note on
    MidiMessage(event=True, note=64, velocity=100, delta_time=0),     # E4 note on
    MidiMessage(event=False, note=64, velocity=0, delta_time=1),      # E4 note off
    MidiMessage(event=True, note=67, velocity=100, delta_time=0),     # G4 note on
    MidiMessage(event=False, note=67, velocity=0, delta_time=1),      # G4 note off
    MidiMessage(event=True, note=72, velocity=100, delta_time=0),     # C5 note on
    MidiMessage(event=False, note=72, velocity=0, delta_time=1),      # C5 note off
    MidiMessage(event=False, note=60, velocity=0, delta_time=10),      # C4 note off
]
# Create MidiTrack and save to MIDI file
track_obj = MidiTrack(my_midi_data)
track_obj.to_midi_file('my_midi.mid')

# Play the saved MIDI file
midi_data = pretty_midi.PrettyMIDI('my_midi.mid')
audio = midi_data.synthesize(fs=22050)
Audio(audio, rate=22050)

# %%
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# %%
class MidiTransformer(nn.Module):
    def __init__(self, 
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_len=128):
        super().__init__()
        
        # Input embedding layer (note, velocity, delta_time, is_note_on)
        self.input_embedding = nn.Linear(4, d_model)  # Added is_note_on flag
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.note_head = nn.Linear(d_model, 128)  # 128 possible MIDI notes
        self.velocity_head = nn.Linear(d_model, 128)  # 128 possible velocities
        self.time_head = nn.Linear(d_model, 1)  # Continuous time prediction
        self.event_type_head = nn.Linear(d_model, 2)  # Binary classification: note_on or note_off
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, 4) - (note, velocity, delta_time, is_note_on)
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        if mask is not None:
            x = self.transformer_encoder(x, mask)
        else:
            x = self.transformer_encoder(x)
            
        # Predict next note, velocity, time, and event type
        note_logits = self.note_head(x)
        velocity_logits = self.velocity_head(x)
        time_pred = self.time_head(x)
        event_type_logits = self.event_type_head(x)
        
        return note_logits, velocity_logits, time_pred, event_type_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# %%
class MidiDataset(Dataset):
    def __init__(self, midi_tracks, seq_length=32):
        self.sequences = []
        self.seq_length = seq_length
        
        # Convert MidiTracks to sequences of (note, velocity, delta_time, is_note_on)
        for track in midi_tracks:
            sequence = []
            for msg in track:
                sequence.append([
                    msg.note,
                    msg.velocity,
                    msg.delta_time,
                    1.0 if msg.event else 0.0  # 1.0 for note_on, 0.0 for note_off
                ])
            
            # Create overlapping sequences
            for i in range(0, len(sequence) - seq_length):
                self.sequences.append(sequence[i:i + seq_length + 1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input sequence (all but last element)
        x = torch.tensor(sequence[:-1], dtype=torch.float32)
        # Target (last element)
        y_note = torch.tensor(sequence[-1][0], dtype=torch.long)
        y_velocity = torch.tensor(sequence[-1][1], dtype=torch.long)
        y_time = torch.tensor(sequence[-1][2], dtype=torch.float32)
        y_event = torch.tensor(sequence[-1][3], dtype=torch.long)
        
        return x, (y_note, y_velocity, y_time, y_event)

# %%
def train_model(model, train_loader, num_epochs=5, learning_rate=0.0003):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Loss functions with weights
    note_criterion = nn.CrossEntropyLoss()
    velocity_criterion = nn.CrossEntropyLoss()
    time_criterion = nn.MSELoss()
    event_criterion = nn.CrossEntropyLoss()
    
    # Loss weights
    note_weight = 1.0
    velocity_weight = 0.5
    time_weight = 0.3
    event_weight = 0.8
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_note_loss = 0
        epoch_velocity_loss = 0
        epoch_time_loss = 0
        epoch_event_loss = 0
        
        for batch_idx, (x, (y_note, y_velocity, y_time, y_event)) in enumerate(train_loader):
            x = x.to(device)
            y_note = y_note.to(device)
            y_velocity = y_velocity.to(device)
            y_time = y_time.to(device)
            y_event = y_event.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            note_logits, velocity_logits, time_pred, event_logits = model(x)
            
            # Calculate losses
            note_loss = note_criterion(note_logits[:, -1], y_note)
            velocity_loss = velocity_criterion(velocity_logits[:, -1], y_velocity)
            time_loss = time_criterion(time_pred[:, -1].squeeze(), y_time)
            event_loss = event_criterion(event_logits[:, -1], y_event)
            
            # Weighted total loss
            loss = (note_weight * note_loss + 
                   velocity_weight * velocity_loss + 
                   time_weight * time_loss +
                   event_weight * event_loss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            epoch_note_loss += note_loss.item()
            epoch_velocity_loss += velocity_loss.item()
            epoch_time_loss += time_loss.item()
            epoch_event_loss += event_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}')
                print(f'Note Loss: {note_loss.item():.4f}, Velocity Loss: {velocity_loss.item():.4f}')
                print(f'Time Loss: {time_loss.item():.4f}, Event Loss: {event_loss.item():.4f}')
                print(f'Total Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_note_loss = epoch_note_loss / len(train_loader)
        avg_velocity_loss = epoch_velocity_loss / len(train_loader)
        avg_time_loss = epoch_time_loss / len(train_loader)
        avg_event_loss = epoch_event_loss / len(train_loader)
        
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Average Note Loss: {avg_note_loss:.4f}')
        print(f'Average Velocity Loss: {avg_velocity_loss:.4f}')
        print(f'Average Time Loss: {avg_time_loss:.4f}')
        print(f'Average Event Loss: {avg_event_loss:.4f}')
        print(f'Average Total Loss: {avg_loss:.4f}')
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_midi_transformer.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

# %%
# Example usage:
# 1. Create model
model = MidiTransformer()

# 2. Create dataset and dataloader
# Assuming you have a list of MidiTracks called 'tracks'
# dataset = MidiDataset([track_obj])  # Use your existing track_obj
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Train the model
# train_model(model, train_loader)

# %%
# Load first 100 MIDI files from the directory
all_tracks = []
max_tracks = 100  # Limit to first 100 tracks
print("Loading MIDI files...")
for midi_file in midi_files[:max_tracks]:
    midi_path = os.path.join(directory, midi_file)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    messages = []
    
    # Sort all notes by start time to ensure proper ordering
    all_notes = []
    for instrument in midi_data.instruments:
        all_notes.extend(instrument.notes)
    all_notes.sort(key=lambda x: x.start)
    
    # Convert pretty_midi notes to MidiMessages with proper timing
    current_time = 0.0
    for note in all_notes:
        # Calculate delta time for note_on
        on_delta = note.start - current_time
        current_time = note.start
        
        # Create note_on message
        messages.append(MidiMessage(
            event=True,
            note=note.pitch,
            velocity=note.velocity,
            delta_time=on_delta
        ))
        
        # Create note_off message
        messages.append(MidiMessage(
            event=False,
            note=note.pitch,
            velocity=note.velocity,
            delta_time=note.end - note.start
        ))
        current_time = note.end
    
    track_obj = MidiTrack(messages)
    all_tracks.append(track_obj)

print(f"Loaded {len(all_tracks)} MIDI tracks")

# %%
# Create dataset and dataloader with moderate batch size and sequence length
dataset = MidiDataset(all_tracks, seq_length=32)  # Increased sequence length
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Increased batch size

print(f"Created dataset with {len(dataset)} sequences")
print(f"Each sequence has length {dataset.seq_length}")

# %%
# Create and train the model with increased size
model = MidiTransformer(
    d_model=256,      # Increased from 64
    nhead=8,         # Increased from 4
    num_layers=4,    # Increased from 2
    dim_feedforward=1024,  # Increased from 256
    max_seq_len=128   # Increased from 64
)

# Train for a few epochs
train_model(model, train_loader, num_epochs=5)  # Increased from 3 epochs

# %%
# Save the trained model
torch.save(model.state_dict(), 'midi_transformer.pth')
print("Model saved successfully!")

# %%
def generate_midi(model, seed_sequence, max_time=10.0, temperature=0.8):
    """
    Generate MIDI sequence from the model.
    seed_sequence: initial sequence to start generation from
    max_time: maximum time in seconds to generate
    temperature: controls randomness (higher = more random)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Convert seed sequence to tensor and ensure correct shape
    current_seq = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 4]
    current_seq = current_seq.to(device)
    generated_messages = []
    current_time = 0.0
    
    with torch.no_grad():
        while current_time < max_time:
            # Get model predictions
            note_logits, velocity_logits, time_pred, event_logits = model(current_seq)
            
            # Sample next note
            note_probs = F.softmax(note_logits[:, -1] / temperature, dim=-1)
            next_note = torch.multinomial(note_probs, 1).item()
            
            # Sample next velocity
            velocity_probs = F.softmax(velocity_logits[:, -1] / temperature, dim=-1)
            next_velocity = torch.multinomial(velocity_probs, 1).item()
            
            # Sample next event type
            event_probs = F.softmax(event_logits[:, -1] / temperature, dim=-1)
            is_note_on = torch.multinomial(event_probs, 1).item()
            
            # Get next time and ensure it's reasonable
            next_time = max(0.05, min(0.5, time_pred[:, -1].item()))
            
            # Add to generated sequence
            generated_messages.append([
                next_note,
                next_velocity,
                next_time,
                is_note_on
            ])
            current_time += next_time
            
            # Update input sequence
            new_message = torch.tensor([[next_note, next_velocity, next_time, is_note_on]], 
                                     dtype=torch.float32).unsqueeze(0).to(device)
            current_seq = torch.cat([current_seq[:, 1:], new_message], dim=1)
    
    return generated_messages

# %%
# Generate 10 different songs with varying temperatures and different seeds
import random

# Create a list of potential seed sequences from different tracks
seed_sequences = []
for track in all_tracks[:20]:  # Use first 20 tracks for seeds
    sequence = []
    for msg in track.messages[:32]:
        sequence.append([
            msg.note,
            msg.velocity,
            msg.delta_time,
            1.0 if msg.event else 0.0  # 1.0 for note_on, 0.0 for note_off
        ])
    if len(sequence) >= 16:  # Only use sequences that are long enough
        seed_sequences.append(sequence)

print(f"Created {len(seed_sequences)} potential seed sequences")

# More varied temperatures
temperatures = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0]

for i, temp in enumerate(temperatures):
    print(f"\nGenerating song {i+1} with temperature {temp}")
    
    # Randomly select a seed sequence
    seed_sequence = random.choice(seed_sequences)
    print(f"Using seed sequence of length {len(seed_sequence)}")
    
    # Generate new sequence
    generated_messages = generate_midi(model, seed_sequence, max_time=10.0, temperature=temp)
    print(f"Generated {len(generated_messages)} messages")
    
    # Convert generated messages to MIDI file
    messages = []
    current_time = 0.0
    
    for note, velocity, delta_time, is_note_on in generated_messages:
        messages.append(MidiMessage(
            event=bool(is_note_on),
            note=int(note),
            velocity=int(velocity),
            delta_time=delta_time
        ))
        current_time += delta_time
    
    print(f"Created {len(messages)} MIDI messages")
    print(f"Total duration: {current_time:.2f} seconds")
    
    # Create and save MIDI file
    track_obj = MidiTrack(messages)
    output_file = f'generated_midi_{i+1}.mid'
    track_obj.to_midi_file(output_file)
    
    # Play the generated MIDI file
    midi_data = pretty_midi.PrettyMIDI(output_file)
    audio = midi_data.synthesize(fs=22050)
    print(f"Playing song {i+1}...")
    display(Audio(audio, rate=22050))

# %%
