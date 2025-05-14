import pretty_midi
import os
import seaborn as sns
import matplotlib.pyplot as plt # For plotting
import numpy as np
import pickle
from IPython.display import Audio
from dataclasses import dataclass

directory = "../Data/melody"
midi_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mid', '.midi'))]

@dataclass
class MidiMessage:
    event: bool  # True for note_on, False for note_off
    note: int
    velocity: int
    dt: int  # delta time in ticks

    @staticmethod
    def from_pretty_midi(note, is_start=True):
        """Convert a pretty_midi Note to MidiMessage.
        
        Args:
            note: pretty_midi.Note object
            is_start: True for note_on, False for note_off
        """
        return MidiMessage(
            event=is_start,
            note=note.pitch,
            velocity=note.velocity,
            dt=int(note.start * 480) if is_start else int(note.end * 480)  # Convert seconds to ticks (assuming 480 ticks per beat)
        )
    
    def to_pretty_midi(self):
        """Convert to pretty_midi Note object.
        
        Returns:
            pretty_midi.Note object
        """
        note = pretty_midi.Note(
            velocity=self.velocity if self.velocity is not None else 0,
            pitch=self.note if self.note is not None else 0,
            start=self.dt / 480 if self.event else None,  # Convert ticks to seconds
            end=None if self.event else self.dt / 480
        )
        return note

