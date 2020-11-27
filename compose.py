import torch
from rnn_model import PianoBox
import pretty_midi
from note_process import extract_sorted_notes, convert_to_trainable_notes, convert_input_record, Note
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'piano_model.pth'
model = PianoBox(256, 88).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

group_dic = {
    0: [i for i in range(21, 24)],
    1: [i for i in range(24, 36)],
    2: [i for i in range(36, 48)],
    3: [i for i in range(48, 60)],
    4: [i for i in range(60, 72)],
    5: [i for i in range(72, 84)],
    6: [i for i in range(84, 96)],
    7: [i for i in range(96, 108)],
    8: [i for i in range(108, 109)],
}

def convert_to_tensor(notes):
    pitches = []
    olvs = []
    for no in notes:
        pitch = no[0]
        offset = no[1]
        pitches.append(pitch)
        olvs.append(offset)
    pitches = torch.tensor(pitches).long().reshape(1, -1).to(device)
    olvs = torch.tensor(olvs).float().reshape(1, -1).to(device)
    return pitches, olvs

def predict_one_note(notes, h_init=None, pre_note=None):
    pre_offset = 0.0
    if pre_note is not None:
        pre_offset = notes[0].start-pre_note.start
    last_note = notes[-1]
    notes = convert_to_trainable_notes(notes, offset=pre_offset)
    pitches, olvs = convert_to_tensor(notes)
    pitch_prob, olv_vec, h9 = model(pitches, olvs, h_init)
    pitch_prob = pitch_prob.detach().cpu().numpy()[0]
    olv_vec = olv_vec.detach().cpu().numpy()[0]
    pitch = np.argmax(pitch_prob)+21
    offset = olv_vec
    if offset>0.01:
        offset = 0.2
    velocity = 125
    offset = last_note.start+offset
    offset_end = offset+0.8
    next_note = pretty_midi.Note(velocity, pitch, offset, offset_end)
    return next_note, h9

def compose_music(init_notes, number, out_path):
    new_notes = init_notes.copy()
    h_init = None
    pre_note = None
    for i in range(number):
        note, h_init = predict_one_note(init_notes, h_init=h_init, pre_note=pre_note)
        new_notes.append(note)
        init_notes.append(note)
        pre_note = init_notes[0]
        init_notes = init_notes[1:]
        if random.random()>0.95:
            notes = new_notes[:20]
            random.shuffle(notes)
            note = notes[-1]
            start = init_notes[-1].start+0.2
            note.start = start
            note.end = start+0.8
            init_notes.append(note)
            pre_note = init_notes[1]
            init_notes = init_notes[-20:]
        print('predict', i, 'notes.')
    for n in new_notes:
        print(n)
    save_midi(new_notes, out_path)

def save_midi(notes, out_path):
    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    piano.notes = notes
    new_midi.instruments.append(piano)
    new_midi.write(out_path)
    print('write to', out_path)
    

if __name__ == "__main__":
    midi_path = 'pathet3.mid'
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = extract_sorted_notes(midi)
    compose_music(notes[0:0+20], 400, 'out_midi.mid')



