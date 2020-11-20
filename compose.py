import torch
from rnn_model import PianoBox
import pretty_midi
from note_process import extract_sorted_notes, convert_to_trainable_notes, convert_input_record
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'piano_model.pth'
model = PianoBox(512).to(device)
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

def predict_one_note(notes, h_init=None):
    last_note = notes[-1]
    assert len(notes)==12
    notes = convert_to_trainable_notes(notes)
    pitches, olvs = convert_input_record(notes)
    pitches = torch.from_numpy(pitches).reshape([1, 12]).to(device)
    olvs = torch.from_numpy(olvs).reshape([1, 12, 3]).to(device)
    pitch_prob, olv_vec, h9 = model(pitches, olvs, h_init)
    pitch_prob = pitch_prob.detach().cpu().numpy()[0]
    olv_vec = olv_vec.detach().cpu().numpy()[0]
    pitch = np.argmax(pitch_prob)+21
    offset = olv_vec[0]
    length = olv_vec[1]
    velocity = olv_vec[2]
    if offset>0.01:
        offset = 0.2
    offset = last_note.start+offset*1.0
    offset_end = offset+0.5
    velocity = int(velocity*127)
    velocity = 127 if velocity>127 else velocity
    next_note = pretty_midi.Note(velocity, pitch, offset, offset_end)
    return next_note, h9

def compose_music(init_notes, number, out_path):
    new_notes = init_notes.copy()
    h_init = None
    for i in range(number):
        note, h_init = predict_one_note(init_notes, h_init=None)
        new_notes.append(note)
        init_notes.append(note)
        init_notes = init_notes[1:]
        if random.random()>0.95:
            notes = new_notes[:12]
            random.shuffle(notes)
            note = notes[-1]
            start = init_notes[-1].start+0.2
            note.start = start
            note.end = start+0.5
            init_notes.append(note)
        init_notes = init_notes[-12:]
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
    midi_path = 'mozk310a.mid'
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = extract_sorted_notes(midi)
    init_notes = []
    note0 = pretty_midi.Note(100, 60, 0, 0.2)
    note1 = pretty_midi.Note(100, 64, 0.2, 0.2+0.2)
    note2 = pretty_midi.Note(100, 62, 0.4, 0.4+0.2)
    note3 = pretty_midi.Note(100, 65, 0.6, 0.6+0.2)
    note4 = pretty_midi.Note(100, 64, 0.8, 0.8+0.2)
    note5 = pretty_midi.Note(100, 67, 1.0, 1.0+0.2)
    note6 = pretty_midi.Note(100, 65, 1.2, 1.2+0.2)
    note7 = pretty_midi.Note(100, 69, 1.4, 1.4+0.2)
    note8 = pretty_midi.Note(100, 67, 1.6, 1.6+0.2)
    note9 = pretty_midi.Note(100, 66, 1.8, 1.8+0.2)
    note10 = pretty_midi.Note(100, 65, 2.0, 2.0+0.2)
    note11 = pretty_midi.Note(100, 62, 2.2, 2.2+0.2)
    init_notes.append(note0)
    init_notes.append(note1)
    init_notes.append(note2)
    init_notes.append(note3)
    init_notes.append(note4)
    init_notes.append(note5)
    init_notes.append(note6)
    init_notes.append(note7)
    init_notes.append(note8)
    init_notes.append(note9)
    init_notes.append(note10)
    init_notes.append(note11)
    compose_music(notes[0:0+12], 400, 'out_midi.mid')



