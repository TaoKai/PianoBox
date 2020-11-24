import torch
from rnn_model import PianoBox
import pretty_midi
from note_process import extract_sorted_notes, convert_to_trainable_notes, convert_input_record, Note
import numpy as np
import random

note_data = Note('raw_pieces.json', 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'piano_model.pth'
model = PianoBox(512, note_data.note_num, note_data.offset_num).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

off_dic = note_data.off_id
off_dic = {int(k):int(v) for k, v in off_dic.items()}
off_reverse = {int(v):int(k) for k, v in off_dic.items()}

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

def convert_to_index(notes):
    pitches = []
    olvs = []
    for no in notes:
        pitch = no[0]
        offset = int(no[1]*100)
        if offset not in off_dic.keys():
            offset = 20
        else:
            offset = off_dic[offset]
        pitches.append(pitch)
        olvs.append(offset)
    return torch.tensor(pitches), torch.tensor(olvs)

def predict_one_note(notes, h_init=None):
    last_note = notes[-1]
    assert len(notes)==12
    notes = convert_to_trainable_notes(notes)
    pitches, olvs = convert_to_index(notes)
    pitches = pitches.reshape(1, 12).to(device)
    olvs = olvs.reshape(1, 12).to(device)
    pitch_prob, olv_vec, h9 = model(pitches, olvs, h_init)
    pitch_prob = pitch_prob.detach().cpu().numpy()[0]
    olv_vec = olv_vec.detach().cpu().numpy()[0]
    pitch = np.argmax(pitch_prob)+21
    offset = off_reverse[np.argmax(olv_vec)]/100.0
    velocity = 125
    offset = last_note.start+offset
    offset_end = offset+0.5
    next_note = pretty_midi.Note(velocity, pitch, offset, offset_end)
    return next_note, h9

def compose_music(init_notes, number, out_path):
    new_notes = init_notes.copy()
    h_init = None
    for i in range(number):
        note, h_init = predict_one_note(init_notes, h_init=h_init)
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
    midi_path = 'bwv773.mid'
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
    compose_music(init_notes[0:0+12], 400, 'out_midi.mid')



