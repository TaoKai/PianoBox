import torch
from rnn_model import PianoBox
import pretty_midi
from note_process import extract_sorted_notes, convert_to_trainable_notes, convert_input_record
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'piano_model.pth'
model = PianoBox(6, 1024).to(device)
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

def predict_one_note(notes):
    last_note = notes[-1]
    assert len(notes)==12
    notes = convert_to_trainable_notes(notes)
    notes = convert_input_record(notes)
    notes = torch.from_numpy(notes).reshape([1, 12, 6]).to(device)
    group_prob, pitch_prob, chord_prob, olv_vec = model(notes)
    group_prob = group_prob.detach().cpu().numpy()[0]
    pitch_prob = pitch_prob.detach().cpu().numpy()[0]
    chord_prob = chord_prob.detach().cpu().numpy()[0]
    olv_vec = olv_vec.detach().cpu().numpy()[0]
    group = np.argmax(group_prob)
    pitch = np.argmax(pitch_prob)
    is_chord = np.argmax(chord_prob)
    offset = olv_vec[0]
    length = olv_vec[1]
    velocity = olv_vec[2]
    true_pitch = group_dic[group][0]+pitch
    if is_chord==1:
        offset = last_note.start
    else:
        offset = last_note.start+offset*1.0
    offset_end = offset+length*1.5
    velocity = int(velocity*127)
    next_note = pretty_midi.Note(velocity, true_pitch, offset, offset_end)
    return next_note

def compose_music(init_notes, number, out_path):
    new_notes = init_notes.copy()
    for i in range(number):
        note = predict_one_note(init_notes)
        new_notes.append(note)
        init_notes.append(note)
        init_notes = init_notes[1:]
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
    init_notes = notes[:12]
    compose_music(init_notes, 500, 'out_midi.mid')



