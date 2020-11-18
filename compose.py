import torch
from rnn_model import PianoBox
import pretty_midi
from note_process import extract_sorted_notes, convert_to_trainable_notes, convert_input_record
import numpy as np

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
    pitch = np.argmax(pitch_prob)
    offset = olv_vec[0]
    length = olv_vec[1]
    velocity = olv_vec[2]
    offset = last_note.start+offset*1.0
    offset_end = offset+length*1.5
    velocity = int(velocity*127)
    velocity = 127 if velocity>127 else velocity
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



