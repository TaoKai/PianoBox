import pretty_midi
import os, sys
import numpy as np
import json
import random
import torch
from codecs import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_midi(path):
    files = []
    for fn in os.listdir(path):
        if '.mid' in fn:
            files.append(path+'/'+fn)
    return files

def sort_by_pitch(chord_list):
    if len(chord_list)<=1:
        return chord_list
    pitch_dic = {}
    pitch_list = []
    for c in chord_list:
        pit = c.pitch
        if pit not in pitch_dic.keys():
            pitch_dic[pit] = [c]
            pitch_list.append(pit)
        else:
            pitch_dic[pit].append(c)
    pitch_list.sort()
    sort_chords = []
    for p in pitch_list:
        sort_chords += pitch_dic[p]
    return sort_chords
    

def extract_sorted_notes(midi):
    note_dic = {}
    start_list = []
    for ins in midi.instruments:
        if ins.is_drum:
            continue
        for i, no in enumerate(ins.notes):
            if no.start in note_dic.keys():
                note_dic[no.start].append(no)
            else:
                note_dic[no.start] = [no]
                start_list.append(no.start)
    start_list.sort()
    notes = []
    for s in start_list:
        nos = note_dic[s]
        nos = sort_by_pitch(nos)
        notes.append(nos)
    return notes

def read_pretty(path):
    midis = list_midi(path)
    name = path.split('/')[-1]
    music_pieces = []
    for i, mp in enumerate(midis):
        try:
            mid = pretty_midi.PrettyMIDI(mp)
            notes = extract_sorted_notes(mid)
            music_pieces.append(notes)
            print(i, 'extract', mp)
        except Exception as e:
            # print(str(e))
            continue
    return music_pieces

def get_map_index(pieces):
    notes_map = {}
    pieces_extract = []
    id_cnt = 0
    for i, p in enumerate(pieces):
        offsets = []
        str_notes = []
        for j, nos in enumerate(p):
            if j == 0:
                offsets.append('0.0')
            else:
                pre_notes = p[j-1]
                now_off = nos[0].start-pre_notes[0].start
                offsets.append(str(now_off))
            chord = []
            for note in nos:
                if note.pitch not in chord:
                    chord.append(note.pitch)
            chord_str = json.dumps(chord)
            if chord_str not in notes_map.keys():
                notes_map[chord_str] = id_cnt
                id_cnt += 1
            str_notes.append(notes_map[chord_str])
        pi = [str_notes, offsets]
        pieces_extract.append(pi)
        print(i, 'pieces extracted.')
    all_dic = {'data':pieces_extract, 'map':notes_map}
    json_str = json.dumps(all_dic)
    open('raw_pieces.json', 'w', 'utf-8').write(json_str)
            
class Note(object):
    def __init__(self, path):
        self.data = json.loads(open(path, 'r', 'utf-8').read())
        self.pieces = self.data['data']
        self.chord_map = self.data['map']
        self.batch_size = len(self.pieces)
        self.cursors = list(np.zeros(self.batch_size, dtype=np.int32))
        self.piece_lens = [len(pi[0]) for pi in self.pieces]
    
    def next(self):
        note_inputs = []
        off_inputs = []
        note_labels = []
        off_labels = []
        mask = list(np.ones(self.batch_size, dtype=np.float32))
        


if __name__ == "__main__":
    # path = 'midi_classics/Bach'
    # pieces = read_pretty(path)
    # get_map_index(pieces)
    note = Note('raw_pieces.json')

    