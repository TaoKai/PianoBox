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

def read_pretty(path):
    midis = list_midi(path)
    name = path.split('/')[-1]
    music_pieces = []
    for i, mp in enumerate(midis):
        try:
            mid = pretty_midi.PrettyMIDI(mp)
            notes = extract_sorted_notes(mid)
            trainable_notes = convert_to_trainable_notes(notes)
            music_pieces.append(trainable_notes)
            print(i, 'extract', mp)
        except Exception as e:
            # print(str(e))
            continue
    return music_pieces

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
        notes += nos
    return notes

def convert_to_trainable_notes(notes):
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
    trainable_notes = []
    for i, no in enumerate(notes):
        raw_pit = no.pitch-21
        group = 4
        g_pitch = 0
        for k, v in group_dic.items():
            if raw_pit in v:
                group = k
                g_pitch = raw_pit-v[0]
        offset = 0.0
        is_chord = 0
        if i>0:
            pre_note = notes[i-1]
            offset = no.start-pre_note.start
            if offset<=0.0001:
                is_chord = 1
        length = no.end-no.start
        velocity = no.velocity
        info = [raw_pit, offset, length, velocity]
        trainable_notes.append(info)
    return trainable_notes

def get_music_range(pieces):
    dic = {
        'group':[999, -1],
        'g_pitch':[999, -1],
        'offset':[999, -1],
        'length':[999, -1],
        'velocity':[999, -1],
        'is_chord':[0, 1],
        'offset_mean':0.0,
        'length_mean':0.0,
    }
    rec_str = ''
    offset_cnt = 0
    length_cnt = 0
    offset_total = 0
    length_total = 0
    offset_ratio = 1.0
    length_ratio = 1.5
    orcnt = 0
    lrcnt = 0
    for i, pi in enumerate(pieces):
        print('analysis', i, 'pieces.')
        for no in pi:
            group = no[0]
            g_pitch = no[1]
            is_chord = no[2]
            offset = no[3]
            length = no[4]
            velocity = no[5]
            rec_str += str(group)+' '+str(g_pitch)+' '+str(is_chord)+' '+str(offset)+' '+str(length)+' '+str(velocity)+'\n'
            if group<dic['group'][0]:
                dic['group'][0] = group
            if group>dic['group'][1]:
                dic['group'][1] = group
            if g_pitch<dic['g_pitch'][0]:
                dic['g_pitch'][0] = g_pitch
            if g_pitch>dic['g_pitch'][1]:
                dic['g_pitch'][1] = g_pitch
            if offset<dic['offset'][0]:
                dic['offset'][0] = offset
            if offset>dic['offset'][1]:
                dic['offset'][1] = offset
            if length<dic['length'][0]:
                dic['length'][0] = length
            if length>dic['length'][1]:
                dic['length'][1] = length
            if velocity<dic['velocity'][0]:
                dic['velocity'][0] = velocity
            if velocity>dic['velocity'][1]:
                dic['velocity'][1] = velocity
            offset_total += offset
            offset_cnt += 1
            length_total += length
            length_cnt += 1
            if offset<=offset_ratio:
                orcnt += 1
            if length<=length_ratio:
                lrcnt += 1
    dic['offset_mean'] = offset_total/offset_cnt
    dic['length_mean'] = length_total/length_cnt
    dic['offset_ratio_count'] = orcnt/offset_cnt
    dic['length_ratio_count'] = lrcnt/length_cnt
    return dic

def convert_input_record(notes):
    offset_div = 1
    length_div = 1.5
    velocity_div = 127
    div_vec = np.array([[offset_div, length_div, velocity_div]], dtype=np.float32)
    notes = np.array(notes, dtype=np.float32)
    olvs = notes[:, 1:]/div_vec
    pitches = notes[:, 0].astype(np.int32)
    return (pitches, olvs)

def convert_label_record(note):
    offset_div = 1
    length_div = 1.5
    velocity_div = 127
    note[1] /= offset_div
    note[2] /= length_div
    note[3] /= velocity_div
    return note

def generate_np_records(pieces):
    pitch_dic = {i:i+21 for i in range(88)}
    off_dic = {}
    off_id = 0
    pitches = []
    olvs = []
    pitch_labels = []
    olv_labels = []
    seq_len = 12
    for j, notes in enumerate(pieces):
        n_len = len(notes) - seq_len
        for i in range(n_len-seq_len):
            nos = notes[i:i+seq_len+1]
            p_input = []
            o_input = []
            for k, no in enumerate(nos):
                pitch = no[0]
                offset = int(no[1]*100)
                if offset not in off_dic:
                    off_dic[offset] = off_id
                    off_id += 1
                if k<seq_len:
                    p_input.append(pitch)
                    o_input.append(off_dic[offset])
                elif k==seq_len:
                    pitch_labels.append(pitch)
                    olv_labels.append(off_dic[offset])
            pitches.append(p_input)
            olvs.append(o_input)
            assert len(p_input)==seq_len
        print('add', j, 'pieces.')
    data = {
        'data':[pitches, olvs, pitch_labels, olv_labels],
        'pitch_id': pitch_dic,
        'offset_id': off_dic
    }
    data_str = json.dumps(data)
    open('raw_pieces.json', 'w', 'utf-8').write(data_str)

class Note(object):
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.cursor = 0
        self.data_path = data_path
        self.dic = json.loads(open(data_path, 'r', 'utf-8').read())
        self.data = self.dic['data']
        self.pitch_id = self.dic['pitch_id']
        self.off_id = self.dic['offset_id']
        self.pitches = np.array(self.data[0], dtype=np.int32)
        self.olvs = np.array(self.data[1], dtype=np.int32)
        self.pitch_labels = np.array(self.data[2], dtype=np.int32)
        self.olv_labels = np.array(self.data[3], dtype=np.int32)
        self.length = self.pitch_labels.shape[0]
        self.indices = list(np.arange(self.length))
        self.note_num = len(list(self.pitch_id.keys()))
        self.offset_num = len(list(self.off_id.keys()))
        random.shuffle(self.indices)
    
    def next(self):
        if self.cursor+self.batch_size<self.length:
            batch_indices = self.indices[self.cursor:self.cursor+self.batch_size]
            pitches = torch.from_numpy(self.pitches[batch_indices]).to(device)
            olvs = torch.from_numpy(self.olvs[batch_indices]).to(device)
            pitch_labels = torch.from_numpy(self.pitch_labels[batch_indices]).to(device)
            olv_labels = torch.from_numpy(self.olv_labels[batch_indices]).to(device)
            self.cursor += self.batch_size
            return [pitches, olvs, pitch_labels, olv_labels]
        else:
            self.cursor = 0
            random.shuffle(self.indices)
            return self.next()

if __name__ == "__main__":
    # pieces = read_pretty('midi_classics/Bach')
    # generate_np_records(pieces)
    note = Note('raw_pieces.json', 32)
    while True:
        p, o, pl, ol = note.next()
        print(note.cursor, p.size(), o.size(), pl.size(), ol.size())