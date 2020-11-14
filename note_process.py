import pretty_midi
import os, sys
import numpy as np
import json

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
        raw_pit = no.pitch
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
        info = [group, g_pitch, is_chord, offset, length, velocity]
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
    group_div = 8
    pitch_div = 11
    chord_div = 1
    offset_div = 1
    length_div = 1.5
    velocity_div = 127
    div_vec = np.array([[group_div, pitch_div, chord_div, offset_div, length_div, velocity_div]], dtype=np.float32)
    notes = np.array(notes, dtype=np.float32)/div_vec
    return notes

def convert_label_record(note):
    group_div = 8
    pitch_div = 11
    chord_div = 1
    offset_div = 1
    length_div = 1.5
    velocity_div = 127
    note[3] /= offset_div
    note[4] /= length_div
    note[5] /= velocity_div
    return note

def generate_np_records(pieces):
    note_features = []
    group_labels = []
    pitch_labels = []
    chord_labels = []
    regression_labels = []
    seq_len = 12
    for j, notes in enumerate(pieces):
        n_len = len(notes) - seq_len
        for i in range(n_len-seq_len):
            input_notes = notes[i:i+seq_len]
            note = notes[i+seq_len]
            input_notes = convert_input_record(input_notes)
            note = convert_label_record(note)
            note_features.append(input_notes)
            group_labels.append(note[0])
            pitch_labels.append(note[1])
            chord_labels.append(note[2])
            regression_labels.append(note[3:])
        print(j, 'training pieces added.')
    note_features = np.array(note_features, dtype=np.float32)
    group_labels = np.array(group_labels, dtype=np.int32)
    pitch_labels = np.array(pitch_labels, dtype=np.int32)
    regression_labels = np.array(regression_labels, dtype=np.float32)
    np.savez('train_data.npz', note_features=note_features, group_labels=group_labels,
    pitch_labels=pitch_labels, chord_labels=chord_labels, regression_labels=regression_labels)



if __name__ == "__main__":
    # notes = np.load('notes.npy', allow_pickle=True)
    pieces = read_pretty('midi_classics/Adson')
    dic = get_music_range(pieces)
    for k, v in dic.items():
        print(k, v)
    generate_np_records(pieces)