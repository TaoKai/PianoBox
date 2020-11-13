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
    }
    rec_str = ''
    for i, pi in enumerate(pieces):
        print('analysis', i, 'piece.')
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
    open('raw_notes.txt', 'w').write(rec_str)
    return dic

if __name__ == "__main__":
    # notes = np.load('notes.npy', allow_pickle=True)
    pieces = read_pretty('midi_classics/Adson')
    dic = get_music_range(pieces)
    for k, v in dic.items():
        print(k, v)