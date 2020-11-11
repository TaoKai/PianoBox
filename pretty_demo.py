import os, sys
import pretty_midi as pm
import scipy.misc

data = pm.PrettyMIDI('mozk310a.mid')

for ins in data.instruments:
    print(ins)
input()
rh = data.instruments[0]
lh = data.instruments[1]

note_dic = {}
for no in rh.notes:
    note_dic[no.start] = no

for no in lh.notes:
    note_dic[no.start] = no

time_list = list(note_dic.keys())
time_list.sort()


# arr = rh.get_piano_roll()

# for i, note in enumerate(rh.notes):
#     print(i, note, note.end - note.start)
#     note.pitch += 2

# scipy.misc.imsave('out.jpg', rh)

new_midi = pm.PrettyMIDI()
piano = pm.Instrument(program=1)
for i, t in enumerate(time_list):
    note = note_dic[t]
    piano.notes.append(note)
    print(i, note)
new_midi.instruments.append(rh)
new_midi.write('out_midi.mid')

