import os, sys
import pretty_midi as pm
import scipy.misc

data = pm.PrettyMIDI('bwv848.mid')

for ins in data.instruments:
    print(ins)

note_dic = {}
# arr = rh.get_piano_roll()

# for i, note in enumerate(rh.notes):
#     print(i, note, note.end - note.start)
#     note.pitch += 2

# scipy.misc.imsave('out.jpg', rh)

new_midi = pm.PrettyMIDI()
# piano = pm.Instrument(program=1)
# for i, t in enumerate(time_list):
#     note = note_dic[t]
#     piano.notes.append(note)
#     print(i, note)
piano = pm.Instrument(program=0)
for ins in data.instruments:
    for no in ins.notes:
        piano.notes.append(no)
new_midi.instruments.append(piano)
for ins in new_midi.instruments:
    for i, no in enumerate(ins.notes):
        # print(i, no)
        if no.start in note_dic.keys():
            print(no)
        note_dic[no.start] = no
new_midi.write('out_midi.mid')

