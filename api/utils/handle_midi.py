from mido import MidiFile

def parse_musical_notes(midi_file_path):
    midi_file = MidiFile(midi_file_path)
    pitches = set()
    for msg in midi_file:
        if msg.type in ['note_on', "note_off"]:
            pitch = msg.note
            pitches.add(pitch)
    return list(pitches)

def get_min_max_vol(midi_file_path):
    midi_file= MidiFile(midi_file_path)
    min_velo = float("inf") 
    max_velo = float('-inf')  
    for msg in midi_file:
        if msg.type in ['note_on', "note_off"]:
            # print(msg.velocity)
            velo=msg.velocity
            min_velo = min(min_velo, velo)
            max_velo=max(max_velo,velo)
    return min_velo,max_velo        

def get_tempo(midi_file_path):
    midi_file = MidiFile(midi_file_path)
    for msg in midi_file:
        if msg.type == 'set_tempo':
            microseconds_per_quarter_note = msg.tempo
            tempo_bpm = 60000000 / microseconds_per_quarter_note
            return tempo_bpm
    return 52  



