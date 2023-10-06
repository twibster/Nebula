import os
import audiolazy
import cv2 as cv
import numpy as np
from functools import reduce
from midiutil import MIDIFile
from pydub import AudioSegment

TEMPO = 250  # 60 beats per minute
CONCERT_PITCH = 443  # A440 Hz pitch standard

MIN_DURATION = 0.8
MAX_DURATION = 6

AEOLIAN_SCALE = [0, 2, 3, 5, 7, 8, 10]
BLUES_SCALE = [0, 2, 3, 4, 5, 7, 9, 10, 11]
CHROMATIC_SCALE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
DIATONIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
DORIAN_SCALE = [0, 2, 3, 5, 7, 9, 10]
HARMONIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 11]
LYDIAN_SCALE = [0, 2, 4, 6, 7, 9, 11]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MELODIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 9, 10, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
MIXOLYDIAN_SCALE = [0, 2, 4, 5, 7, 9, 10]
NATURAL_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
PENTATONIC_SCALE = [0, 2, 4, 7, 9]

note_names = [
    # "C1",
    # "C2",
    # "G2",
    # "C3",
    # "E3",
    # "G3",
    # "A3",
    # "B3",
    "D4",
    "E4",
    "G4",
    "A4",
    "B4",
    "D5",
    "E5",
    "G5",
    "A5",
    "B5",
    "D6",
    "E6",
    "F#6",
    "G6",
    "A6",
]
# note_names = ['C2','D2','E2','G2','A2',
#              'C3','D3','E3','G3','A3',
#              'C4','D4','E4','G4','A4']



def convert_strnote_to_midi(strnote):
    return audiolazy.str2midi(strnote)

def check_tuning_file(tuning_file):
    if os.path.exists(tuning_file):
        from api.utils.handle_midi import parse_musical_notes, get_min_max_vol, get_tempo
        
        # tuning_file =  "twinkle-twinkle-little-star.mid" # superior till now 
        # tuning_file =  "hungarian.mid" 
        
        MIN_VOLUME, MAX_VOLUME = get_min_max_vol(tuning_file)
        note_midis = parse_musical_notes(tuning_file)
        
        # TEMPO = get_tempo(tuning_file) * 2.5 
        # print(TEMPO)
    else:
        note_midis = [convert_strnote_to_midi(n) for n in note_names]
        MIN_VOLUME = 35
        MAX_VOLUME = 127  # 7bit int max
    return MIN_VOLUME, MAX_VOLUME, note_midis
# MIN_PITCH = min(note_midis)
# MAX_PITCH = max(note_midis)  # 7bit int max


def linearize_gamma_encoded_channel(channel):
    """
    given a gamma encoded channel, linearize
    it (inverse gamma function)
    """
    channel = channel / 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def apply_gamma_correction(brightness):
    """
    apply gamma correction to linearized value
    """
    if brightness <= 0.0031308:
        brightness *= 12.92
    else:
        brightness = 1.055 * brightness ** (1.0 / 2.4) - 0.055

    return round(brightness * 255)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def compute_luminance(pixel):
    red_weight = 0.212655
    green_weight = 0.715158
    blue_weight = 0.072187
    red, green, blue = pixel

    return apply_gamma_correction(
        red_weight * linearize_gamma_encoded_channel(red)
        + green_weight * linearize_gamma_encoded_channel(green)
        + blue_weight * linearize_gamma_encoded_channel(blue)
    )


def compute_perceived_lightness(luminance):
    if luminance <= (216 / 24389):
        return luminance * (24389 / 27)
    return luminance ** (1 / 3) * 116 - 16


def compute_minmax_pixel_values(image):
    """
    computes both minimum and maximum
    [luminosity, red, blue] and memorize the results

    """
    luminance_values = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flat_luminance_values = luminance_values.flatten()
    flat_image = image.reshape((image.shape[0] * image.shape[1], 3))
    print(flat_image.shape)
    min_luminance = reduce(lambda x, y: min(x, y), flat_luminance_values, float("inf"))
    max_luminance = reduce(lambda x, y: max(x, y), flat_luminance_values, 0)

    min_red = reduce(lambda x, y: min(x, y[0]), flat_image, float("inf"))
    max_red = reduce(lambda x, y: max(x, y[0]), flat_image, 0)

    min_blue = reduce(lambda x, y: min(x, y[2]), flat_image, float("inf"))
    max_blue = reduce(lambda x, y: max(x, y[2]), flat_image, 0)

    print(max_blue)
    print(max_red)

    print(max_luminance)
    print(min_luminance)


def sonify_pixel(pixel, luminance, MIN_VOLUME, MAX_VOLUME, note_midis):
    red, green, blue = pixel
    # pitch = map_scale(luminance, 0, 255, MIN_PITCH, MAX_PITCH, scale=MAJOR_SCALE)
    # pitch = find_nearest(np.array(note_midis), pitch)
    duration = map_value(red, 0, 255, MIN_DURATION, MAX_DURATION)
    volume = map_value(blue, 0, 255, MIN_VOLUME, MAX_VOLUME)
    pitch = note_midis[round(map_value(luminance, 0, 255, 0, len(note_midis) - 1))]
    return pitch, duration, volume

def map_scale(value, min_value, max_value, min_result, max_result, scale=CHROMATIC_SCALE, key=0):

    value = float(value)  
    normal = (value - min_value) / (max_value - min_value)  
    chromatic_step = normal * (max_result - min_result) + min_result - key
    pitch_row_step = chromatic_step * len(scale) / 12   # note in pitch row
    scale_degree  = int(pitch_row_step % len(scale))    # find index into pitchRow list
    register = int(pitch_row_step / len(scale))    # find pitch register (e.g. 4th, 5th, etc.)
    result = register * 12 + scale[scale_degree]
    result = result + key
    result = int(result)   
    return result


def map_value(value, min_value, max_value, min_result, max_result):
    """
    maps value /array from one range to another
    """
    result = min_result + ((value - min_value) / (max_value - min_value)) * (
        max_result - min_result
    )

    if isinstance(min_result, int) and isinstance(max_result, int):
        return round(result)
    return result


def sonify_image(image, MIN_VOLUME, MAX_VOLUME, note_midis):
    luminance_values = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    midi_file = MIDIFile(1)  # one track
    midi_file.addTempo(track=0, time=0, tempo=TEMPO)  # add midi notes
    time = 0 
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            pitch, duration, volume = sonify_pixel(pixel, luminance_values[y][x], MIN_VOLUME, MAX_VOLUME, note_midis)
            midi_file.addNote(
                track=0,
                channel=0,
                time=time,
                pitch=pitch,
                volume=volume,
                duration=duration,
            )
        time += .9

    return midi_file

def write_midi_file(midi_file,filename):
    with open(filename + ".mid", "wb") as f:
        midi_file.writeFile(f)


def convert_midi_to_frequency(midi):
    """
    converts note midi number to frequency/pitch
    based on 440 Hz tuning A0 Concert std.
    """
    frequency = CONCERT_PITCH * 2 ** ((midi - 69) / 12)
    return round(frequency)


def convert_frequency_to_midi(frequency):
    """
    converts frequency/pitch to note midi number
    based on 440 Hz tuning A0 Concert std.
    """
    midi = 69 + 12 * np.log(frequency / CONCERT_PITCH)
    return round(midi)


def convert_midi_to_mp3(midi_file, soundfont, mp3_file):
    wav_file = mp3_file.replace('.mp3', '.wav')
    os.system(f'fluidsynth -ni {soundfont} {midi_file} -F {wav_file} -r 44100')
    audio = AudioSegment.from_wav(wav_file)
    audio.export(mp3_file, format='mp3')
    os.remove(wav_file)


def extract_frames_from_video(video, range_in_sec, dir_name, sample_rate ,fps=30):
    vidcap = cv.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    frame_num = 0
    start = int(range_in_sec[0]*fps)
    end = int(range_in_sec[1]*fps)
    if not os.path.exists(dir_name): 
        os.mkdir(dir_name)       
    while success:
        if start <= count <= end:
            if ((count-start) % sample_rate) == 0:
                cv.imwrite(f"{dir_name}/frame-{frame_num}.jpg" , image) 
                frame_num += 1  
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        if count == end :
            print("Done")
            return



def process_frame(frame):
    scale = .25 
    image = cv.imread(frame)
    print(image)
    _width  = int(scale * image.shape[0])
    _height = int(scale * image.shape[1])
    image = cv.resize(image,(_width,_height), interpolation=cv.INTER_LINEAR)
    image = cv.transpose(image)
    return image 

if __name__ == "__main__":

    # extract_frames_from_video("flyby-1.mp4",[16,63],"flyby-1-frames", 20)
    image = process_frame(os.getcwd()+"/api/utils/test2.jpg")
    file = sonify_image(image)
    write_midi_file(file, os.getcwd()+"/api/utils/test5")
    convert_midi_to_mp3(os.getcwd()+"/api/utils/test5.mid", os.getcwd()+"/api/utils/sound-font.sf2", os.getcwd()+"/api/utils/test5.mp3")


    # import music21

    # fctr = .5 # scale (in this case stretch) the overall tempo by this factor
    # score = music21.converter.parse('test1.mid')
    # newscore = score.scaleOffsets(fctr).scaleDurations(fctr)

    # newscore.write('midi','song_fast.mid') 

