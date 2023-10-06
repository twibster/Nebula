import os, uuid
from midiutil import MIDIFile
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.utils.sonify import (process_frame,
                              sonify_image,
                              write_midi_file,
                              convert_midi_to_mp3,
                              compute_luminance,
                              apply_gamma_correction,
                              sonify_pixel, TEMPO
                              ,check_tuning_file)


app = FastAPI()

utils_dir = "./api/utils/"

@app.post("/sonify")
async def sonfiy(image: UploadFile = File(...), melody: UploadFile = File(None)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/gif", "image/bmp", "image/webp"]:
        raise HTTPException(status.HTTP_409_CONFLICT, "image must be of jpeg, png, jpg, gif, bmp or webp type")
    
    
    if melody:
        if melody.content_type not in ["audio/mid"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "melody must be of midi type")
        with open(utils_dir + melody.filename, "wb") as f:
            melody_contents = await melody.read()
            f.write(melody_contents)
    
    general_name = f"{uuid.uuid4()}"
    image.filename = f"{general_name}.jpg"
    
    with open(utils_dir + image.filename, "wb") as f:
        image_contents = await image.read()
        f.write(image_contents)

    
    down_scaled_image = process_frame(utils_dir+image.filename)
    MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file(utils_dir + melody.filename if melody else "") 
    print(MIN_VOLUME, MAX_VOLUME, note_midis)       
    midi_file = sonify_image(down_scaled_image, MIN_VOLUME, MAX_VOLUME, note_midis)
    write_midi_file(midi_file, utils_dir + general_name)
    convert_midi_to_mp3(utils_dir+f"{general_name}.mid", utils_dir+ "sound-font.sf2", utils_dir+ f"{general_name}.mp3")
    
    return FileResponse(utils_dir+ f"{general_name}.mp3")

        
@app.post("/color_tone")
async def get_color_tone(rgb: tuple[int, int, int]):
    
    # if melody:
    #     if melody.content_type not in ["audio/mid"]:
    #         raise HTTPException(status.HTTP_409_CONFLICT, "melody must be of midi type")
    #     with open(utils_dir + melody.filename, "wb") as f:
    #         melody_contents = await melody.read()
    #         f.write(melody_contents)
    
    # MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file(utils_dir + melody.filename if melody else "") 
    MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file("") 
    
    
    luminance = compute_luminance(rgb)
    pitch, duration, volume = sonify_pixel(rgb, luminance, MAX_VOLUME, MAX_VOLUME, note_midis)
    midi_filename = str(rgb[0]) + "-" + str(rgb[1]) + "-" + str(rgb[2])
    midi_file = MIDIFile(1)
    midi_file.addTempo(track=0, time=0, tempo=TEMPO)  # add midi notes
    midi_file.addNote(
                track=0,
                channel=0,
                time=0,
                pitch=pitch,
                volume=volume,
                duration=duration,
            )
    write_midi_file(midi_file, utils_dir+ midi_filename)
    convert_midi_to_mp3(utils_dir+ f"{midi_filename}.mid", utils_dir+ "sound-font.sf2", utils_dir+ f"{midi_filename}.mp3")
    
    return FileResponse(utils_dir+ f"{midi_filename}.mp3")
    