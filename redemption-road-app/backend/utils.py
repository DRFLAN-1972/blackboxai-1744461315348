import os
from pydub import AudioSegment

def process_audio(stems):
    # Process each audio stem and return processed stems
    processed_stems = {}
    
    for stem in stems:
        # Load the audio file
        audio = AudioSegment.from_file(stems[stem])
        # Example processing: apply a fade in effect
        processed_audio = audio.fade_in(2000)  # 2 seconds fade in
        processed_stems[stem] = save_audio_file(f"processed_{stem}", processed_audio.export(format="wav").read())  # Save processed audio
    
    return processed_stems

def save_audio_file(stem_name, audio_data):
    # Save the audio file to the specified path
    file_path = os.path.join("audio", stem_name)
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_data)  # Write actual audio data
    return file_path
