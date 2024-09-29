from io import BytesIO

import pygame
from gtts import gTTS

from speaker.speaker import Speaker


class GTTSSpeaker(Speaker):
    def speak(self, message: str, lang: str = 'en'):
        # Initialize pygame mixer
        pygame.mixer.init()

        # Create an in-memory bytes buffer
        mp3_fp = BytesIO()

        # Generate and write MP3 data to the buffer
        tts = gTTS(text=message, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Load the MP3 data from the buffer
        pygame.mixer.music.load(mp3_fp, 'mp3')

        # Play the audio
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
