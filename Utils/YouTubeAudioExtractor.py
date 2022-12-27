import urllib
from moviepy.editor import *

class YouTubeAudioExtractor:
    def __init__(self, url):
        self.url = url

    def download_video(self):
        # Download the video and save it to a temporary file
        video_file, _ = urllib.request.urlretrieve(self.url)
        return video_file

    def extract_audio(self, video_file, audio_file):
        # Load the video file and extract the audio for the first 30 seconds
        audio = AudioFileClip(video_file).subclip(0, 30)

        # Save the audio as a .wav file
        audio.write_audiofile(audio_file)

    def download_and_extract(self, output_file):
        # Download the video
        video_file = self.download_video()

        # Extract the audio and save it as a .wav file
        self.extract_audio(video_file, output_file)

# Create an instance of the class with a YouTube URL
extractor = YouTubeAudioExtractor("https://www.youtube.com/watch?v=vmDDOFXSgAs")

# Call the download_and_extract method to download the video and extract the audio
extractor.download_and_extract("output.wav")

