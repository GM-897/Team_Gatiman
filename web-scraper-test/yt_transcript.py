from pytube import YouTube
import os

# Replace with the URL of the YouTube video you want to download
video_url = 'https://www.youtube.com/watch?v=wamZH-ga0sI'

try:
    # Create a YouTube object
    yt = YouTube(video_url)

    # Get the highest quality audio stream available (MP4 format)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()

    if audio_stream:
        # Download the audio stream
        audio_stream.download()

        # Rename the downloaded file to a more meaningful name
        original_filename = audio_stream.default_filename
        new_filename = 'output.mp3'  # Rename to MP3 format if needed
        os.rename(original_filename, new_filename)

        print(f"Audio download complete: {new_filename}")
    else:
        print("No audio stream found for this video.")

except Exception as e:
    print(f"Error: {str(e)}")
