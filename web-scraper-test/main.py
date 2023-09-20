import sys
import warnings
import whisper
from pathlib import Path
import yt_dlp
import subprocess
import torch
import shutil
import numpy as np

device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

def transcribe(link):
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)

    Model = 'small' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']

    whisper_model = whisper.load_model(Model)



    Type = "Youtube video or playlist" #@param ['Youtube video or playlist', 'Google Drive']
    #URL = "https://www.youtube.com/watch?v=wQPzaxhURTc&ab_channel=ZeeNews" #@param {type:"string"}
    URL=link
    # store_audio = True #@param {type:"boolean"}
    #video_path = "https://www.youtube.com/watch?v=xkk6UCICpRE&ab_channel=ETVAndhraPradesh" #@param {type:"string"}
    video_path_local_list = []

    if Type == "Youtube video or playlist":

        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            # ℹ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([URL])
            list_video_info = [ydl.extract_info(URL, download=False)]

        for video_info in list_video_info:
            video_path_local_list.append(Path(f"{video_info['id']}.wav"))

    elif Type == "Google Drive":
        # video_path_drive = drive_mount_path / Path(video_path.lstrip("/"))
        video_path = ""
        if video_path.is_dir():
            for video_path_drive in video_path.glob("**/*"):
                if video_path_drive.is_file():
                    print(f"**{str(video_path_drive)} selected for transcription.**")
                elif video_path_drive.is_dir():
                    print(f"**Subfolders not supported.**")
                else:
                    print(f"**{str(video_path_drive)} does not exist, skipping.**")
                video_path_local = Path(".").resolve() / (video_path_drive.name)
                shutil.copy(video_path_drive, video_path_local)
                video_path_local_list.append(video_path_local)
        elif video_path.is_file():
            video_path_local = Path(".").resolve() / (video_path.name)
            shutil.copy(video_path, video_path_local)
            video_path_local_list.append(video_path_local)
            print(f"**{str(video_path)} selected for transcription.**")
        else:
            print(f"**{str(video_path)} does not exist.**")

    else:
        raise(TypeError("Please select supported input type."))

    for video_path_local in video_path_local_list:
        if video_path_local.suffix == ".mp4":
            video_path_local = video_path_local.with_suffix(".wav")
            result  = subprocess.run(["ffmpeg", "-i", str(video_path_local.with_suffix(".mp4")), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(video_path_local)])












    language = "Auto detection" #@param ['Auto detection', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese', 'Castilian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'Flemish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Moldavian', 'Moldovan', 'Mongolian', 'Myanmar', 'Nepali', 'Norwegian', 'Nynorsk', 'Occitan', 'Panjabi', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto', 'Romanian', 'Russian', 'Sanskrit', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Sinhalese', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba']
    verbose = 'Live transcription' #@param ['Live transcription', 'Progress bar', 'None']
    output_format = 'all' #@param ['txt', 'vtt', 'srt', 'tsv', 'json', 'all']
    task = 'transcribe' #@param ['transcribe', 'translate']
    temperature = 0.15 #@param {type:"slider", min:0, max:1, step:0.05}
    temperature_increment_on_fallback = 0.2 #@param {type:"slider", min:0, max:1, step:0.05}
    best_of = 5 #@param {type:"integer"}
    beam_size = 8 #@param {type:"integer"}
    patience = 1.0 #@param {type:"number"}
    length_penalty = -0.05 #@param {type:"slider", min:-0.05, max:1, step:0.05}
    suppress_tokens = "-1" #@param {type:"string"}
    initial_prompt = "" #@param {type:"string"}
    condition_on_previous_text = True #@param {type:"boolean"}
    fp16 = True #@param {type:"boolean"}
    compression_ratio_threshold = 2.4 #@param {type:"number"}
    logprob_threshold = -1.0 #@param {type:"number"}
    no_speech_threshold = 0.6 #@param {type:"slider", min:-0.0, max:1, step:0.05}


    verbose_lut = {
        'Live transcription': True,
        'Progress bar': True,
        'None': None
    }

    args = dict(
        language = (None if language == "Auto detection" else language),
        verbose = verbose_lut[verbose],
        task = task,
        temperature = temperature,
        temperature_increment_on_fallback = temperature_increment_on_fallback,
        best_of = best_of,
        beam_size = beam_size,
        patience=patience,
        length_penalty=(length_penalty if length_penalty>=0.0 else None),
        suppress_tokens=suppress_tokens,
        initial_prompt=(None if not initial_prompt else initial_prompt),
        condition_on_previous_text=condition_on_previous_text,
        fp16=fp16,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold
    )

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if Model.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{Model} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    for video_path_local in video_path_local_list:
        print(f"### {video_path_local}")

        video_transcription = whisper.transcribe(
            whisper_model,
            str(video_path_local),
            temperature=temperature,
            **args,
        )

        # Save output
        whisper.utils.get_writer(
            output_format=output_format,
            output_dir=video_path_local.parent
        )(
            video_transcription,
            str(video_path_local.stem),
            options=dict(
                highlight_words=False,
                max_line_count=None,
                max_line_width=None,
            )
        )

        def exportTranscriptFile(ext: str):
            local_path = video_path_local.parent / video_path_local.with_suffix(ext)
            #export_path = video_path_local.parent / video_path_local.with_suffix(ext)
    #         shutil.copy(
    #             local_path,
    #             export_path)

            #display(Markdown(f"**Transcript file created: {export_path}**"))

        if output_format=="all":
            for ext in ('.txt', '.vtt', '.srt', '.tsv', '.json'):
                exportTranscriptFile(ext)
        else:
            exportTranscriptFile("." + output_format)
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import subprocess


# Disable parallelism in Transformers to avoid issues with TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Chrome WebDriver options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')

# Define the sentiment analysis pipeline
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

channel_urls = [
    'https://www.youtube.com/@aajtak/videos',
    'https://www.youtube.com/@zeenews/videos',
    # Add more channel URLs as needed
]

# Check if the CSV file exists, and create it if not
csv_filename = 'yt_results.csv'
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=['Title', 'Sentiment', 'Link'])
    df.to_csv(csv_filename, index=False, encoding='utf-8')

# Initialize the last scraped titles
last_scraped_titles = []

# Set the scraping interval (in seconds)
scraping_interval = 600  # 10 minutes

print("Starting YouTube scraper...")

while True:
    for channel_url in channel_urls:
        print(f"Scraping {channel_url}...")
        # Initialize lists to store scraped data
        titles = []
        links = []

        # Navigate to the YouTube channel URL
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(channel_url)

        # Wait for the page to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.ID, 'video-title-link')))

        # Parse the page source
        content = driver.page_source
        soup = BeautifulSoup(content, features="html.parser")

        # Extract video titles and links
        for element in soup.findAll('a', attrs={'id': 'video-title-link'}):
            title = element.text
            link = element['href']
            link = 'https://www.youtube.com' + link
            titles.append(title)
            links.append(link)

        # Close the WebDriver
        driver.quit()

        # Filter out new titles that were not in the last scrape
        new_titles = list(set(titles) - set(last_scraped_titles))

        # Skip if no new titles were found
        if new_titles == []:
            print("No new videos found")
            continue

        # Perform sentiment analysis on new titles
        sentiments = sentiment_pipeline(new_titles)

        # Update the last scraped titles
        last_scraped_titles.extend(new_titles)

        # Print and update the CSV file
        if new_titles:
            print(f'{len(new_titles)} videos found on {channel_url}:')
            df = pd.DataFrame({'Title': new_titles, 'Sentiment': sentiments, 'Link': links})
            df.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8')

        # Transcribe new videos
        for link in new_titles:
            print(f"Transcribing video: {link}")
            transcribe(link)

    # Wait for the specified interval before the next scrape.
    print(f"Waiting {scraping_interval/60} minutes before next scrape...")
    time.sleep(scraping_interval)
    print("Starting next scrape...")