import os
import shutil
import tempfile
import streamlit as st
from pytube import YouTube
from pytube.exceptions import RegexMatchError
import re
import openai
#from langchain.llms import OpenAI
import dotenv

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=True)

headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json"
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("YouTube Video Summary")
st.header("Provide youtube url link to summarize")

#link = st.text_input("Enter YouTube URL")
link = st.text_input("YouTube URL Link", "")

def transcribe(audio_file, not_english=False):
    if not os.path.exists(audio_file):
        print('Audio file does not exist!')
        return False
    
    if not_english:  
        # translation to english
        with open(audio_file, 'rb') as f:
            print('Starting translating to English ...', end='')
            transcript = openai.Audio.translate('whisper-1', f)
            print('Done!')
        
    else: # transcription
        with open(audio_file, 'rb') as f:
            print('Starting transcribing ... ', end='')
            transcript = openai.Audio.transcribe('whisper-1', f)
            print('Done!')
        
    name, extension = os.path.splitext(audio_file)
    transcript_filename = f'{name}.txt'
    print(transcript_filename)

    with open(transcript_filename, 'w') as f:
        f.write(transcript['text'])
            
    return transcript_filename

def summarize(transcript_filename):
    import openai 
    import os
    
    if not os.path.exists(transcript_filename):
        print('The transcript file does not exist!')
        return False
    
    with open(transcript_filename) as f:
        transcript = f.read()
        
    system_prompt = 'I want you to act as a Life Coach.'
    prompt = f'''Create a summary of the following text.
    Text: {transcript}
    
    Add a title to the summary.
    Your summary should be informative and factual, covering the most important aspects of the topic.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.'''
    
    print('Starting summarizing ... ', end='')
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=2048,
        temperature=1
    
    )
    
    print('Done')
    r = response['choices'][0]['message']['content']
    return r

def youtube_audio_downloader(link):
    if 'youtube.com' not in link:
        print('Invalid YouTube link!')
        return False
    
    yt = YouTube(link)
    #st.write(f"Downloading '{yt.title}'...")
    audio = yt.streams.filter(only_audio=True).first()
    print('Downloading the audio stream ...', end='')
    
    audio_file = tempfile.NamedTemporaryFile(delete=False,suffix=".mp3")
    
    download_file=audio.download()
    print('yes')
    #audio.write_audiofile(audio_file.name)
    shutil.move(download_file, audio_file.name)

    audio_path = audio_file.name

    print(audio_path)

    transcript_filename=transcribe(audio_path)
    summary=summarize(transcript_filename)

    return summary

if link:
    with st.spinner('Model working ...'):
        summary = youtube_audio_downloader(link)
        st.write('<p style="font-size:26px; color:red;">Summary Text: </p>', unsafe_allow_html=True)
        st.write(summary)
