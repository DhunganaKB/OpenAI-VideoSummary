from moviepy.editor import *
from pydub import AudioSegment
#from io import BytesIO
import openai
#import json
import numpy as np
import os
import streamlit as st
import tempfile
import moviepy.editor as mp
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title('Video Summarization')
original_text = '<p style="font-family:Courier; color:Blue; font-size: 20px;">upload mp4 file to extract summary</p>'
st.markdown(original_text, unsafe_allow_html=True)

#st.header('Please, upload mp4 video for analysis here')

#### environment variable
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=True)

headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json"
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

## Reading input mp4 file and converting it into .mp3 file
uploaded_file = st.sidebar.file_uploader("Upload your mp4 file", type="mp4")

def get_summary(combined_txt):

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.create_documents([combined_txt])

    #####map_reduce wich Custom Prompts

    map_prompt = '''
    Write an extended summary of the following:
    Text: `{text}`
    CONCISE SUMMARY:
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )

    combine_prompt = '''
    Write a extended summary of the following text that covers the key points.
    Add a title to the summary.
    Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
    Text: `{text}`
    '''
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    output = summary_chain.run(chunks)
    st.header('Summary Report')
    st.text(output)

if uploaded_file is not None:
    st.write('Waiting time depends upon the size of mp4 file')
    # Create a temporary file to store the uploaded MP4
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_filename = temp_file.name

    # Save the uploaded MP4 to the temporary file
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Convert the MP4 to MP3 using moviepy
    video = mp.VideoFileClip(temp_filename)
    audio = video.audio

    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio.write_audiofile(audio_file.name)

    audio_path = audio_file.name

    # Transcribe the audio
    audio_segment = AudioSegment.from_mp3(audio_path)
    segment_duration = 8 * 60 * 1000  # 10 minutes in milliseconds

    # Read the converted audio file
    audio_bytes = audio_file.read()

    # Display audio player using st.audio
    st.audio(audio_bytes, format='audio/mp3')

    # Create a temporary directory to store audio segments
    temp_dir = tempfile.TemporaryDirectory()

    all_text = []

    #print(len(audio_segment))
    len_set = len(audio_segment)

    combined_txt=''

    for i, start_time in enumerate(range(0, len(audio_segment), segment_duration)):
        if (start_time + segment_duration) > len_set:
            segment = audio_segment[start_time:len_set]
            # print('yes')
            #print(i, start_time, len_set)
        else:
            segment = audio_segment[start_time:start_time + segment_duration]
            #print(i, start_time, (start_time + segment_duration))

        # Save the segment to a temporary file
        segment_path = os.path.join(temp_dir.name, f"segment_{i}.mp3")
        segment.export(segment_path, format="mp3")
        #print(segment_path)

        with open(segment_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            #print(transcript['text'])

        combined_txt = combined_txt + '' + transcript['text']

    with st.spinner('Model working ...'):
        get_summary(combined_txt)

