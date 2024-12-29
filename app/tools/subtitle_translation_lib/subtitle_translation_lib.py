import boto3
import tempfile
import time
import os
import json
import subprocess
from typing import List

# Initialize AWS Clients
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')
translate_client = boto3.client('translate')


# Function to upload a file to S3 with correct MIME type
def upload_file_to_s3(file_bytes, bucket_name, file_path, content_type):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()

    s3_client.upload_file(
        temp_file.name, bucket_name, file_path,
        ExtraArgs={'ContentType': content_type}
    )
    return f"File '{file_path}' uploaded successfully to S3 bucket '{bucket_name}'."


# Function to generate a pre-signed URL for the video in S3
def get_video_url_from_s3(bucket_name, video_filename, expiration=3600):
    video_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': video_filename},
        ExpiresIn=expiration
    )
    return video_url


# Function to extract transcript using AWS Transcribe
def extract_transcript_from_s3(region, bucket_name, video_filename):
    job_name = f"transcribe-{int(time.time())}"
    media_uri = f"s3://{bucket_name}/{video_filename}"

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': media_uri},
        MediaFormat=video_filename.split('.')[-1],
        LanguageCode='en-US',
        OutputBucketName=bucket_name
    )

    # Wait for job completion
    while True:
        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = result['TranscriptionJob']['TranscriptionJobStatus']
        if status in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for Transcription to complete...")
        time.sleep(15)

    if status == 'COMPLETED':
        transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
        return transcript_uri
    else:
        raise Exception("Transcription failed")


# Function to download the transcript, upload it to S3, and return the content
def download_transcript_and_upload(transcript_uri, bucket_name, transcript_filename):
    # Download the transcript
    response = s3_client.get_object(Bucket=transcript_uri.split('/')[-2], Key=transcript_uri.split('/')[-1])
    transcript_json = response['Body'].read().decode('utf-8')

    # Save transcript to S3 in "transcripts" folder
    transcript_path = f"transcripts/{transcript_filename}"
    upload_file_to_s3(transcript_json.encode('utf-8'), bucket_name, transcript_path, 'application/json')

    return transcript_json


# Helper function to chunk text into smaller parts to avoid AWS Translate text size limits
def chunk_text(text, max_size=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to translate transcript using AWS Translate and upload the translation to S3
def translate_transcript_and_upload(text, source_language, target_language, bucket_name, translation_filename):
    translated_text = ""

    # Break the text into chunks that fit within the Translate API limit
    text_chunks = chunk_text(text)

    for chunk in text_chunks:
        response = translate_client.translate_text(
            Text=chunk,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language
        )
        translated_text += response['TranslatedText'] + " "

    # Save translated transcript to S3 in "translations" folder
    translation_path = f"translations/{translation_filename}"
    upload_file_to_s3(translated_text.encode('utf-8'), bucket_name, translation_path, 'text/plain')

    return translated_text


# Function to download video from S3 to local temporary storage
def download_video_from_s3(bucket_name, video_filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        s3_client.download_fileobj(bucket_name, video_filename, temp_video_file)
        return temp_video_file.name


# Function to create a properly timed .srt subtitle file
def create_srt_file(translated_text, total_duration, num_subtitles):
    subtitle_file = tempfile.NamedTemporaryFile(delete=False, suffix=".srt")
    time_per_subtitle = total_duration / num_subtitles
    
    # Split translated text into smaller chunks
    subtitle_lines = translated_text.split('. ')
    
    with open(subtitle_file.name, "w") as f:
        for i, line in enumerate(subtitle_lines):
            start_time = i * time_per_subtitle
            end_time = (i + 1) * time_per_subtitle
            f.write(f"{i+1}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{line.strip()}\n\n")
    
    return subtitle_file.name

# Helper function to format time for .srt file (hours:minutes:seconds,milliseconds)
def format_time(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


# Function to add subtitles to the video using ffmpeg
def add_subtitles_with_ffmpeg(video_path, subtitle_file, output_path):
    """Add subtitles to video using ffmpeg with automatic overwrite"""
    command = [
        "ffmpeg",
        "-y",  # Add this flag to automatically overwrite files
        "-i", video_path,
        "-vf", f"subtitles={subtitle_file}:force_style='Alignment=2'",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path


# Main function to handle the entire video processing flow
def process_video_with_subtitles(region, inbucket, infile, outbucket, outfilename, outfiletype, target_language):
    # Step 1: Download the video from S3
    local_video_path = download_video_from_s3(inbucket, infile)
    
    # Step 2: Extract transcript using AWS Transcribe
    transcript_uri = extract_transcript_from_s3(region, inbucket, infile)
    
    # Step 3: Download transcript, upload to "transcripts" folder, and get transcript content
    transcript_filename = f"{outfilename}.json"
    transcript_data = download_transcript_and_upload(transcript_uri, outbucket, transcript_filename)
    transcript_json = json.loads(transcript_data)
    transcript_text = transcript_json['results']['transcripts'][0]['transcript']
    
    # Step 4: Translate transcript to the selected language and upload to "translations" folder
    translation_filename = f"{outfilename}_{target_language}.txt"
    translated_transcript = translate_transcript_and_upload(transcript_text, 'en', target_language, outbucket, translation_filename)
    
    # Step 5: Get the total duration of the video for subtitle timing
    video = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                            "-of", "default=noprint_wrappers=1:nokey=1", local_video_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    total_duration = float(video.stdout)

    # Step 6: Create SRT file for the translated transcript, using the video duration
    subtitle_file = create_srt_file(translated_transcript, total_duration, len(translated_transcript.split('. ')))
    
    # Step 7: Create a video with subtitles using ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{outfiletype}") as temp_output_file:
        output_file_path = temp_output_file.name

    add_subtitles_with_ffmpeg(local_video_path, subtitle_file, output_file_path)
    
    # Append "_subtitle" to the output filename
    final_output_filename = f"{outfilename}_subtitle.{outfiletype}"
    
    # Step 8: Upload the new video to S3 with "_subtitle" appended to the filename
    with open(output_file_path, "rb") as file:
        video_data = file.read()
    
    upload_status = upload_file_to_s3(video_data, outbucket, final_output_filename, 'video/mp4')
    
    return upload_status
