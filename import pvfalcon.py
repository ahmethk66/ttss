import pvfalcon
import whisper

# Load Whisper model
model = whisper.load_model("medium")

# Transcribe audio file
result = model.transcribe("C:\\Users\\ahmet\\Desktop\\cagri.kayitlar\\cagri.kaydi1.wav")
transcript_segments = result["segments"]

# Initialize Falcon for speaker diarization
falcon = pvfalcon.create(access_key="1z+2JZCl6TRLr/tLmry44gn0d6OztCJshQ7vkfDo46JjZ2IjnuO7hw==")
speaker_segments = falcon.process_file("C:\\Users\\ahmet\\Desktop\\cagri.kayitlar\\cagri.kaydi1.wav")

# Define a function to calculate segment overlap
def segment_score(transcript_segment, speaker_segment):
    transcript_segment_start = transcript_segment["start"]
    transcript_segment_end = transcript_segment["end"]
    speaker_segment_start = speaker_segment.start_sec
    speaker_segment_end = speaker_segment.end_sec

    overlap = min(transcript_segment_end, speaker_segment_end) - max(transcript_segment_start, speaker_segment_start)
    overlap_ratio = overlap / (transcript_segment_end - transcript_segment_start)
    return overlap_ratio

# Create a mapping for speaker tags to numerical labels
speaker_mapping = {}
speaker_count = 2

# Dictionary to store speaker segments with their assigned labels
speaker_dict = {}

# Loop through speaker segments and assign labels
for s_segment in speaker_segments:
    speaker_tag = s_segment.speaker_tag
    if speaker_tag not in speaker_mapping:
        speaker_mapping[speaker_tag] = f"Speaker{speaker_count}"
        speaker_count = 2
    speaker_dict[s_segment] = speaker_mapping[speaker_tag]

# Loop through transcript segments to match with speaker segments
for t_segment in transcript_segments:
    max_score = 0
    best_s_segment = None
    for s_segment in speaker_segments:
        score = segment_score(t_segment, s_segment)
        if score > max_score:
            max_score = score
            best_s_segment = s_segment

    if best_s_segment:
        speaker_label = speaker_dict.get(best_s_segment, "Unknown Speaker")
        print(f"{speaker_label}: {t_segment['text']}")
    else:
        print("Unknown speaker: {t_segment['text']}")
