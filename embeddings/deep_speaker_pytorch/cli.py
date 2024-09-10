from deep_speaker_pytorch.audio import Audio
from deep_speaker_pytorch.batcher import KerasFormatConverter
from deep_speaker_pytorch.constants import NUM_FRAMES

def build_keras_inputs(working_dir, counts_per_speaker="600,100"):
    # counts_per_speaker: If you specify --counts_per_speaker 600,100, that means for each speaker,
    # you're going to generate 600 samples for training and 100 for testing. One sample is 160 frames
    # by default (~roughly 1.6 seconds).
    counts_per_speaker = [int(b) for b in counts_per_speaker.split(',')]
    kc = KerasFormatConverter(working_dir)
    kc.generate(max_length=NUM_FRAMES, counts_per_speaker=counts_per_speaker)
    kc.persist_to_disk()

if __name__=="__main__":
    build_keras_inputs('/mnt/d/Programs/Python/PW/projects/speech/embeddings/working_dir')