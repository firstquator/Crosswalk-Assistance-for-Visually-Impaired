import os
import argparse
from gtts import gTTS

parser = argparse.ArgumentParser(description="CREAT VOICE FILES")
parser.add_argument(
    "--text", type=str, help="Enter the text you want to convert to speech."
)
parser.add_argument(
    "--save_path",
    type=str,
    default="../sound",
    help="Path to save voice files.",
)
parser.add_argument(
    "--save_name",
    type=str,
    default="sample",
    help="Name to save",
)
parser.add_argument(
    "--fourcc",
    type=str,
    default=".mp3",
    help="Voice extension name.",
)
parser.add_argument(
    "--lang",
    type=str,
    default="en",
    help="Name to save",
)


args = parser.parse_args()

if not args.text:
    assert "Please enter the text you want to convert to speech using --text."

text = args.text
save_path = os.path.join(args.save_path, args.save_name + args.fourcc)

tts = gTTS(text, lang=args.lang)
tts.save(save_path)

print(f"✅ Success save : {save_path}.")
