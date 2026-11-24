from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})

audio_file = client.files.upload(path="/Users/mac/Downloads/mazhar mp3 interview/WhatsApp Audio 2026-11-23 at 17.06.46.mp3")

response = client.interactions.create(
    model="gemini-3.5-transcribe",
    input=[audio_file],
)

print(response.output_text)