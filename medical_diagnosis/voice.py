import gradio as gr
import os
import base64
from openai import OpenAI
import wave

OR_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OR_API_KEY)

# SYSTEM_PROMPT = """
# Act as a brilliant polymath and storyteller. Explain entropy in thermodynamics through the lens of a crumbling ancient library.
# Use the library as an analogy for order/disorder, fading ink for energy dissipation, and remain scientifically accurate.
# """
SYSTEM_PROMPT = """
Act as a brilliant polymath and storyteller. The user is about to ask you anything, answer scientifically accurately and clearly.
"""


class Voice:
    def transcribe_audio(audio_path):
        if not audio_path:
            return "Explain entropy."

        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            response = client.chat.completions.create(
                model="openai/gpt-4o-audio-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please transcribe this audio file.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": "wav",  # Specify the format of your audio file
                                },
                            },
                        ],
                    }
                ],
            )
        print("Transcribed: " + response.choices[0].message.content)
        return response.choices[0].message.content

    def stream_agent_response(self, audio_input, model):
        print(f"Received audio input: {audio_input}, model choice: {model}")
        user_query = self.transcribe_audio(audio_input)
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}],
            stream=True
        )
        full_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                yield full_text # , None
        yield full_text # , generate_speech(full_text)

    def generate_speech(text):
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[{"role": "user", "content": "Say the following: " + text}],
            modalities=["text", "audio"],
            audio={"format": "pcm16", "voice": "alloy"},
            stream=True,
        )

        output_path = "output.wav"

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)

            for chunk in response:
                chunk_dict = chunk.model_dump()

                choices = chunk_dict.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    audio_data = delta.get("audio", {}).get("data")

                    if audio_data:
                        raw_bytes = base64.b64decode(audio_data)
                        wav_file.writeframes(raw_bytes)

        return output_path
