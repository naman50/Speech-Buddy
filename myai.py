from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import speech_recognition as sr

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change to a female voice
engine.setProperty('rate', 150)           # Adjust speaking speed
engine.setProperty('volume', 0.9)         # Adjust volume


# Function to speak text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()


# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening for wake word...")
            audio = recognizer.listen(source, timeout=5)  # Timeout after 5 seconds
            text = recognizer.recognize_google(audio).lower()
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            print("Speech recognition service is unavailable.")
            speak_text("Speech recognition service is unavailable.")
            return ""
        except sr.WaitTimeoutError:
            return ""


# Chatbot function
def chatbot():
    print("Voice-Enabled Chatbot (Say 'Hey man' to start talking, and 'exit' to quit)\n")
    speak_text("Hello! Say 'Hey man' to talk to me.")

    chat_history_ids = None

    while True:
        # Listen for the wake word
        wake_word = recognize_speech()
        if "hey man" in wake_word:
            print("Wake word detected! Listening for your input...")
            speak_text("I'm listening!")

            # Listen for the actual input after wake word
            user_input = recognize_speech()
            if not user_input:
                speak_text("I couldn't understand that. Can you try again?")
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                speak_text("Goodbye!")
                break

            # Encode user input and generate response
            input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            chat_history_ids = (
                input_ids if chat_history_ids is None else torch.cat([chat_history_ids, input_ids], dim=-1)
            )
            response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            # Display and speak the response
            print(f"Bot: {response}")
            speak_text(response)


if __name__ == "__main__":
    chatbot()
