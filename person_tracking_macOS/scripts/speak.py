import speech_recognition as sr
import sys
import pyttsx3
import os

def list_microphones():
    """List all available microphones"""
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone with name \"{name}\" found for `device_index={index}`")

def speak(text):
    """Use macOS built-in voices to speak the provided text"""
    # Set Siri-like voice (Samantha is a built-in macOS voice)
    os.system(f"say -v Samantha '{text}'")

def main():
    recognizer = sr.Recognizer()
    
    print("Available microphones:")
    list_microphones()
    
    # Speak a message to the user
    #speak("Available microphones listed. Starting speech recognition.")
    
    # Try to use default microphone
    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            speak("Adjusting for ambient noise. Please wait.")
            
            print("\nStarting speech recognition. Press Ctrl+C to exit.")
            speak("Starting speech recognition. Please say something.")
            while True:
                print("\nListening...")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    print("Processing audio...")
                    text = recognizer.recognize_google(audio)
                    print(f"Transcribed: {text}")
                    speak(f"Transcribed text: {text}")
                    
                    # Check if "yes" is said
                    if 'yes' in text.lower():
                        print("Recognized 'yes'! Returning 'succeeded'.")
                        speak("Recognized yes! Returning succeeded.")
                        return 'succeeded'
                
                except sr.WaitTimeoutError:
                    print("No speech detected")
                    speak("No speech detected.")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    speak("Could not understand your speech.")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    speak(f"Error requesting results: {e}")
                    
    except OSError as e:
        print(f"Error accessing microphone: {e}")
        speak(f"Error accessing microphone: {e}")
        print("\nTry running these commands to fix ALSA issues:")
        print("sudo apt-get install python3-pyaudio")
        print("sudo apt-get install portaudio19-dev")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting speech recognition...")
        #speak("Exiting speech recognition.")
        sys.exit(0)

if __name__ == "__main__":
    result = main()
    if result == 'succeeded':
        print("Speech recognition succeeded.")
        speak("Speech recognition succeeded.")