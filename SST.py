import pyttsx3
import time
import threading

engine = pyttsx3.init()
#Function to voice the text
def speak_text(text):
        engine.say(text)
        engine.runAndWait()

# Function to print text word by word
def print_word_by_word(text):
    words = text.split()
    for word in words:
        print(word, end=' ', flush=True)
        time.sleep(0.5)  # Adjust the sleep time to control the speed of printing

# Threading the speak_text and print_word_by_word
def speech_text(text):
    # Create two threads for concurrent execution
    print_thread = threading.Thread(target=print_word_by_word, args=(text,))
    speak_thread = threading.Thread(target=speak_text, args=(text,))

    # Start the threads
    print_thread.start()
    speak_thread.start()

    # Wait for both threads to finish
    print_thread.join()
    speak_thread.join()
