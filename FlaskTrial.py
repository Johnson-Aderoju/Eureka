
import cv2
import SST
import time
import openai
import asyncio
import numpy as np
import mediapipe as mp
from queue import Queue
import tensorflow as tf
from Constant import API_KEY
from threading import Thread, Event
from flask import Flask, jsonify
from flask import request
import numpy as np
import mediapipe as mp
import tensorflow as tf

app = Flask(__name__)
out_q=Queue()
pause_event = Event()

def chatgpt_result(text):
    if "team Eureka" in text:
        return SST.speech_text("Hi Team Eureka, How can I assist you?")
    else:
        chatgpt_result = ''
        openai.api_key = API_KEY
        prompt = text
        model = "text-davinci-003"
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=200)
        generated_text =  response.choices[0].text
        chatgpt_result += generated_text+"\n"
        mytext = chatgpt_result
        
        return mytext

async def model_code(queue):
# initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils
    # Load the gesture recognizer model
    model = tf.keras.models.load_model('mp_hand_gesture')
    print('Success')
    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    start_time=time.time()

    while time.time()-start_time<10:
        # Read each frame from the webcam
        _, frame = cap.read()
        x, y, c = frame.shape
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)
        # print(result)
        className = ''
       # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
       # show the prediction on the frame

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2, cv2.LINE_AA)
        # Show the final output
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        await asyncio.sleep(0)
    # release the webcam and destroy all active windows

    cap.release()
    cv2.destroyAllWindows()
    if className!='':
        await queue.put(className)

async def additional_task(queue):
    message = await queue.get()  # Wait for the message from the queue
    print(f"Received message: {message}")
    SST.speech_text(chatgpt_result(message))


async def main():
    queue = asyncio.Queue()
    opencv_task_coroutine = asyncio.create_task(model_code(queue))
    #asyncio.create_task(welcome_msg(queue))
    additional_task_coroutine = asyncio.create_task(additional_task(queue))
    while not opencv_task_coroutine.done():
        await asyncio.sleep(1)
    # When opencv_task is done, cancel the additional_task
    # additional_task_coroutine.cancel()
    try:
        await additional_task_coroutine
    except asyncio.CancelledError:  
        pass
    await asyncio.gather(opencv_task_coroutine, additional_task_coroutine)
 
# Create a global event loop
loop = asyncio.get_event_loop()


# Define a function to run the asyncio loop in the background
def run_async():
    loop.run_until_complete(main())
    

# Start the background task to run the model_code and additional_task asynchronously
task = loop.create_task(run_async())

# Initialize mediapipe and TensorFlow models outside the endpoint to prevent multiple initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
model = tf.keras.models.load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# Define the endpoint to handle hand gesture recognition requests
@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    # Get the request data
    data = request.json
    message = data.get('message', '')

    # Start hand gesture recognition and GPT-3 processing
    queue = asyncio.Queue()
    opencv_task_coroutine = asyncio.create_task(model_code(queue))
    additional_task_coroutine = asyncio.create_task(additional_task(queue))

    # Wait for the model_code to complete
    loop.run_until_complete(opencv_task_coroutine)

    # Get the result from GPT-3 processing
    while not out_q.empty():
        gpt_result = out_q.get()

    # Cancel the additional_task coroutine
    additional_task_coroutine.cancel()

    # Return the result in JSON format
    return jsonify({
        'gesture_prediction': message,
        'gpt_result': gpt_result
    })
