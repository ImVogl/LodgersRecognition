import image_msg as image_msg
import recognition_msg as rcg_msg
import zmq
import asyncio
import numpy

all_processors = {}

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555")
subscribe_message = rcg_msg.RecognitionResult()
subscribe_message.change_to_subscribe()
socket.send_json(subscribe_message.convert_to_json())
response = socket.recv_json()

# Processing images
def process_responce(response: dict):
    response_message = rcg_msg.RecognitionResult(response)
    print('Recognizer object id is', response['object_id'])

all_processors[subscribe_message.id] = process_responce

# Main message loop
async def messages_processing():
    while True:
        message = rcg_msg.RecognitionResult()
        socket.send_json(message.convert_to_json())
        response = socket.recv_json()
        if not 'id' in list(response.keys()):
            await asyncio.sleep(0.01)
            continue
        
        if response["id"] in list(all_processors.keys()):
            all_processors[response["id"]](response)
            
        await asyncio.sleep(0.01)

# Sending test data
async def send_image():
    await asyncio.sleep(5)
    print('Sending image')
    data = numpy.zeros([100, 100, 3], dtype = int)
    message = image_msg.ImageMessage(data)
    socket.send_json(message.convert_to_json())
    response = socket.recv_json()
    print(response)

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(messages_processing(), send_image()))
loop.close()
