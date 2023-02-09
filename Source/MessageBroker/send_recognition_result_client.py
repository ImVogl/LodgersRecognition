import image_msg as image_msg
import recognition_msg as rcg_msg
import zmq
import asyncio

all_processors = {}

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555")
subscribe_message = image_msg.ImageMessage()
subscribe_message.change_to_subscribe()
socket.send_json(subscribe_message.convert_to_json())
response = socket.recv_json()

# Processing images
def process_image(response: dict):
    response_message = image_msg.ImageMessage(response)
    test_object_id = 1
    request_message = rcg_msg.RecognitionResult(test_object_id)
    socket.send_json(request_message.convert_to_json())
    response = socket.recv_json()

all_processors[subscribe_message.id] = process_image

# Main message loop
async def messages_processing():
    while True:
        message = image_msg.ImageMessage()
        socket.send_json(message.convert_to_json())
        response = socket.recv_json()
        if not 'id' in list(response.keys()):
            await asyncio.sleep(0.01)
            continue
        
        if response["id"] in list(all_processors.keys()):
            all_processors[response["id"]](response)
            
        await asyncio.sleep(0.01)

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(messages_processing()))
loop.close()
