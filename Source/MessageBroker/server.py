import zmq
import time
import base_msg as base_message

# Messages for subscribers.
messages_queue = {}

# Start server.
def deamon_start(socket):
    while True:
        message = socket.recv_json()
        if message['command'] == str(base_message.CommandType.SUBSCRIBE):
            messages_queue[message['id']] = []
        elif message['command'] == str(base_message.CommandType.SET) and message['id'] in list(messages_queue.keys()):
            messages_queue[message['id']].append(message)
        elif (message['command'] == str(base_message.CommandType.GET) and message['id'] in list(messages_queue.keys())) and len(messages_queue[message['id']]) > 0:
            message = messages_queue[message['id']][-1]
            socket.send_json(message)
            del messages_queue[message['id']][-1]
            continue

        socket.send_json({'result': 'ok'})
        time.sleep(0.1)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
deamon_start(socket)
