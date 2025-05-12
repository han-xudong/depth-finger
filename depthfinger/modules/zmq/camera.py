#!/usr/bin/env python

import sys
import zmq
import numpy as np
from ..protobuf import cam_msg_pb2


class CameraSubscriber:
    def __init__(self, ip, port, hwm: int = 1, conflate: bool = True) -> None:
        """Subscriber initialization.

        Args:
            addr: The address of the subscriber.
        """

        print(f"Address: tcp://{ip}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{ip}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Use poller to implement timeout
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)

        # Init the message
        self.cam = cam_msg_pb2.Camera()

        print("Package Camera")
        print("Message Camera")
        print("{\n\tbytes img = 1;\n}")

        print("Camera Subscriber Initialization Done.")

    def subscribeMessage(self, timeout=2000):
        """Subscribe the message.

        Args:
            timeout: Maximum time to wait for a message in milliseconds. Default is 2000ms.

        Returns:
            img: The image captured by the camera.

        Raises:
            zmq.ZMQError: If no message is received within the timeout period.
        """
        

        # Wait for message with timeout
        if self.poller.poll(timeout):
            # Receive the message
            self.cam.ParseFromString(self.subscriber.recv())
            return np.frombuffer(self.cam.img, dtype=np.uint8)
        else:
            print("\033[31mTimeout: No image received!\033[0m")
            print("\033[31mPlease check the connection!\033[0m")
            sys.exit()

    def close(self):
        """Close ZMQ socket and context to prevent memory leaks."""
        if hasattr(self, 'subscriber') and self.subscriber:
            self.poller.unregister(self.subscriber)
            self.subscriber.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()
