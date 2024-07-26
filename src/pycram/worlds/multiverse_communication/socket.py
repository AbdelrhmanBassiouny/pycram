#!/usr/bin/env python3

"""Multiverse Client base class."""

import dataclasses
from typing import List, Dict, Callable, TypeVar

from multiverse_client_pybind import MultiverseClientPybind  # noqa

T = TypeVar("T")


@dataclasses.dataclass
class MultiverseMetaData:
    """Meta data for the Multiverse Client, the simulation_name should be non-empty and unique for each simulation"""
    world_name: str = "world"
    simulation_name: str = "cram"
    length_unit: str = "m"
    angle_unit: str = "rad"
    mass_unit: str = "kg"
    time_unit: str = "s"
    handedness: str = "rhs"


class SocketAddress:
    """Socket address for the Multiverse Server and the Multiverse Client"""
    host: str = "tcp://127.0.0.1"
    port: str = ""

    def __init__(self, port: str) -> None:
        self.port = port


class MultiverseClient:
    """Base class for the Multiverse Client"""
    _server_addr: SocketAddress = SocketAddress(port="7000")
    _client_addr: SocketAddress
    _meta_data: MultiverseMetaData
    _multiverse_socket: MultiverseClientPybind
    _start_time: float
    _api_callbacks: Dict[str, Callable[[List[str]], List[str]]]

    def __init__(
            self,
            client_addr: SocketAddress,
            multiverse_meta_data: MultiverseMetaData,
    ) -> None:
        """

        Args:
            client_addr (SocketAddress): The address of the client.
            multiverse_meta_data (MultiverseMetaData): The meta data for the Multiverse Client.
        """
        if not isinstance(client_addr.port, str) or client_addr.port == "":
            raise ValueError(f"Must specify client port for {self.__class__.__name__}.")
        if multiverse_meta_data.simulation_name == "":
            raise ValueError(f"Must specify simulation name.")
        self._send_data = None
        self._client_addr = client_addr
        self._meta_data = multiverse_meta_data
        self._multiverse_socket = MultiverseClientPybind(
            f"{self._server_addr.host}:{self._server_addr.port}"
        )
        self.request_meta_data = {
            "meta_data": self._meta_data.__dict__,
            "send": {},
            "receive": {},
        }
        self._start_time = 0.0

    def loginfo(self, message: str) -> None:
        """Log information.

        Args:
            message (str): The message to log.

        Returns:
            None
        """
        print(message)

    def logwarn(self, message: str) -> None:
        """Warn the user.

        Args:
            message (str): The message to warn about.

        Returns:
            None
        """
        print(message)

    def run(self) -> None:
        """Run the client.

        Returns:
            None
        """
        message = f"[Client {self._client_addr.port}] Start {self.__class__.__name__}{self._client_addr.port}."
        self.loginfo(message)
        self._run()

    def _run(self) -> None:
        """Run the client, should call the _connect_and_start() method. It's left to the user to implement this method
        in threaded or non-threaded fashion.

        Returns:
            None
        """
        self._connect_and_start()

    def stop(self) -> None:
        """Stop the client.

        Returns:
            None
        """
        self._disconnect()

    @property
    def request_meta_data(self) -> Dict:
        """Get the request meta data."""
        return self._request_meta_data

    @request_meta_data.setter
    def request_meta_data(self, request_meta_data: Dict) -> None:
        """Set the request_meta_data, make sure to clear the `send` and `receive` field before setting the request"""
        self._request_meta_data = request_meta_data
        self._multiverse_socket.set_request_meta_data(self._request_meta_data)

    @property
    def response_meta_data(self) -> Dict:
        """Get the response_meta_data."""
        response_meta_data = self._multiverse_socket.get_response_meta_data()
        assert isinstance(response_meta_data, dict)
        if response_meta_data == {}:
            message = f"[Client {self._client_addr.port}] Receive empty response meta data."
            self.logwarn(message)
        return response_meta_data

    def send_and_receive_meta_data(self):
        self._communicate(True)

    def send_and_receive_data(self):
        self._communicate(False)

    @property
    def send_data(self) -> List[float]:
        """Get the send_data."""
        return self._send_data

    @send_data.setter
    def send_data(self, send_data: List[float]) -> None:
        """Set the send_data, the first element should be the current simulation time,
        the rest should be the data to send with the following order:
        double -> uint8_t -> uint16_t"""
        assert isinstance(send_data, list)
        self._send_data = send_data
        self._multiverse_socket.set_send_data(self._send_data)

    @property
    def receive_data(self) -> List[float]:
        """Get the receive_data, the first element should be the current simulation time,
        the rest should be the received data with the following order:
        double -> uint8_t -> uint16_t"""
        receive_data = self._multiverse_socket.get_receive_data()
        assert isinstance(receive_data, list)
        return receive_data

    @property
    def api_callbacks(self) -> Dict[str, Callable[[List[str]], List[str]]]:
        """Get the api_callbacks."""
        return self._api_callbacks

    @api_callbacks.setter
    def api_callbacks(self, api_callbacks: Dict[str, Callable[[List[str]], List[str]]]) -> None:
        """Set the api_callbacks."""
        self._multiverse_socket.set_api_callbacks(api_callbacks)
        self._api_callbacks = api_callbacks

    def _bind_request_meta_data(self, request_meta_data: T) -> T:
        """Bind the request_meta_data before sending it to the server.

        Args:
            request_meta_data (T): The request_meta_data to bind.

        Returns:
            T: The binded request_meta_data.
        """
        pass

    def _bind_response_meta_data(self, response_meta_data: T) -> T:
        """Bind the response_meta_data after receiving it from the server.

        Args:
            response_meta_data (T): The response_meta_data to bind.

        Returns:
            T: The binded response_meta_data.
        """
        pass

    def _bind_send_data(self, send_data: T) -> T:
        """Bind the send_data before sending it to the server.

        Args:
            send_data (T): The send_data to bind.

        Returns:
            T: The binded send_data.
        """
        pass

    def _bind_receive_data(self, receive_data: T) -> T:
        """Bind the receive_data after receiving it from the server.

        Args:
            receive_data (T): The receive_data to bind.

        Returns:
            T: The binded receive_data.
        """
        pass

    def _connect_and_start(self) -> None:
        """Connect to the server and start the client.

        Returns:
            None
        """
        self._multiverse_socket.connect(self._client_addr.host, self._client_addr.port)
        self._multiverse_socket.start()
        self._start_time = self._multiverse_socket.get_time_now()

    def _disconnect(self) -> None:
        """Disconnect from the server.

        Returns:
            None
        """
        self._multiverse_socket.disconnect()

    def _communicate(self, resend_request_meta_data: bool = False) -> bool:
        """Communicate with the server.

        Args:
            resend_request_meta_data (bool): Resend the request meta data.

        Returns:
            bool: True if the communication was successful, False otherwise.
        """
        return self._multiverse_socket.communicate(resend_request_meta_data)

    def _restart(self) -> None:
        """Restart the client.

        Returns:
            None
        """
        self._disconnect()
        self._connect_and_start()

    @property
    def world_time(self) -> float:
        """Get the world time from the server.

        Returns:
            float: The world time.
        """
        return self._multiverse_socket.get_world_time()

    @property
    def sim_time(self) -> float:
        """Get the current simulation time.

        Returns:
            float: The current simulation time.
        """
        return self._multiverse_socket.get_time_now() - self._start_time
