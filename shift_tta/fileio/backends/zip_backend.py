
import os
from abc import abstractmethod
from zipfile import ZipFile

from pathlib import Path
from typing import Literal, Union

from mmengine.fileio.backends.base import BaseStorageBackend


class ZipBackend(BaseStorageBackend):
    """Backend for loading data from .zip files.

    This backend works with filepaths pointing to valid .zip files. We assume
    that the given .zip file contains the whole dataset associated to this
    backend.
    """

    def __init__(self, zip_path: Union[str, Path] = '') -> None:
        self.zip_path = str(zip_path)
        self._client = None

    def get(self, filepath: str) -> bytes:
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Here, filepath is the zip key.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = TarBackend('path/to/tar')
            >>> backend.get('key')
            b'hello world'
        """
        
        if self._client is None:
            self._client = self._get_client('r')

        filepath = str(filepath)
        try:
            with self._client.open(filepath) as data:
                data = data.read()
        except KeyError as e:
            raise ValueError(
                f"Value '{filepath}' not found in {self._client}!") from e
        return bytes(data)

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError
    
    def _get_client(self, mode: Literal["r", "w", "a", "x"]) -> ZipFile:
        """Get Zip client.

        Args:
            mode (str): Mode to open the file in.

        Returns:
            ZipFile: the zip file.
        """
        assert len(mode) == 1, "Mode must be a single character for zip file."

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(
                f"Corresponding zip file not found:" f" {self.zip_path}")
        
        return ZipFile(self.zip_path, mode)
    
    def __del__(self):
        if self._client is not None:
            self._client.close()