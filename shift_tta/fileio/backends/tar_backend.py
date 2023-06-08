from pathlib import Path
from typing import Union
from tarfile import TarFile

import os

from mmengine.fileio.backends.base import BaseStorageBackend


class TarBackend(BaseStorageBackend):
    """Backend for loading data from .tar files.

    This backend works with filepaths pointing to valid .tar files. We assume
    that the given .tar file contains the whole dataset associated to this
    backend.
    """

    def __init__(self, tar_path='', **kwargs):
        self.tar_path = str(tar_path)
        self._client = None

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Here, filepath is the tar key.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = TarBackend('path/to/tar')
            >>> backend.get('key')
            b'hello world'
        """
        if self._client is None:
            self._client = self._get_client()

        filepath = str(filepath)
        try:
            with self._client.extractfile(filepath) as data:
                data = data.read()
        except KeyError as e:
            raise ValueError(
                f"Value '{filepath}' not found in {self._client}!") from e
        return data

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self) -> TarFile:
        """Get Tar client.

        Returns:
            TarFile: the tar file.
        """

        if not os.path.exists(self.tar_path):
            raise FileNotFoundError(
                f"Corresponding tar file not found:" f" {self.tar_path}")

        return TarFile(self.tar_path)

    def __del__(self):
        if self._client is not None:
            self._client.close()