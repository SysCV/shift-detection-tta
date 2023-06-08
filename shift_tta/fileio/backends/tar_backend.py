from pathlib import Path
from typing import Union

from mmengine.fileio.backends.base import BaseStorageBackend


class TarBackend(BaseStorageBackend):

    def __init__(self, tar_path="", **kwargs):
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
        data = self.client.extractfile(filepath)
        data = data.read()
        return data

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self):
        import tarfile
        
        return tarfile.TarFile(self.tar_path)

    def __del__(self):
        if self._client is not None:
            self._client.close()