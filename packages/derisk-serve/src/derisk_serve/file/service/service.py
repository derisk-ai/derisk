import logging
from typing import BinaryIO, List, Optional, Tuple

from fastapi import HTTPException, UploadFile

from derisk.component import SystemApp
from derisk.core.interface.file import FileMetadata, FileStorageClient, FileStorageURI
from derisk.storage.metadata import BaseDao
from derisk.util.tracer import trace
from derisk_serve.core import BaseService

from ..api.schemas import (
    FileMetadataResponse,
    ServeRequest,
    ServerResponse,
    UploadFileResponse,
)
from ..config import SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..models.models import ServeDao, ServeEntity

logger = logging.getLogger(__name__)


class Service(BaseService[ServeEntity, ServeRequest, ServerResponse]):
    """The service class for File"""

    name = SERVE_SERVICE_COMPONENT_NAME

    def __init__(
        self, system_app: SystemApp, config: ServeConfig, dao: Optional[ServeDao] = None
    ):
        self._system_app = None
        self._serve_config: ServeConfig = config
        self._dao: ServeDao = dao
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp) -> None:
        """Initialize the service

        Args:
            system_app (SystemApp): The system app
        """
        super().init_app(system_app)
        self._dao = self._dao or ServeDao(self._serve_config)
        self._system_app = system_app

    @property
    def dao(self) -> BaseDao[ServeEntity, ServeRequest, ServerResponse]:
        """Returns the internal DAO."""
        return self._dao

    @property
    def config(self) -> ServeConfig:
        """Returns the internal ServeConfig."""
        return self._serve_config

    @property
    def file_storage_client(self) -> FileStorageClient:
        """Returns the internal FileStorageClient.

        Returns:
            FileStorageClient: The internal FileStorageClient
        """
        file_storage_client = FileStorageClient.get_instance(
            self._system_app, default_component=None
        )
        if file_storage_client:
            return file_storage_client
        else:
            from ..serve import Serve

            file_storage_client = Serve.get_instance(
                self._system_app
            ).file_storage_client
            self._system_app.register_instance(file_storage_client)
            return file_storage_client

    @trace("upload_files")
    def upload_files(
        self,
        bucket: str,
        storage_type: str,
        files: List[UploadFile],
        user_name: Optional[str] = None,
        sys_code: Optional[str] = None,
    ) -> List[UploadFileResponse]:
        """Upload files by a list of UploadFile."""
        results = []
        for file in files:
            file_name = file.filename
            logger.info(f"Uploading file {file_name} to bucket {bucket}")
            custom_metadata = {
                "user_name": user_name,
                "sys_code": sys_code,
            }
            uri = self.file_storage_client.save_file(
                bucket,
                file_name,
                file_data=file.file,
                storage_type=storage_type,
                custom_metadata=custom_metadata,
            )
            parsed_uri = FileStorageURI.parse(uri)
            logger.info(f"Uploaded file {file_name} to bucket {bucket}, uri={uri}")
            results.append(
                UploadFileResponse(
                    file_name=file_name,
                    file_id=parsed_uri.file_id,
                    bucket=bucket,
                    uri=uri,
                )
            )
        return results

    @trace("download_file")
    def download_file(self, bucket: str, file_id: str) -> Tuple[BinaryIO, FileMetadata]:
        """Download a file by file_id."""
        return self.file_storage_client.get_file_by_id(bucket, file_id)

    def delete_file(self, bucket: str, file_id: str) -> None:
        """Delete a file by file_id."""
        self.file_storage_client.delete_file_by_id(bucket, file_id)

    def get_file_metadata(
        self,
        uri: Optional[str] = None,
        bucket: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Optional[FileMetadataResponse]:
        """Get the metadata of a file by file_id."""
        if uri:
            parsed_uri = FileStorageURI.parse(uri)
            bucket, file_id = parsed_uri.bucket, parsed_uri.file_id
        if not (bucket and file_id):
            raise ValueError("Either uri or bucket and file_id must be provided.")
        metadata = self.file_storage_client.storage_system.get_file_metadata(
            bucket, file_id
        )
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"File metadata not found: bucket={bucket}, file_id={file_id}, "
                f"uri={uri}",
            )
        return FileMetadataResponse(
            file_name=metadata.file_name,
            file_id=metadata.file_id,
            bucket=metadata.bucket,
            uri=metadata.uri,
            file_size=metadata.file_size,
            user_name=metadata.user_name,
            sys_code=metadata.sys_code,
        )
