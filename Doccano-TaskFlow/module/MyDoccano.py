from doccano_client import DoccanoClient
import pathlib
from typing import Any, Dict, Iterator, List, Literal, Optional

from doccano_client.models.comment import Comment
from doccano_client.models.data_download import Option as DataExportOption
from doccano_client.models.data_upload import Option as DataImportOption
from doccano_client.models.data_upload import Task
from doccano_client.models.example import Example
from doccano_client.models.label import (
    BoundingBox,
    Category,
    Relation,
    Segment,
    Span,
    Text,
)
from doccano_client.models.label_type import PREFIX_KEY, SUFFIX_KEY, LabelType
from doccano_client.models.member import Member
from doccano_client.models.metrics import LabelDistribution, MemberProgress, Progress
from doccano_client.models.project import Project
from doccano_client.models.role import Role
from doccano_client.models.task_status import TaskStatus
from doccano_client.models.user import User
from doccano_client.models.user_details import PasswordUpdated, UserDetails
from doccano_client.repositories.base import BaseRepository
from doccano_client.repositories.comment import CommentRepository
from doccano_client.repositories.data_download import DataDownloadRepository
from doccano_client.repositories.data_upload import DataUploadRepository
from doccano_client.repositories.example import ExampleRepository
from doccano_client.repositories.label import (
    BoundingBoxRepository,
    CategoryRepository,
    RelationRepository,
    SegmentRepository,
    SpanRepository,
    TextRepository,
)
from doccano_client.repositories.label_type import (
    CategoryTypeRepository,
    RelationTypeRepository,
    SpanTypeRepository,
)
from doccano_client.repositories.member import MemberRepository
from doccano_client.repositories.metrics import MetricsRepository
from doccano_client.repositories.project import ProjectRepository
from doccano_client.repositories.role import RoleRepository
from doccano_client.repositories.task_status import TaskStatusRepository
from doccano_client.repositories.user import UserRepository
from doccano_client.repositories.user_details import UserDetailsRepository
from doccano_client.services.label_type import LabelTypeService
from doccano_client.usecase.comment import CommentUseCase
from doccano_client.usecase.data_download import DataDownloadUseCase
from doccano_client.usecase.data_upload import DataUploadUseCase
from doccano_client.usecase.example import ExampleUseCase
from doccano_client.usecase.label import (
    BoundingBoxUseCase,
    CategoryUseCase,
    RelationUseCase,
    SegmentUseCase,
    SpanUseCase,
    TextUseCase,
)
from doccano_client.usecase.label_type import LabelTypeUseCase
from doccano_client.usecase.member import MemberUseCase
from doccano_client.usecase.project import ProjectType, ProjectUseCase
from doccano_client.usecase.user_details import UserDetailsUseCase

class DoccanoObject():
    def __init__(self,DOCCANO_URL,USERNAME,PASSWORD):

        self.client = DoccanoClient(DOCCANO_URL)
        self.client.login(username=USERNAME, password=PASSWORD)

    def list_all_projects(self):
        projects = self.client.list_projects()
        for elem in projects:
            print(elem)

    @staticmethod
    def create_user(self, username: str, password: str):
        """
        创建账户
        :param self:
        :param username:
        :param password:
        :return:
        """
        ...

    @staticmethod
    def create_project(
            self,
            name: str,
            project_type: ProjectType,
            description: str,
            guideline: str = "",
            random_order: bool = False,
            collaborative_annotation: bool = False,
            single_class_classification: bool = False,
            allow_overlapping: bool = False,
            grapheme_mode: bool = False,
            use_relation: bool = False,
            tags: Optional[List[str]] = None,
    ) -> Project:
        """
        创建标注任务
        :param self:
        :param name:
        :param project_type:
        :param description:
        :param guideline:
        :param random_order:
        :param collaborative_annotation:
        :param single_class_classification:
        :param allow_overlapping:
        :param grapheme_mode:
        :param use_relation:
        :param tags:
        :return:
        """
        ...

    # @staticmethod
    # def delete_project(self, project_id: int):
    #     """
    #     删除标注任务
    #     :param self:
    #     :param project_id:
    #     :return:
    #     """
    #     ...
    #
    @staticmethod
    def download(self, project_id: int, format: str, only_approved=False, dir_name=".") -> pathlib.Path:
        ...


    @staticmethod
    def upload(
            self,
            project_id: int,
            file_paths: List[str],
            task: Task,
            format: str,
            column_data: str = "text",
            column_label: str = "label",
    ) -> TaskStatus:
        ...


    @staticmethod
    def add_member(
            self,
            project_id: int,
            username: str,
            role_name: str,
    ) -> Member:
        """Create a new member.

        Args:
            project_id (int): The id of the project.
            username (str): The username of the future member.
            role_name (str): The role of the future member.

        Returns:
            Member: The created member.
        """
        ...


class MyDoccanoClient(DoccanoObject):
    def __init__(self,DOCCANO_URL,USERNAME,PASSWORD):
        super().__init__(
            DOCCANO_URL= DOCCANO_URL,
            USERNAME = USERNAME,
            PASSWORD = PASSWORD
        )

    def create_project(
        self,
        name: str,
        project_type: ProjectType,
        description: str,
        guideline: str = "",
        random_order: bool = False,
        collaborative_annotation: bool = True,
        single_class_classification: bool = False,
        allow_overlapping: bool = False,
        grapheme_mode: bool = False,
        use_relation: bool = False,
        tags: Optional[List[str]] = None,
    ) -> Project:
        """
        Create a new project. `ProjectType` is one of the
        `DocumentClassification`, `SequenceLabeling`, `Seq2seq`, `Speech2text`,
        `ImageClassification`, `BoundingBox`, `Segmentation`, `ImageCaptioning`,
        and `IntentDetectionAndSlotFilling`.

        Args:
            name (str): The name of the project.
            project_type (ProjectType): The type of the project.
            description (str): The description of the project.
            guideline (str): The annotation guideline. Defaults to "".
            random_order (bool): Whether to shuffle the uploaded data. Defaults to False.
            collaborative_annotation (bool): If True, a data can be annotated by multiple users. Defaults to False.
            single_class_classification (bool): If True, only one label can apply a data. Defaults to False.
            allow_overlapping (bool): If True, span overlapping is allowed. Defaults to False.
            grapheme_mode (bool): If True, count multi-byte characters as one character. Defaults to False.
            use_relation (bool): If True, relation labeling is allowed. Defaults to False.
            tags (Optional[List[str]], optional): The tags of the project. Defaults to None.

        Returns:
            Project: The created project.
        """
        return self.client.project.create(
            name=name,
            project_type=project_type,
            description=description,
            guideline=guideline,
            random_order=random_order,
            collaborative_annotation=collaborative_annotation,
            single_class_classification=single_class_classification,
            allow_overlapping=allow_overlapping,
            grapheme_mode=grapheme_mode,
            use_relation=use_relation,
            tags=tags,
        )

    def upload(
        self,
        project_id: int,
        file_paths: List[str],
        task: Task,
        format: str,
        column_data: str = "text",
        column_label: str = "label",
    ):
        """
        Upload a file. `task` is one of the
        `DocumentClassification`, `SequenceLabeling`, `Seq2seq`, `Speech2text`,
        `ImageClassification`, `BoundingBox`, `Segmentation`, `ImageCaptioning`,
        , `IntentDetectionAndSlotFilling`, and `RelationExtraction`.

        Args:
            project_id (int): The id of the project.
            file_paths (List[str]): The list of the file paths.
            task (Task): The task of the upload.
            format (str): The format of the upload.
            column_data (str): The column name of the data.
            column_label (str): The column name of the label.

        Returns:
            TaskStatus: The status of the upload task.
        """
        return self.client.data_import.upload(project_id, file_paths, task, format, column_data, column_label)


    def download(self, project_id: int, format: str, only_approved=False, dir_name=".") -> pathlib.Path:
        """
        Download a file.

        Args:
            project_id (int): The id of the project.
            format (str): The format of the download.
            only_approved (bool): Whether to export approved data only.
            dir_name (str): The directory to save the file.

        Returns:
            pathlib.Path: The path to the downloaded file.
        """
        return self.client.data_export.download(project_id, format, only_approved, dir_name)

    def create_user(self, username: str, password: str):
        return self.client._user_repository.create_user(username=username, password=password)

    def add_member(
        self,
        project_id: int,
        username: str,
        role_name: str,
    ) -> Member:
        """
        Create a new member.

        Args:
            project_id (int): The id of the project.
            username (str): The username of the future member.
            role_name (str): The role of the future member. one of ["annotator","project_admin",""]

        Returns:
            Member: The created member.
        """
        return self.client.member.add(project_id, username, role_name)


