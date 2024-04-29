import functools
import importlib
import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional, Union
from flask_cors import CORS

from mlflow.server import app
from flask import Flask, Response, flash, jsonify, make_response, request
from flask_session import Session
from werkzeug.datastructures import Authorization
from mlflow.utils.proto_json_utils import message_to_json, parse_dict

from .permissions import MANAGE, Permission, get_permission
from .config import read_auth_config
from .keycloakclient import KeycloakClient

from mlflow import MlflowException,  get_experiment
from mlflow.entities import Experiment
from mlflow.store.entities import PagedList
from mlflow.entities.model_registry import RegisteredModel
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils

from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow.server.handlers import (
    _get_model_registry_store,
    _get_request_message,
    _get_tracking_store,
    catch_mlflow_exception,
    get_endpoints,
)

from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    RenameRegisteredModel,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.service_pb2 import (
    CreateExperiment,
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    DeleteTag,
    GetExperiment,
    GetExperimentByName,
    GetMetricHistory,
    GetRun,
    ListArtifacts,
    LogBatch,
    LogMetric,
    LogModel,
    LogParam,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SetExperimentTag,
    SetTag,
    UpdateExperiment,
    UpdateRun,
)

_logger = logging.getLogger(__name__)

auth_config = read_auth_config()


def is_unprotected_route(path: str) -> bool:
    return path.startswith(("/static", "/favicon.ico", "/health"))

def make_basic_auth_response() -> Response:
    res = make_response(
        "You are not authenticated. Please see "
        "https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow "
        "on how to authenticate."
    )
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res

def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res

def _get_request_param(param: str) -> str:
    if request.method == "GET":
        args = request.args
    elif request.method in ("POST", "PATCH", "DELETE"):
        args = request.json
    else:
        raise MlflowException(
            f"Unsupported HTTP method '{request.method}'",
            BAD_REQUEST,
        )

    if param not in args:
        # Special handling for run_id
        if param == "run_id":
            return _get_request_param("run_uuid")
        raise MlflowException(
            f"Missing value for required parameter '{param}'. "
            "See the API docs for more information about request parameters.",
            INVALID_PARAMETER_VALUE,
        )
    return args[param]

def _get_permission_from_experiment_id():
    experiment_id = _get_request_param("experiment_id")
    exp_name = get_experiment(experiment_id).name
    return _get_permission_from_experiment(exp_name)

def _get_permission_from_experiment_name() -> Permission:
    exp_name = _get_request_param("experiment_name")
    return _get_permission_from_experiment(exp_name)

def _get_permission_from_experiment(exp_name: str) -> Permission:
    permissions = authenticate_request().permissions
    try: 
        permission = permissions["Experiments"][exp_name]
        return get_permission(permission)
    except KeyError:
        return get_permission(auth_config.default_permission)

_EXPERIMENT_ID_PATTERN = re.compile(r"^(\d+)/")

def _get_experiment_id_from_view_args():
    if artifact_path := request.view_args.get("artifact_path"):
        if m := _EXPERIMENT_ID_PATTERN.match(artifact_path):
            return m.group(1)
    return None

def _get_permission_from_experiment_id_artifact_proxy() -> Permission:
    if experiment_id := _get_experiment_id_from_view_args():
        exp_name = get_experiment(experiment_id).name
        return _get_permission_from_experiment(exp_name)
    return get_permission(auth_config.default_permission)

#  we need to look at this code again
def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    exp_name = get_experiment(experiment_id).name
    return _get_permission_from_experiment(exp_name)

def _get_permission_from_registered_model_name() -> Permission:
    model_name = _get_request_param("name")
    permissions = authenticate_request().permissions
    try: 
        permission = permissions["RegisteredModels"][model_name]
        return get_permission(permission)
    except KeyError:
        return get_permission(auth_config.default_permission)

def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_read_experiment_by_name():
    return _get_permission_from_experiment_name().can_read


def validate_can_update_experiment():
    return _get_permission_from_experiment_id().can_update


def validate_can_delete_experiment():
    return _get_permission_from_experiment_id().can_delete


def validate_can_manage_experiment():
    return _get_permission_from_experiment_id().can_manage


def validate_can_read_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_read


def validate_can_update_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_update


def validate_can_read_run():
    return _get_permission_from_run_id().can_read


def validate_can_update_run():
    return _get_permission_from_run_id().can_update


def validate_can_delete_run():
    return _get_permission_from_run_id().can_delete


def validate_can_manage_run():
    return _get_permission_from_run_id().can_manage


def validate_can_read_registered_model():
    return _get_permission_from_registered_model_name().can_read


def validate_can_update_registered_model():
    return _get_permission_from_registered_model_name().can_update


def validate_can_delete_registered_model():
    return _get_permission_from_registered_model_name().can_delete


def validate_can_manage_registered_model():
    return _get_permission_from_registered_model_name().can_manage

def validate_can_create_experiment_or_registeredModels():
    return authenticate_request().is_manager
        

def username_is_sender():
    """Validate if the request username is the sender"""
    username = _get_request_param("username")
    sender = authenticate_request().username
    return username == sender

def validate_can_read_user():
    return username_is_sender()

BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    CreateExperiment: validate_can_create_experiment_or_registeredModels,
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment_by_name,
    DeleteExperiment: validate_can_delete_experiment,
    RestoreExperiment: validate_can_delete_experiment,
    UpdateExperiment: validate_can_update_experiment,
    SetExperimentTag: validate_can_update_experiment,
    # Routes for runs
    CreateRun: validate_can_update_experiment,
    GetRun: validate_can_read_run,
    DeleteRun: validate_can_delete_run,
    RestoreRun: validate_can_delete_run,
    UpdateRun: validate_can_update_run,
    LogMetric: validate_can_update_run,
    LogBatch: validate_can_update_run,
    LogModel: validate_can_update_run,
    SetTag: validate_can_update_run,
    DeleteTag: validate_can_update_run,
    LogParam: validate_can_update_run,
    GetMetricHistory: validate_can_read_run,
    ListArtifacts: validate_can_read_run,
    # Routes for model registry
    CreateRegisteredModel: validate_can_create_experiment_or_registeredModels,
    GetRegisteredModel: validate_can_read_registered_model,
    DeleteRegisteredModel: validate_can_delete_registered_model,
    UpdateRegisteredModel: validate_can_update_registered_model,
    RenameRegisteredModel: validate_can_update_registered_model,
    GetLatestVersions: validate_can_read_registered_model,
    CreateModelVersion: validate_can_update_registered_model,
    GetModelVersion: validate_can_read_registered_model,
    DeleteModelVersion: validate_can_delete_registered_model,
    UpdateModelVersion: validate_can_update_registered_model,
    TransitionModelVersionStage: validate_can_update_registered_model,
    GetModelVersionDownloadUri: validate_can_read_registered_model,
    SetRegisteredModelTag: validate_can_update_registered_model,
    DeleteRegisteredModelTag: validate_can_update_registered_model,
    SetModelVersionTag: validate_can_update_registered_model,
    DeleteModelVersionTag: validate_can_delete_registered_model,
    SetRegisteredModelAlias: validate_can_update_registered_model,
    DeleteRegisteredModelAlias: validate_can_delete_registered_model,
    GetModelVersionByAlias: validate_can_read_registered_model,
}

def get_before_request_handler(request_class):
    return BEFORE_REQUEST_HANDLERS.get(request_class)

BEFORE_REQUEST_VALIDATORS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_before_request_handler)
    for method in methods
}

def _is_proxy_artifact_path(path: str) -> bool:
    return path.startswith(f"{_REST_API_PATH_PREFIX}/mlflow-artifacts/artifacts/")

def _get_proxy_artifact_validator(
    method: str, view_args: Optional[Dict[str, Any]]
) -> Optional[Callable[[], bool]]:
    if view_args is None:
        return validate_can_read_experiment_artifact_proxy  # List

    return {
        "GET": validate_can_read_experiment_artifact_proxy,  # Download
        "PUT": validate_can_update_experiment_artifact_proxy,  # Upload
        "DELETE": validate_can_update_experiment_artifact_proxy,  # Delete
    }.get(method)

def authenticate_request() -> Union[Authorization, Response]:
    """Use configured authorization function to get request authorization."""
    auth_func = get_auth_func(auth_config.authorization_function)
    return auth_func()

@functools.lru_cache(maxsize=None)
def get_auth_func(authorization_function: str) -> Callable[[], Union[Authorization, Response]]:
    """Import and return the specified authorization function.

    :param authorization_function: A string of the form "module.submodule:auth_func
    """
    mod_name, fn_name = authorization_function.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)

@catch_mlflow_exception
def _before_request():
    if is_unprotected_route(request.path):
        return

    authorization = authenticate_request()
    if isinstance(authorization, Response):
        return authorization
    elif not isinstance(authorization, Authorization):
        raise MlflowException(
            f"Unsupported result type from {auth_config.authorization_function}: "
            f"'{type(authorization).__name__}'",
            INTERNAL_ERROR,
        )

    # admins don't need to be authorized
    if authenticate_request().is_admin:
        return

    # authorization
    if validator := BEFORE_REQUEST_VALIDATORS.get((request.path, request.method)):
        if not validator():
            return make_forbidden_response()
    elif _is_proxy_artifact_path(request.path):
        if validator := _get_proxy_artifact_validator(request.method, request.view_args):
            if not validator():
                return make_forbidden_response()
            
def set_can_manage_experiment_permission(resp: Response):
    response_message = CreateExperiment.Response()
    parse_dict(resp.json, response_message)
    name = get_experiment(response_message.experiment_id).name
    userid = authenticate_request().user_id
    client = KeycloakClient()
    client.createGroup(group_name=name, parentGroup="Experiments", user_id=userid)
    
def set_can_manage_registered_model_permission(resp: Response):
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    userid = authenticate_request().user_id
    client = KeycloakClient()
    client.createGroup(group_name=name, parentGroup="RegisteredModels", user_id=userid)
            
def filter_search_experiments(resp: Response):
    if authenticate_request().is_admin:
        return
    
    response_message = SearchExperiments.Response()
    parse_dict(resp.json, response_message)
    try:
        perms = (authenticate_request().permissions)["Experiments"]
    except KeyError:
        perms ={}
    can_read = {exp_name : get_permission(permission).can_read for exp_name, permission in perms.items()}
    default_can_read = get_permission(auth_config.default_permission).can_read
    for e in list(response_message.experiments):
        if not can_read.get(e.name, default_can_read):
            response_message.experiments.remove(e)

    # re-fetch to fill max results
    request_message = _get_request_message(SearchExperiments())
    while (
        len(response_message.experiments) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[Experiment] = _get_tracking_store().search_experiments(
            view_type=request_message.view_type,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            filter_string=request_message.filter,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[: request_message.max_results - len(response_message.experiments)]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [
            e.to_proto() for e in refetched if can_read.get(e.name, default_can_read)
        ]
        response_message.experiments.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)

def filter_search_registered_models(resp: Response):
    if authenticate_request().is_admin:
        return
    
    response_message = SearchRegisteredModels.Response()
    parse_dict(resp.json, response_message)
    try: 
        perms = (authenticate_request().permissions)["RegisteredModels"]
    except KeyError:
        perms ={}
    can_read = {model_name : get_permission(permission).can_read for model_name, permission in perms.items()}
    default_can_read = get_permission(auth_config.default_permission).can_read

    # filter out unreadable
    for rm in list(response_message.registered_models):
        if not can_read.get(rm.name, default_can_read):
            response_message.registered_models.remove(rm)
    
    # re-fetch to fill max results
    request_message = _get_request_message(SearchRegisteredModels())
    while (
        len(response_message.registered_models) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[
            RegisteredModel
        ] = _get_model_registry_store().search_registered_models(
            filter_string=request_message.filter,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[
            : request_message.max_results - len(response_message.registered_models)
        ]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [
            rm.to_proto() for rm in refetched if can_read.get(rm.name, default_can_read)
        ]
        response_message.registered_models.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)
        
AFTER_REQUEST_PATH_HANDLERS = {
    CreateExperiment: set_can_manage_experiment_permission,
    CreateRegisteredModel: set_can_manage_registered_model_permission,
    SearchExperiments: filter_search_experiments,
    SearchRegisteredModels: filter_search_registered_models,
}

def get_after_request_handler(request_class):
    return AFTER_REQUEST_PATH_HANDLERS.get(request_class)

AFTER_REQUEST_HANDLERS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_after_request_handler)
    for method in methods
}

@catch_mlflow_exception
def _after_request(resp: Response):
    if 400 <= resp.status_code < 600:
        return resp

    if handler := AFTER_REQUEST_HANDLERS.get((request.path, request.method)):
        handler(resp)
    return resp
            

def create_app(app: Flask = app):
    """
    A factory to enable authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    :return: The app with authentication and authorization enabled.
    """
    _logger.warning(
        "Authentication from Keycloak Enabled."
    )
    # secret key required for flashing
    if not app.secret_key:
        app.secret_key = str(uuid.uuid4())

    CORS(app)

    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)

    app.before_request(_before_request)
    app.after_request(_after_request)

    return app




